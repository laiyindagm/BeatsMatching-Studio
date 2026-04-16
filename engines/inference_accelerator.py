"""
engines/inference_accelerator.py
====================================
深度学习推理加速模块 v2

加速策略：
  1. SharedYOLO: 共享 YOLO 实例 + 帧级结果缓存，消除 ViTPose/HMR2 重复检测
  2. True Batch Inference: 多帧 crop 堆叠为一个 batch，单次 model forward
  3. FramePrefetcher: 独立线程预读视频帧，与 GPU 推理流水线并行
  4. torch.compile: PyTorch 2.x 图编译优化（可选，首次需 warmup）
  5. FP16 Autocast: HMR2 半精度推理（ViTPose 因 scipy 后处理不兼容而禁用）

使用方式：
    from engines.inference_accelerator import (
        AcceleratedViTPose, AcceleratedHMR2, SharedYOLO,
        FramePrefetcher, InferenceConfig,
    )

    shared_yolo = SharedYOLO(yolo_model_path)
    acc_vitpose = AcceleratedViTPose(base_detector, config, shared_yolo)
    acc_hmr2    = AcceleratedHMR2(base_reconstructor, config, shared_yolo)
    prefetcher  = FramePrefetcher(video_path, max_queue=32)
"""

from __future__ import annotations

import os
import sys
import time
import threading
import warnings
from typing import Optional, List, Tuple, Any, Dict
from collections import OrderedDict

import numpy as np
import cv2
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


# ══════════════════════════════════════════════════════════════════════════════
#  配置
# ══════════════════════════════════════════════════════════════════════════════

class InferenceConfig:
    """推理加速配置"""
    def __init__(
        self,
        use_fp16: bool = True,
        use_compile: bool = False,
        use_batch: bool = True,
        batch_size: int = 8,
        fp16_enabled_modules: List[str] = None,
        compile_mode: str = "reduce-overhead",
        prefetch_queue_size: int = 32,
        shared_yolo: bool = True,
    ):
        self.use_fp16 = use_fp16
        self.use_compile = use_compile
        self.use_batch = use_batch
        self.batch_size = batch_size
        self.compile_mode = compile_mode
        self.prefetch_queue_size = prefetch_queue_size
        self.shared_yolo = shared_yolo

        if fp16_enabled_modules is None:
            self.fp16_enabled_modules = [
                "*.attention*", "*.mlp*", "*.transformer*",
                "*.encoder*", "*.decoder*",
            ]
        else:
            self.fp16_enabled_modules = fp16_enabled_modules


DEFAULT_CONFIG = InferenceConfig()


# ══════════════════════════════════════════════════════════════════════════════
#  SharedYOLO：共享人体检测 + 帧级缓存
# ══════════════════════════════════════════════════════════════════════════════

class SharedYOLO:
    """
    共享 YOLO 实例 + LRU 帧缓存。

    在联合模式（ViTPose + HMR2）中，同一帧只做一次人体检测，
    第二次请求直接命中缓存，消除 ~3-5ms/帧 的冗余开销。
    """

    _instance: Optional["SharedYOLO"] = None
    _lock = threading.Lock()

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.35,
        bbox_expand: float = 0.10,
        cache_size: int = 256,
    ):
        self._yolo = None
        self._model_path = model_path
        self.conf_threshold = conf_threshold
        self.bbox_expand = bbox_expand
        self._cache: OrderedDict = OrderedDict()
        self._cache_size = cache_size
        self._hits = 0
        self._misses = 0

        try:
            from ultralytics import YOLO as _YOLO
            self._yolo = _YOLO(model_path)
            print(f"[SharedYOLO] 加载成功: {model_path}")
        except Exception as e:
            warnings.warn(f"[SharedYOLO] YOLO 加载失败: {e}")

    @classmethod
    def get_instance(cls, model_path: str, **kwargs) -> "SharedYOLO":
        """获取单例（线程安全）"""
        with cls._lock:
            if cls._instance is None or cls._instance._model_path != model_path:
                cls._instance = cls(model_path, **kwargs)
            return cls._instance

    def detect(
        self,
        image_rgb: np.ndarray,
        frame_idx: int = -1,
    ) -> List[Tuple[float, float, float, float]]:
        """检测人体框，支持帧级缓存。"""
        # 缓存查找
        if frame_idx >= 0 and frame_idx in self._cache:
            self._hits += 1
            return self._cache[frame_idx]

        H, W = image_rgb.shape[:2]
        if self._yolo is None:
            return [(0.0, 0.0, float(W), float(H))]

        results = self._yolo(
            image_rgb, classes=[0], conf=self.conf_threshold, verbose=False,
        )
        boxes = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                dx = (x2 - x1) * self.bbox_expand
                dy = (y2 - y1) * self.bbox_expand
                x1 = max(0.0, x1 - dx)
                y1 = max(0.0, y1 - dy)
                x2 = min(float(W), x2 + dx)
                y2 = min(float(H), y2 + dy)
                boxes.append((x1, y1, x2, y2))

        if not boxes:
            boxes = [(0.0, 0.0, float(W), float(H))]

        # 缓存存储（LRU 淘汰）
        if frame_idx >= 0:
            self._misses += 1
            self._cache[frame_idx] = boxes
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)

        return boxes

    def clear_cache(self):
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    @property
    def cache_stats(self) -> str:
        total = self._hits + self._misses
        rate = self._hits / total * 100 if total > 0 else 0
        return f"YOLO缓存: {self._hits}/{total} 命中 ({rate:.0f}%)"


# ══════════════════════════════════════════════════════════════════════════════
#  FramePrefetcher：视频帧预取流水线
# ══════════════════════════════════════════════════════════════════════════════

class FramePrefetcher:
    """
    独立线程预读视频帧，与 GPU 推理流水线并行。

    使用方式：
        pf = FramePrefetcher(video_path, max_queue=32)
        pf.start()
        while True:
            frame_idx, frame_bgr = pf.get()
            if frame_bgr is None:
                break
            # ... 推理 ...
        pf.stop()
    """

    def __init__(
        self,
        video_path: str,
        max_queue: int = 32,
        start_frame: int = 0,
        end_frame: int = -1,
    ):
        self.video_path = video_path
        self._max_queue = max_queue
        self._start_frame = start_frame
        self._end_frame = end_frame

        import queue as _queue_mod
        self._queue: _queue_mod.Queue = _queue_mod.Queue(maxsize=max_queue)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._total_frames = 0
        self._fps = 30.0

    def start(self):
        """启动预取线程"""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """停止预取"""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def get(self, timeout: float = 10.0) -> Tuple[int, Optional[np.ndarray]]:
        """获取下一帧。返回 (frame_idx, frame_bgr)，视频结束时 frame_bgr=None。超时抛出 TimeoutError。"""
        import queue as _q
        try:
            return self._queue.get(timeout=timeout)
        except _q.Empty:
            raise TimeoutError(f"FramePrefetcher: 等待帧超时 ({timeout}s)")

    @property
    def is_alive(self) -> bool:
        """预取线程是否仍在运行"""
        return self._thread is not None and self._thread.is_alive()

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def total_frames(self) -> int:
        return self._total_frames

    def _reader_loop(self):
        """后台读帧循环"""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self._queue.put((-1, None))
            return

        self._fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        end = self._end_frame if self._end_frame > 0 else self._total_frames

        if self._start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, self._start_frame)

        for i in range(self._start_frame, end):
            if self._stop_event.is_set():
                break
            ret, frame = cap.read()
            if not ret:
                break
            self._queue.put((i, frame))

        cap.release()
        self._queue.put((-1, None))  # 结束哨兵


# ══════════════════════════════════════════════════════════════════════════════
#  FP16 Autocast 包装器
# ══════════════════════════════════════════════════════════════════════════════

class FP16AutocastWrapper:
    """FP16 混合精度上下文管理器"""

    def __init__(self, config: InferenceConfig = DEFAULT_CONFIG):
        self.config = config
        self._enabled = config.use_fp16 and torch.cuda.is_available()
        self._cast = None
        if self._enabled:
            self._cast = torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=True,
            )

    def __enter__(self):
        if self._enabled and self._cast is not None:
            return self._cast.__enter__()
        return self

    def __exit__(self, *args):
        if self._enabled and self._cast is not None:
            return self._cast.__exit__(*args)


# ══════════════════════════════════════════════════════════════════════════════
#  加速版 ViTPose
# ══════════════════════════════════════════════════════════════════════════════

class AcceleratedViTPose:
    """
    加速版 ViTPose：真正的张量级批推理。

    与旧版的区别：
      旧版: 缓存 N 帧 → flush() 时逐帧 for 循环调用 base.detect_frame()
      新版: 缓存 N 帧 → flush() 时调用 base.detect_batch() 做单次 forward

    额外特性：
      - 可接入 SharedYOLO 避免重复检测
      - torch.compile 支持（可选）
    """

    def __init__(
        self,
        base_detector,
        config: InferenceConfig = DEFAULT_CONFIG,
        shared_yolo: Optional[SharedYOLO] = None,
    ):
        self.base = base_detector
        self.config = config
        self._shared_yolo = shared_yolo

        # ViTPose 禁用 FP16（scipy gaussian_filter 不兼容）
        self._fp16_enabled = False

        # torch.compile
        self._compiled = False
        if config.use_compile and hasattr(base_detector, 'model'):
            try:
                model = base_detector.model
                if hasattr(model, 'forward') and not getattr(model, '_compiled', False):
                    print("[AccelViTPose] 正在编译模型（首次需 30-120s）...")
                    t0 = time.time()
                    model.forward = torch.compile(
                        model.forward, mode=config.compile_mode,
                    )
                    model._compiled = True
                    self._compiled = True
                    print(f"[AccelViTPose] 编译完成 ({time.time()-t0:.1f}s)")
            except Exception as e:
                print(f"[AccelViTPose] compile 失败: {e}")

        # 批量缓存
        self._batch_frames: List[Tuple[np.ndarray, int, float]] = []
        self._batch_boxes: List[Optional[List[Tuple]]] = []
        self._batch_results: List[Any] = []

        # 统计
        self._total_inferences = 0
        self._total_batch_calls = 0

    def detect_frame(self, frame_bgr, frame_idx=0, timestamp=0.0) -> Any:
        """兼容原接口（带批量缓存 + shared YOLO）"""
        if not self.config.use_batch:
            return self._detect_single(frame_bgr, frame_idx, timestamp)

        # YOLO 检测（使用 shared）
        boxes = None
        if self._shared_yolo is not None:
            image_rgb = self.base._to_rgb_numpy(frame_bgr)
            boxes = self._shared_yolo.detect(image_rgb, frame_idx=frame_idx)

        self._batch_frames.append((frame_bgr, frame_idx, timestamp))
        self._batch_boxes.append(boxes)

        if len(self._batch_frames) >= self.config.batch_size:
            self._execute_batch()

        return None  # 批量模式由 flush() 统一返回

    def detect_frame_batch(self, frames: List[tuple]) -> list:
        """真正的批推理：所有帧的 crop 合并为单次 model forward。"""
        if not frames:
            return []

        images = []
        frame_indices = []
        timestamps = []
        predetected_list = []

        for item in frames:
            frame_bgr, fidx, ts = item[0], item[1], item[2]
            images.append(frame_bgr)
            frame_indices.append(fidx)
            timestamps.append(ts)

            # SharedYOLO 预检测
            if self._shared_yolo is not None:
                image_rgb = self.base._to_rgb_numpy(frame_bgr)
                boxes = self._shared_yolo.detect(image_rgb, frame_idx=fidx)
                predetected_list.append(boxes)
            else:
                predetected_list.append(None)

        self._total_batch_calls += 1
        self._total_inferences += len(frames)

        # 调用 base 的真正批推理
        try:
            results = self.base.detect_batch(
                images,
                frame_indices=frame_indices,
                timestamps=timestamps,
                predetected_boxes_list=predetected_list,
            )
            return results
        except Exception as e:
            # 回退逐帧
            warnings.warn(f"[AccelViTPose] 批推理失败 ({e})，回退逐帧")
            results = []
            for i, (frame_bgr, fidx, ts) in enumerate(
                zip(images, frame_indices, timestamps)
            ):
                boxes = predetected_list[i]
                r = self.base.detect_frame(
                    frame_bgr, frame_idx=fidx, timestamp=ts,
                    predetected_boxes=boxes,
                )
                results.append(r)
            return results

    def _execute_batch(self):
        """执行当前缓存的批次"""
        frames_with_meta = list(self._batch_frames)
        results = self.detect_frame_batch(frames_with_meta)
        self._batch_results.extend(results)
        self._batch_frames = []
        self._batch_boxes = []

    def flush(self) -> list:
        """刷新所有缓存帧并返回结果"""
        if self._batch_frames:
            self._execute_batch()
        results = self._batch_results
        self._batch_results = []
        return results

    def _detect_single(self, frame_bgr, frame_idx=0, timestamp=0.0) -> Any:
        """单帧推理（使用 shared YOLO）"""
        boxes = None
        if self._shared_yolo is not None:
            image_rgb = self.base._to_rgb_numpy(frame_bgr)
            boxes = self._shared_yolo.detect(image_rgb, frame_idx=frame_idx)
        self._total_inferences += 1
        return self.base.detect_frame(
            frame_bgr, frame_idx=frame_idx, timestamp=timestamp,
            predetected_boxes=boxes,
        )

    @property
    def device(self):
        return self.base.device

    def visualize_frame(self, frame_bgr, result, **kwargs):
        return self.base.visualize_frame(frame_bgr, result, **kwargs)

    @property
    def stats(self) -> str:
        return (
            f"ViTPose: {self._total_inferences} 帧, "
            f"{self._total_batch_calls} 次批推理, "
            f"compiled={self._compiled}"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  加速版 HMR2
# ══════════════════════════════════════════════════════════════════════════════

class AcceleratedHMR2:
    """
    加速版 HMR2：多帧多人 crop 合并为单次 forward。

    加速原理：
      1. SharedYOLO 缓存：同一帧不重复检测
      2. batch_reconstruct: 所有帧所有人的 crop 拼成一个 batch
      3. 向量化 rotmat→aa：scipy Rotation 单次调用处理所有关节
      4. FP16 autocast：对 HMR2 forward 使用半精度
      5. torch.compile：JIT 编译（可选）
    """

    def __init__(
        self,
        base_reconstructor,
        config: InferenceConfig = DEFAULT_CONFIG,
        shared_yolo: Optional[SharedYOLO] = None,
    ):
        self.base = base_reconstructor
        self.config = config
        self._shared_yolo = shared_yolo

        # torch.compile
        self._compiled = False
        if config.use_compile and hasattr(base_reconstructor, 'model'):
            try:
                model = base_reconstructor.model
                if hasattr(model, 'forward') and not getattr(model, '_compiled', False):
                    print("[AccelHMR2] 正在编译模型...")
                    t0 = time.time()
                    model.forward = torch.compile(
                        model.forward, mode=config.compile_mode,
                    )
                    model._compiled = True
                    self._compiled = True
                    print(f"[AccelHMR2] 编译完成 ({time.time()-t0:.1f}s)")
            except Exception as e:
                print(f"[AccelHMR2] compile 失败: {e}")

        # 批量缓存
        self._batch_frames: List[Tuple[np.ndarray, int, float]] = []
        self._batch_boxes: List[Optional[List[Tuple]]] = []
        self._batch_results: List[Any] = []

        # 统计
        self._total_inferences = 0
        self._total_batch_calls = 0

    def reconstruct_frame(self, frame_bgr, frame_idx=0, timestamp=0.0) -> Any:
        """兼容原接口（带批量缓存）"""
        if not self.config.use_batch:
            return self._reconstruct_single(frame_bgr, frame_idx, timestamp)

        boxes = None
        if self._shared_yolo is not None:
            image_rgb = self.base._to_rgb_numpy(frame_bgr)
            boxes = self._shared_yolo.detect(image_rgb, frame_idx=frame_idx)

        self._batch_frames.append((frame_bgr, frame_idx, timestamp))
        self._batch_boxes.append(boxes)

        if len(self._batch_frames) >= self.config.batch_size:
            self._execute_batch()

        return None

    def reconstruct_batch(self, frames: list) -> list:
        """真正的批推理：多帧多人 crop 合并为单次 forward。"""
        if not frames:
            return []

        # 准备 frames_data 供 base.batch_reconstruct 使用
        frames_data = []
        for item in frames:
            frame_bgr, fidx, ts = item[0], item[1], item[2]
            image_rgb = self.base._to_rgb_numpy(frame_bgr)

            boxes = None
            if self._shared_yolo is not None:
                boxes = self._shared_yolo.detect(image_rgb, frame_idx=fidx)

            frames_data.append((image_rgb, boxes, fidx, ts))

        self._total_batch_calls += 1
        self._total_inferences += len(frames)

        # 尝试使用真正的批量推理
        try:
            if self.config.use_fp16 and torch.cuda.is_available():
                with FP16AutocastWrapper(self.config):
                    batch_results = self.base.batch_reconstruct(frames_data)
            else:
                batch_results = self.base.batch_reconstruct(frames_data)
            return batch_results
        except Exception as e:
            # 回退逐帧
            warnings.warn(f"[AccelHMR2] 批推理失败 ({e})，回退逐帧")
            results = []
            for image_rgb, boxes, fidx, ts in frames_data:
                r = self.base.reconstruct_frame(
                    image_rgb, frame_idx=fidx, timestamp=ts,
                    predetected_boxes=boxes,
                )
                results.append(r)
            return results

    def _execute_batch(self):
        """执行当前缓存的批次"""
        frames_with_meta = list(self._batch_frames)
        results = self.reconstruct_batch(frames_with_meta)
        self._batch_results.extend(results)
        self._batch_frames = []
        self._batch_boxes = []

    def flush(self) -> list:
        if self._batch_frames:
            self._execute_batch()
        results = self._batch_results
        self._batch_results = []
        return results

    def _reconstruct_single(self, frame_bgr, frame_idx=0, timestamp=0.0) -> Any:
        boxes = None
        if self._shared_yolo is not None:
            image_rgb = self.base._to_rgb_numpy(frame_bgr)
            boxes = self._shared_yolo.detect(image_rgb, frame_idx=frame_idx)
        self._total_inferences += 1
        if self.config.use_fp16 and torch.cuda.is_available():
            with FP16AutocastWrapper(self.config):
                return self.base.reconstruct_frame(
                    frame_bgr, frame_idx=frame_idx, timestamp=timestamp,
                    predetected_boxes=boxes,
                )
        return self.base.reconstruct_frame(
            frame_bgr, frame_idx=frame_idx, timestamp=timestamp,
            predetected_boxes=boxes,
        )

    @property
    def device(self):
        return self.base.device

    @property
    def stats(self) -> str:
        return (
            f"HMR2: {self._total_inferences} 帧, "
            f"{self._total_batch_calls} 次批推理, "
            f"compiled={self._compiled}"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  工厂函数
# ══════════════════════════════════════════════════════════════════════════════

def create_accelerated_detector(
    vitpose_detector,
    hmr2_reconstructor=None,
    config: InferenceConfig = None,
    yolo_model_path: Optional[str] = None,
) -> tuple:
    """工厂函数：创建加速版检测器 + 共享 YOLO。

    Returns:
        (acc_vitpose, acc_hmr2, shared_yolo)
    """
    cfg = config or DEFAULT_CONFIG

    # 共享 YOLO
    shared_yolo = None
    if cfg.shared_yolo and yolo_model_path:
        shared_yolo = SharedYOLO.get_instance(yolo_model_path)

    acc_vitpose = AcceleratedViTPose(vitpose_detector, cfg, shared_yolo)
    acc_hmr2 = None
    if hmr2_reconstructor is not None:
        acc_hmr2 = AcceleratedHMR2(hmr2_reconstructor, cfg, shared_yolo)

    return acc_vitpose, acc_hmr2, shared_yolo


# ══════════════════════════════════════════════════════════════════════════════
#  性能基准测试工具
# ══════════════════════════════════════════════════════════════════════════════

class InferenceBenchmark:
    """推理性能计时器"""

    def __init__(self, name: str = ""):
        self.name = name
        self.timings: List[Tuple[str, float]] = []

    def record(self, label: str, t_seconds: float):
        self.timings.append((label, t_seconds))

    def summary(self) -> dict:
        if not self.timings:
            return {}
        total = sum(t for _, t in self.timings)
        return {
            "name": self.name,
            "total_time_s": total,
            "avg_ms": total / len(self.timings) * 1000 if self.timings else 0,
            "count": len(self.timings),
        }

    def print_summary(self):
        s = self.summary()
        if s:
            print(
                f"[Benchmark:{s['name']}] "
                f"总耗时 {s['total_time_s']:.2f}s, "
                f"平均 {s['avg_ms']:.1f}ms/次, "
                f"共 {s['count']} 次"
            )
