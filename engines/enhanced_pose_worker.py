"""
增强版姿态提取工作线程
支持：
- 全帧提取（非抽帧）
- 异常检测与插值/丢弃
- 加权关节速度
- 角速度（李代数）
- 多种融合策略
- GIF 可视化生成
"""
from __future__ import annotations

import os
import sys
import time
from typing import List, NamedTuple, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import cv2
import torch
from PySide6.QtCore import QThread, Signal
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

# 路径设置
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from core.algorithm_config import AlgorithmConfig, FusionStrategy, PeakMode


# ══════════════════════════════════════════════════════════════════════════════
#  模型预加载缓存辅助
# ══════════════════════════════════════════════════════════════════════════════
def _get_cached_model(key: str):
    """
    从 ModelPreloader 的全局缓存中获取预加载的模型实例。
    
    Args:
        key: 缓存键名（如 "vitpose_detector", "hmr2_reconstructor"）
    
    Returns:
        模型实例，或 None
    """
    try:
        from engines.model_preloader import _model_cache, _cache_mutex
        with _cache_mutex:
            return _model_cache.get(key)
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  推理加速配置
# ══════════════════════════════════════════════════════════════════════════════
_USE_ACCELERATION = True  # 全局开关：是否启用推理加速


class _AccelerationHelper:
    """
    推理加速辅助类，管理加速检测器的生命周期
    
    v2: 支持 SharedYOLO + 真正张量批推理 + FramePrefetcher
    """
    def __init__(self):
        self._acc_vitpose = None
        self._acc_hmr2 = None
        self._config = None
        self._shared_yolo = None
        self._enabled = False
        
        if _USE_ACCELERATION and torch.cuda.is_available():
            try:
                from engines.inference_accelerator import InferenceConfig
                self._config = InferenceConfig(
                    use_fp16=True,
                    use_compile=False,      # 可改为 True（首次较慢）
                    use_batch=True,
                    batch_size=8,
                    shared_yolo=True,
                )
                self._enabled = True
            except Exception as e:
                print(f"[Accel] 加速器加载失败: {e}，使用标准推理")
    
    @property
    def enabled(self) -> bool:
        return self._enabled
    
    def _ensure_shared_yolo(self):
        """延迟创建 SharedYOLO 单例"""
        if self._shared_yolo is None and self._config and self._config.shared_yolo:
            try:
                from engines.inference_accelerator import SharedYOLO
                from pose_extraction.weights_config import YOLO_WEIGHTS
                self._shared_yolo = SharedYOLO.get_instance(YOLO_WEIGHTS)
            except Exception as e:
                print(f"[Accel] SharedYOLO 初始化失败: {e}")
    
    def get_accelerated_vitpose(self, base_detector):
        """获取或创建加速版 ViTPose（支持 SharedYOLO + 真正批推理）"""
        if not self._enabled or self._config is None:
            return base_detector
        self._ensure_shared_yolo()
        if self._acc_vitpose is None or self._acc_vitpose.base is not base_detector:
            from engines.inference_accelerator import AcceleratedViTPose
            self._acc_vitpose = AcceleratedViTPose(
                base_detector, self._config, self._shared_yolo,
            )
            yolo_info = "SharedYOLO" if self._shared_yolo else "独立YOLO"
            print(f"[Accel] ViTPose 加速: 真批推理(#{self._config.batch_size}) + {yolo_info}")
        return self._acc_vitpose
    
    def get_accelerated_hmr2(self, base_reconstructor):
        """获取或创建加速版 HMR2（支持 SharedYOLO + 真正批推理）"""
        if not self._enabled or self._config is None:
            return base_reconstructor
        self._ensure_shared_yolo()
        if self._acc_hmr2 is None or self._acc_hmr2.base is not base_reconstructor:
            from engines.inference_accelerator import AcceleratedHMR2
            self._acc_hmr2 = AcceleratedHMR2(
                base_reconstructor, self._config, self._shared_yolo,
            )
            yolo_info = "SharedYOLO" if self._shared_yolo else "独立YOLO"
            print(f"[Accel] HMR2 加速: FP16 + 真批推理(#{self._config.batch_size}) + {yolo_info}")
        return self._acc_hmr2
    
    def flush_all(self):
        """刷新所有缓存的批量帧"""
        results_vit = []
        results_hmr = []
        if self._acc_vitpose is not None:
            results_vit = self._acc_vitpose.flush()
        if self._acc_hmr2 is not None:
            results_hmr = self._acc_hmr2.flush()
        return results_vit, results_hmr
    
    def clear_yolo_cache(self):
        """清除 SharedYOLO 帧缓存（新视频时调用）"""
        if self._shared_yolo is not None:
            self._shared_yolo.clear_cache()
    
    def print_stats(self):
        """打印加速统计信息"""
        if self._acc_vitpose is not None:
            print(f"[Accel] {self._acc_vitpose.stats}")
        if self._acc_hmr2 is not None:
            print(f"[Accel] {self._acc_hmr2.stats}")
        if self._shared_yolo is not None:
            print(f"[Accel] {self._shared_yolo.cache_stats}")


# 全局单例（延迟初始化，避免模块加载时 import 失败）
_accel_helper: _AccelerationHelper = None


def _get_accel_helper() -> _AccelerationHelper:
    """获取全局加速辅助实例（延迟创建）"""
    global _accel_helper
    if _accel_helper is None:
        _accel_helper = _AccelerationHelper()
    return _accel_helper


class PoseData(NamedTuple):
    """姿态提取完整结果"""
    keyframe_indices: List[int]
    beat_timestamps: List[float]
    joint_coords: Optional[np.ndarray]      # (T, 17, 2) ViTPose（平滑后）
    joint_rotations: Optional[np.ndarray]   # (T, 24, 3) HMR2 SMPL（平滑后）
    joint_velocities: Optional[np.ndarray]  # (T-1,) 加权速度
    rotation_gradients: Optional[np.ndarray]  # (T-1,) 旋转梯度/角速度
    fps: float
    total_frames: int
    valid_mask: Optional[np.ndarray]        # (T,) 有效帧掩码
    # 原始数据（未经异常处理和平滑）
    joint_coords_raw: Optional[np.ndarray] = None      # (T, 17, 2) ViTPose 原始
    joint_rotations_raw: Optional[np.ndarray] = None    # (T, 24, 3) HMR2 原始
    valid_mask_vit: Optional[np.ndarray] = None         # (T,) ViTPose 有效帧
    valid_mask_hmr: Optional[np.ndarray] = None         # (T,) HMR2 有效帧
    keyframe_sources: Optional[List[str]] = None        # 每个关键帧的来源: 'peak'/'valley'/'boundary'


@dataclass
class OutlierResult:
    """异常检测结果"""
    is_outlier: np.ndarray      # (T,) bool
    outlier_indices: List[int]  # 异常帧索引
    valid_ratio: float          # 有效帧比例
    interpolated: bool          # 是否进行了插值


class EnhancedPoseExtractionWorker(QThread):
    """
    增强版姿态提取工作线程
    
    信号：
        progress(int): 0-100 进度
        status(str): 状态文字
        finished(list): 关键帧索引列表
        pose_ready(PoseData): 完整姿态数据
        error(str): 错误信息
        gif_saved(str): GIF 保存路径（可视化完成）
    """
    
    progress = Signal(int)
    status = Signal(str)
    finished = Signal(list)
    pose_ready = Signal(object)
    error = Signal(str)
    gif_saved = Signal(str)
    
    def __init__(
        self,
        video_path: str,
        config: AlgorithmConfig,
        mode: str = "joint",  # "vitpose" | "hmr2" | "joint"
        parent=None,
    ):
        super().__init__(parent)
        self.video_path = video_path
        self.config = config
        self.mode = mode
        self._cancelled = False
        
        # 权重配置（必须在导入模型前设置）
        from pose_extraction.weights_config import HF_HOME
        os.environ["HF_HOME"] = HF_HOME
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
    
    def cancel(self):
        self._cancelled = True
    
    def run(self):
        try:
            self._run_impl()
        except Exception as e:
            import traceback
            self.error.emit(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
    
    def _run_impl(self):
        # 读取视频
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.error.emit(f"无法打开视频: {self.video_path}")
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        self.status.emit(f"视频共 {total_frames} 帧，{fps:.1f} fps")
        self.progress.emit(2)
        
        # 根据配置确定实际运行模式
        actual_mode = self._resolve_mode()
        
        joint_coords = None
        joint_rotations = None
        valid_mask_vit = None
        valid_mask_hmr = None
        
        # === ViTPose 管线 ===
        if actual_mode in ("vitpose", "joint"):
            joint_coords, valid_mask_vit = self._extract_vitpose(fps, total_frames)
        
        # === HMR2 管线 ===
        if actual_mode in ("hmr2", "joint"):
            joint_rotations, valid_mask_hmr = self._extract_hmr2(fps, total_frames)
        
        # === 保留原始数据（异常处理前）===
        joint_coords_raw = joint_coords.copy() if joint_coords is not None else None
        joint_rotations_raw = joint_rotations.copy() if joint_rotations is not None else None
        
        # === 异常处理 ===
        if joint_coords is not None:
            joint_coords, valid_mask_vit = self._handle_outliers(
                joint_coords, valid_mask_vit, "ViTPose"
            )
        
        if joint_rotations is not None:
            joint_rotations, valid_mask_hmr = self._handle_outliers(
                joint_rotations, valid_mask_hmr, "HMR2"
            )
        
        # === 计算特征 ===
        velocities = None
        rot_grads = None
        
        if joint_coords is not None:
            velocities = self._compute_weighted_velocity(joint_coords)
        
        if joint_rotations is not None:
            rot_grads = self._compute_rotation_metric(joint_rotations)
        
        # === 节拍检测 ===
        keyframe_indices, beat_timestamps, kf_sources = self._detect_beats(
            velocities, rot_grads, joint_coords, joint_rotations, fps
        )
        
        # === 生成可视化 ===
        if self.config.generate_gif:
            self._generate_visualizations(joint_coords, joint_rotations, fps)
        
        # === 汇总 ===
        pose_data = PoseData(
            keyframe_indices=keyframe_indices,
            beat_timestamps=beat_timestamps,
            joint_coords=joint_coords,
            joint_rotations=joint_rotations,
            joint_velocities=velocities,
            rotation_gradients=rot_grads,
            fps=fps,
            total_frames=total_frames,
            valid_mask=valid_mask_vit if actual_mode == "vitpose" else valid_mask_hmr,
            # 原始数据（未经异常处理和平滑）
            joint_coords_raw=joint_coords_raw,
            joint_rotations_raw=joint_rotations_raw,
            valid_mask_vit=valid_mask_vit,
            valid_mask_hmr=valid_mask_hmr,
            keyframe_sources=kf_sources,
        )
        
        self.progress.emit(100)
        self.finished.emit(keyframe_indices)
        self.pose_ready.emit(pose_data)
    
    def _resolve_mode(self) -> str:
        """根据配置解析实际运行模式"""
        strategy = self.config.fusion_strategy
        if strategy == "vitpose_only":
            return "vitpose"
        elif strategy == "hmr2_only":
            return "hmr2"
        else:
            return self.mode  # weighted_sum, adaptive_alpha, pca_fusion
    
    def _extract_vitpose(self, fps: float, total_frames: int) -> Tuple[np.ndarray, np.ndarray]:
        """提取 ViTPose 关节坐标（全帧），支持真正批推理 + SharedYOLO + 帧预取"""
        from pose_extraction.vitpose.detector import ViTPoseDetector
        from pose_extraction.weights_config import VITPOSE_MODEL_ID, YOLO_WEIGHTS
        
        self.status.emit("准备 ViTPose 模型...")
        self.progress.emit(5)
        
        # ★ 优先使用预加载的模型缓存 ★
        base_detector = _get_cached_model("vitpose_detector")
        if base_detector is None:
            self.status.emit("加载 ViTPose 模型...")
            base_detector = ViTPoseDetector(
                vitpose_model=VITPOSE_MODEL_ID,
                yolo_model=YOLO_WEIGHTS,
            )
        else:
            self.status.emit("ViTPose 模型已就绪（复用缓存）")
        
        # ★ 应用推理加速（真正批推理 + SharedYOLO）★
        accel = _get_accel_helper()
        accel.clear_yolo_cache()  # 新一轮提取，清除旧帧缓存
        detector = accel.get_accelerated_vitpose(base_detector)
        is_batch_mode = accel.enabled and hasattr(detector, 'flush')
        self.progress.emit(10)
        
        # 全帧提取
        kp_list = []
        valid_mask = np.ones(total_frames, dtype=bool)
        
        # ★ 帧预取流水线：后台线程解码帧，主线程做 GPU 推理 ★
        from engines.inference_accelerator import FramePrefetcher
        prefetcher = FramePrefetcher(self.video_path, max_queue=32)
        prefetcher.start()
        
        try:
            if is_batch_mode:
                # ── 批量模式：先喂所有帧，再一次性 flush 获取结果 ──
                fed_count = 0
                _timeout_retries = 0
                _MAX_RETRIES = 5
                while True:
                    if self._cancelled:
                        break
                    try:
                        frame_idx, frame = prefetcher.get(timeout=30.0)
                        _timeout_retries = 0
                    except TimeoutError:
                        _timeout_retries += 1
                        if _timeout_retries >= _MAX_RETRIES:
                            print(f"[WARNING] ViTPose batch prefetcher 连续超时 {_MAX_RETRIES} 次，中断")
                            self.status.emit(f"⚠️ ViTPose 帧读取超时，已处理部分帧")
                            break
                        continue
                    if frame is None:
                        break
                    
                    detector.detect_frame(frame, frame_idx=frame_idx, timestamp=frame_idx/fps)
                    fed_count += 1
                    
                    if frame_idx % 30 == 0:
                        pct = 10 + int(frame_idx / total_frames * 35)
                        self.progress.emit(pct)
                        self.status.emit(f"ViTPose 推理 {frame_idx}/{total_frames}")
                
                # 一次性获取所有结果
                all_results = detector.flush() if hasattr(detector, 'flush') else []
                
                # 填充结果
                for i in range(total_frames):
                    if i < len(all_results):
                        r = all_results[i]
                        n_persons = len(r.persons) if r and r.persons else 0
                        if n_persons >= 1:
                            kp = list(r.persons.values())[0][:, :2].astype(np.float32)
                            kp_list.append(kp)
                        else:
                            valid_mask[i] = False
                            kp_list.append(np.zeros((17, 2), dtype=np.float32))
                    else:
                        valid_mask[i] = False
                        kp_list.append(np.zeros((17, 2), dtype=np.float32))
            
            else:
                # ── 单帧模式：逐帧推理 ──
                _timeout_retries = 0
                _MAX_RETRIES = 5
                while True:
                    if self._cancelled:
                        break
                    try:
                        frame_idx, frame = prefetcher.get(timeout=30.0)
                        _timeout_retries = 0
                    except TimeoutError:
                        _timeout_retries += 1
                        if _timeout_retries >= _MAX_RETRIES:
                            print(f"[WARNING] ViTPose single-frame prefetcher 连续超时 {_MAX_RETRIES} 次，中断")
                            break
                        continue
                    if frame is None:
                        break
                    
                    result = detector.detect_frame(frame, frame_idx=frame_idx, timestamp=frame_idx/fps)
                    
                    # 填充跳过的帧
                    while len(kp_list) < frame_idx:
                        valid_mask[len(kp_list)] = False
                        kp_list.append(np.zeros((17, 2), dtype=np.float32))
                    
                    n_persons = len(result.persons) if result and result.persons else 0
                    if n_persons >= 1:
                        kp = list(result.persons.values())[0][:, :2].astype(np.float32)
                        kp_list.append(kp)
                    else:
                        valid_mask[frame_idx] = False
                        kp_list.append(np.zeros((17, 2), dtype=np.float32))
                    
                    if frame_idx % 30 == 0:
                        pct = 10 + int(frame_idx / total_frames * 35)
                        self.progress.emit(pct)
                        self.status.emit(f"ViTPose 推理 {frame_idx}/{total_frames}")
        
        finally:
            prefetcher.stop()
        
        # 补齐不足的帧
        while len(kp_list) < total_frames:
            valid_mask[len(kp_list)] = False
            kp_list.append(np.zeros((17, 2), dtype=np.float32))
        
        coords = np.stack(kp_list) if kp_list else np.zeros((total_frames, 17, 2))
        return coords, valid_mask
    
    def _extract_hmr2(self, fps: float, total_frames: int) -> Tuple[np.ndarray, np.ndarray]:
        """提取 HMR2 SMPL 旋转（全帧），支持真正批推理 + SharedYOLO + 帧预取"""
        from pose_extraction.hmr2.reconstructor import HMR2Reconstructor
        from pose_extraction.weights_config import HMR2_CHECKPOINT, YOLO_WEIGHTS
        
        self.status.emit("准备 HMR2 模型...")
        self.progress.emit(50)
        
        # ★ 优先使用预加载的模型缓存 ★
        base_reconstructor = _get_cached_model("hmr2_reconstructor")
        if base_reconstructor is None:
            self.status.emit("加载 HMR2 模型...")
            base_reconstructor = HMR2Reconstructor(
                checkpoint_path=HMR2_CHECKPOINT,
                yolo_model=YOLO_WEIGHTS,
            )
        else:
            self.status.emit("HMR2 模型已就绪（复用缓存）")
        
        # ★ 应用推理加速（FP16 + 真正批推理 + SharedYOLO）★
        accel = _get_accel_helper()
        reconstructor = accel.get_accelerated_hmr2(base_reconstructor)
        is_batch_mode = accel.enabled and hasattr(reconstructor, 'flush')
        self.progress.emit(55)
        
        rot_list = []
        valid_mask = np.ones(total_frames, dtype=bool)
        
        # ★ 帧预取流水线 ★
        from engines.inference_accelerator import FramePrefetcher
        prefetcher = FramePrefetcher(self.video_path, max_queue=32)
        prefetcher.start()
        
        try:
            if is_batch_mode:
                # ── 批量模式：先喂所有帧，再一次性 flush 获取结果 ──
                _timeout_retries = 0
                _MAX_RETRIES = 5
                while True:
                    if self._cancelled:
                        break
                    try:
                        frame_idx, frame = prefetcher.get(timeout=30.0)
                        _timeout_retries = 0
                    except TimeoutError:
                        _timeout_retries += 1
                        if _timeout_retries >= _MAX_RETRIES:
                            print(f"[WARNING] HMR2 prefetcher 连续超时 {_MAX_RETRIES} 次，中断")
                            self.status.emit(f"⚠️ HMR2 帧读取超时，已处理部分帧")
                            break
                        print(f"[WARNING] HMR2 prefetcher 超时，重试 ({_timeout_retries}/{_MAX_RETRIES})")
                        continue
                    if frame is None:
                        break
                    
                    reconstructor.reconstruct_frame(frame, frame_idx=frame_idx, timestamp=frame_idx/fps)
                    
                    if frame_idx % 10 == 0:
                        pct = 55 + int(frame_idx / total_frames * 30)
                        self.progress.emit(pct)
                        self.status.emit(f"HMR2 推理 {frame_idx}/{total_frames}")
                
                # 一次性获取所有结果
                all_results = reconstructor.flush() if hasattr(reconstructor, 'flush') else []
                
                # 填充结果
                for i in range(total_frames):
                    if i < len(all_results):
                        results = all_results[i]
                        n_persons = len(results) if results else 0
                        if n_persons >= 1:
                            r = results[0]
                            go = r.global_orient
                            bp = r.body_pose
                            if go is not None and bp is not None:
                                full = np.vstack([go.reshape(1, 3), bp])
                                if self._check_joint_limits(full):
                                    rot_list.append(full.astype(np.float32))
                                else:
                                    valid_mask[i] = False
                                    rot_list.append(np.zeros((24, 3), dtype=np.float32))
                            else:
                                valid_mask[i] = False
                                rot_list.append(np.zeros((24, 3), dtype=np.float32))
                        else:
                            valid_mask[i] = False
                            rot_list.append(np.zeros((24, 3), dtype=np.float32))
                    else:
                        valid_mask[i] = False
                        rot_list.append(np.zeros((24, 3), dtype=np.float32))
            
            else:
                # ── 单帧模式：逐帧推理 ──
                _timeout_retries = 0
                _MAX_RETRIES = 5
                while True:
                    if self._cancelled:
                        break
                    try:
                        frame_idx, frame = prefetcher.get(timeout=30.0)
                        _timeout_retries = 0
                    except TimeoutError:
                        _timeout_retries += 1
                        if _timeout_retries >= _MAX_RETRIES:
                            print(f"[WARNING] HMR2 single-frame prefetcher 连续超时 {_MAX_RETRIES} 次，中断")
                            break
                        continue
                    if frame is None:
                        break
                    
                    # 填充跳过的帧
                    while len(rot_list) < frame_idx:
                        valid_mask[len(rot_list)] = False
                        rot_list.append(np.zeros((24, 3), dtype=np.float32))
                    
                    results = reconstructor.reconstruct_frame(frame, frame_idx=frame_idx, timestamp=frame_idx/fps)
                    
                    n_persons = len(results) if results else 0
                    if n_persons >= 1:
                        r = results[0]
                        go = r.global_orient
                        bp = r.body_pose
                        if go is not None and bp is not None:
                            full = np.vstack([go.reshape(1, 3), bp])
                            if self._check_joint_limits(full):
                                rot_list.append(full.astype(np.float32))
                            else:
                                valid_mask[frame_idx] = False
                                rot_list.append(np.zeros((24, 3), dtype=np.float32))
                        else:
                            valid_mask[frame_idx] = False
                            rot_list.append(np.zeros((24, 3), dtype=np.float32))
                    else:
                        valid_mask[frame_idx] = False
                        rot_list.append(np.zeros((24, 3), dtype=np.float32))
                    
                    if frame_idx % 10 == 0:
                        pct = 55 + int(frame_idx / total_frames * 30)
                        self.progress.emit(pct)
                        self.status.emit(f"HMR2 推理 {frame_idx}/{total_frames}")
        
        finally:
            prefetcher.stop()
        
        # 补齐不足的帧
        while len(rot_list) < total_frames:
            valid_mask[len(rot_list)] = False
            rot_list.append(np.zeros((24, 3), dtype=np.float32))
        
        # 打印加速统计
        accel.print_stats()
        
        rotations = np.stack(rot_list) if rot_list else np.zeros((total_frames, 24, 3))
        return rotations, valid_mask
    
    # ──────────────────────────────────────────────────────────────────────────
    # 关节旋转极限检测
    # ──────────────────────────────────────────────────────────────────────────
    
    # SMPL 24 关节名称索引
    # 0: root, 1-3: 左腿(hip_x, hip_z, hip_y), 4: 左膝, 5-6: 左踝
    # 7-9: 右腿, 10: 右膝, 11-12: 右踝
    # 13-15: 腰部, 16-18: 左臂(shoulder_x, shoulder_z, shoulder_y)
    # 19: 左肘, 20-21: 左腕, 22-24: 右臂, 25: 右肘, 26-27: 右腕
    
    # 基于 Unitree G1 和人体运动学文献的关节旋转极限（弧度）
    # SMPL 轴角每个关节 3 个分量，对应 (x, y, z) 轴旋转
    # 每个关节的极限为 (lower, upper) 对，三个轴各一个
    JOINT_LIMITS = {
        # 索引: (joint_name, [(x_low, x_high), (y_low, y_high), (z_low, z_high)])
        0:  ("root",         [(-3.14, 3.14), (-1.0, 1.0), (-3.14, 3.14)]),
        1:  ("left_hip_x",   [(-2.53, 2.88), (-2.76, 2.76), (-0.52, 2.97)]),
        2:  ("left_hip_z",   [(-2.53, 2.88), (-2.76, 2.76), (-0.52, 2.97)]),
        3:  ("left_hip_y",   [(-2.53, 2.88), (-2.76, 2.76), (-0.52, 2.97)]),
        4:  ("left_knee",    [(-0.09, 2.88), (-0.5, 0.5), (-0.5, 0.5)]),
        5:  ("left_ankle_x", [(-0.87, 0.52), (-0.26, 0.26), (-0.5, 0.5)]),
        6:  ("left_ankle_y", [(-0.87, 0.52), (-0.26, 0.26), (-0.5, 0.5)]),
        7:  ("right_hip_x",  [(-2.53, 2.88), (-2.76, 2.76), (-2.97, 0.52)]),
        8:  ("right_hip_z",  [(-2.53, 2.88), (-2.76, 2.76), (-2.97, 0.52)]),
        9:  ("right_hip_y",  [(-2.53, 2.88), (-2.76, 2.76), (-2.97, 0.52)]),
        10: ("right_knee",   [(-0.09, 2.88), (-0.5, 0.5), (-0.5, 0.5)]),
        11: ("right_ankle_x",[(-0.87, 0.52), (-0.26, 0.26), (-0.5, 0.5)]),
        12: ("right_ankle_y",[(-0.87, 0.52), (-0.26, 0.26), (-0.5, 0.5)]),
        13: ("spine1",       [(-2.62, 2.62), (-0.52, 0.52), (-0.52, 0.52)]),
        14: ("spine2",       [(-2.62, 2.62), (-0.52, 0.52), (-0.52, 0.52)]),
        15: ("spine3",       [(-2.62, 2.62), (-0.52, 0.52), (-0.52, 0.52)]),
        16: ("left_shoulder_x",[(-3.09, 2.67), (-2.62, 2.62), (-1.59, 2.25)]),
        17: ("left_shoulder_z",[(-3.09, 2.67), (-2.62, 2.62), (-1.59, 2.25)]),
        18: ("left_shoulder_y",[(-3.09, 2.67), (-2.62, 2.62), (-1.59, 2.25)]),
        19: ("left_elbow",   [(-1.05, 2.09), (-1.97, 1.97), (-1.61, 1.61)]),
        20: ("left_wrist_x", [(-1.97, 1.97), (-1.61, 1.61), (-1.97, 1.97)]),
        21: ("left_wrist_y", [(-1.97, 1.97), (-1.61, 1.61), (-1.97, 1.97)]),
        22: ("right_shoulder_x",[(-3.09, 2.67), (-2.62, 2.62), (-2.25, 1.59)]),
        23: ("right_shoulder_z",[(-3.09, 2.67), (-2.62, 2.62), (-2.25, 1.59)]),
    }
    # 补充 24-23 的索引（SMPL 只有 24 个关节，0-23）
    # 实际上 HMR2 输出 24 关节: 1 global_orient + 23 body_pose
    
    @staticmethod
    def _check_joint_limits(rotation: np.ndarray) -> bool:
        """
        检查关节旋转量是否在人体极限范围内
        
        Args:
            rotation: (24, 3) SMPL 轴角旋转参数
        
        Returns:
            True = 合法，False = 超出极限
        """
        # 轴角转换为旋转角度
        for j in range(24):
            angle = np.linalg.norm(rotation[j])  # 旋转角度（弧度）
            
            # 大关节允许更大旋转
            if j == 0:
                # 根节点允许较大旋转
                max_angle = np.pi  # 180°
            elif j in (4, 10):
                # 膝关节主要是弯曲
                max_angle = 2.9  # ~166°
            elif j in (19,):
                # 肘关节
                max_angle = 2.1  # ~120°
            elif j in (5, 6, 11, 12):
                # 踝关节较小
                max_angle = 1.0  # ~57°
            elif j in (20, 21, 22, 23):
                # 腕关节、手部
                max_angle = 2.0  # ~115°
            else:
                # 其他关节
                max_angle = 3.14  # 180°
            
            if angle > max_angle:
                return False
        
        return True
    
    def _handle_outliers(
        self, 
        data: np.ndarray, 
        valid_mask: np.ndarray,
        source: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        多层异常检测与鲁棒平滑
        
        三层检测策略：
        1. 模型失败帧（valid_mask=False）
        2. 滑动窗口自适应阈值检测跳变帧
        3. 连续异常段分析（短段插值、长段丢弃）
        
        平滑策略：
        - 孤立异常帧: 三次样条插值
        - 短连续异常段 (≤5帧): 三次样条插值
        - 长连续异常段 (>5帧): 标记为不可靠区间
        - 最终: 全局 Savitzky-Golay 平滑去除高频抖动
        """
        self.status.emit(f"{source}: 多层异常检测...")
        
        T = len(data)
        if T < 3:
            return data, valid_mask
        
        outlier_mask = ~valid_mask.copy()
        
        # ── 第1层: 帧间跳变检测（滑动窗口自适应阈值）──
        flat = data.reshape(T, -1)
        diffs = np.linalg.norm(np.diff(flat, axis=0), axis=1)  # (T-1,)
        
        # 滑动窗口中值 + MAD (Median Absolute Deviation) —— 对异常值鲁棒
        window_size = max(15, int(T * 0.05))  # 5% 帧数或至少15帧
        half_w = window_size // 2
        
        for i in range(len(diffs)):
            lo = max(0, i - half_w)
            hi = min(len(diffs), i + half_w + 1)
            local_window = diffs[lo:hi]
            
            med = np.median(local_window)
            mad = np.median(np.abs(local_window - med))
            mad = max(mad, 1e-6)  # 防零
            
            # Modified Z-score: 超过阈值倍的 MAD 视为异常
            if (diffs[i] - med) / mad > self.config.outlier_threshold * 1.4826:
                outlier_mask[i + 1] = True
                # 跳变通常影响前后帧：如果 i+2 也有大跳变，同时标记
                if i + 2 < T and i + 1 < len(diffs):
                    if (diffs[i + 1] - med) / mad > self.config.outlier_threshold * 1.0:
                        outlier_mask[i + 2] = True
        
        # ── 第2层: 骨架一致性检测（仅ViTPose，检查关节间距突变）──
        if source == "ViTPose" and data.ndim == 3 and data.shape[1] == 17:
            # 关键骨骼长度: 肩宽(5-6), 髋宽(11-12), 躯干(5-11, 6-12)
            bone_pairs = [(5, 6), (11, 12), (5, 11), (6, 12)]
            for a, b in bone_pairs:
                bone_lengths = np.sqrt(((data[:, a] - data[:, b]) ** 2).sum(axis=-1))
                med_len = np.median(bone_lengths[~outlier_mask])
                if med_len > 0:
                    ratio = bone_lengths / med_len
                    # 骨骼长度变化超过50%视为异常
                    outlier_mask |= (ratio > 1.5) | (ratio < 0.5)
        
        # ── 第3层: 连续异常段分析 ──
        outlier_count = np.sum(outlier_mask)
        valid_ratio = np.mean(~outlier_mask)
        
        self.status.emit(f"{source}: 异常帧 {outlier_count}/{T} ({valid_ratio*100:.1f}% 有效)")
        
        if valid_ratio < self.config.min_valid_ratio:
            print(f"[WARNING] {source}: 有效帧比例 {valid_ratio*100:.1f}% 过低，跳过进一步异常检测")
            # 保留当前已有效的帧，而不是全部标为无效
            return data, (~outlier_mask).copy()
        
        # 分析连续异常段
        segments = self._find_outlier_segments(outlier_mask)
        max_interp_gap = 5  # 最大插值跨度（帧）
        
        for start, end in segments:
            gap_len = end - start
            if gap_len > max_interp_gap:
                # 长异常段：不插值，保持标记
                print(f"[{source}] 长异常段 [{start}:{end}] ({gap_len}帧) → 跳过")
            # 短异常段的插值在 _interpolate_outliers_spline 中处理
        
        # ── 插值修复有效短异常段 ──
        if outlier_count > 0:
            data = self._interpolate_outliers_spline(data, outlier_mask, max_gap=max_interp_gap)
            valid_mask = ~outlier_mask
        
        # ── 全局 Savitzky-Golay 平滑（去除高频抖动而保留动作结构）──
        data = self._savgol_smooth(data, valid_mask)
        
        return data, valid_mask
    
    @staticmethod
    def _find_outlier_segments(mask: np.ndarray) -> List[Tuple[int, int]]:
        """找到连续异常段的 [start, end) 区间列表"""
        segments = []
        in_seg = False
        start = 0
        for i in range(len(mask)):
            if mask[i] and not in_seg:
                start = i
                in_seg = True
            elif not mask[i] and in_seg:
                segments.append((start, i))
                in_seg = False
        if in_seg:
            segments.append((start, len(mask)))
        return segments
    
    def _interpolate_outliers_spline(
        self, data: np.ndarray, outlier_mask: np.ndarray, max_gap: int = 5
    ) -> np.ndarray:
        """
        三次样条插值修复短异常段
        
        相比线性插值的优势：
        - 保持位姿的时序连续性（C2 连续）
        - 过渡更自然，无尖锐折痕
        - 利用更远的有效帧进行拟合
        """
        from scipy.interpolate import CubicSpline
        
        T = len(data)
        result = data.copy()
        flat = result.reshape(T, -1)  # (T, D)
        D = flat.shape[1]
        
        valid_indices = np.where(~outlier_mask)[0]
        if len(valid_indices) < 4:
            # 有效帧太少，无法做样条拟合
            return data
        
        # 找需要插值的短异常段
        segments = self._find_outlier_segments(outlier_mask)
        
        for start, end in segments:
            gap_len = end - start
            if gap_len > max_gap:
                continue  # 跳过长段
            
            # 收集上下文有效帧（前后各取 min(4, 可用) 个）
            ctx_before = valid_indices[valid_indices < start][-4:]
            ctx_after = valid_indices[valid_indices >= end][:4]
            
            if len(ctx_before) < 1 or len(ctx_after) < 1:
                # 边界情况：退化为最近邻
                if len(ctx_before) >= 1:
                    flat[start:end] = flat[ctx_before[-1]]
                elif len(ctx_after) >= 1:
                    flat[start:end] = flat[ctx_after[0]]
                continue
            
            ctx_indices = np.concatenate([ctx_before, ctx_after])
            ctx_values = flat[ctx_indices]
            
            # 对每个维度做三次样条插值
            interp_indices = np.arange(start, end)
            for d in range(D):
                try:
                    cs = CubicSpline(ctx_indices, ctx_values[:, d],
                                     bc_type='natural')
                    flat[start:end, d] = cs(interp_indices)
                except Exception:
                    # 退化到线性插值
                    flat[start:end, d] = np.interp(
                        interp_indices, ctx_indices, ctx_values[:, d]
                    )
        
        return flat.reshape(data.shape)
    
    @staticmethod
    def _savgol_smooth(data: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        """
        Savitzky-Golay 平滑：去除高频抖动，保留动作结构
        
        对有效帧应用 SG 滤波器（window=7, polyorder=3）
        比高斯平滑更好地保留峰值形状
        """
        from scipy.signal import savgol_filter as sg
        
        T = len(data)
        if T < 9:
            return data
        
        result = data.copy()
        flat = result.reshape(T, -1)
        
        # 仅对有足够有效帧的维度做平滑
        window = min(7, T if T % 2 == 1 else T - 1)
        if window < 5:
            return data
        
        for d in range(flat.shape[1]):
            col = flat[:, d].copy()
            try:
                col_smooth = sg(col, window_length=window, polyorder=3, mode='nearest')
                # 只在有效帧位置使用平滑值（异常帧保留插值结果）
                flat[:, d] = col_smooth
            except Exception:
                pass
        
        return flat.reshape(data.shape)
    
    def _interpolate_outliers(self, data: np.ndarray, outlier_mask: np.ndarray) -> np.ndarray:
        """对异常帧进行线性插值（保留作为回退方法）"""
        T = len(data)
        result = data.copy()
        
        outlier_indices = np.where(outlier_mask)[0]
        
        for idx in outlier_indices:
            prev_valid = None
            next_valid = None
            
            for i in range(idx - 1, -1, -1):
                if not outlier_mask[i]:
                    prev_valid = i
                    break
            
            for i in range(idx + 1, T):
                if not outlier_mask[i]:
                    next_valid = i
                    break
            
            if prev_valid is not None and next_valid is not None:
                alpha = (idx - prev_valid) / (next_valid - prev_valid)
                result[idx] = (1 - alpha) * data[prev_valid] + alpha * data[next_valid]
            elif prev_valid is not None:
                result[idx] = data[prev_valid]
            elif next_valid is not None:
                result[idx] = data[next_valid]
        
        return result
    
    def _compute_weighted_velocity(self, kp_arr: np.ndarray) -> np.ndarray:
        """
        计算加权关节速度
        末端执行器（手腕、脚踝）权重更高
        """
        T = len(kp_arr)
        if T < 2:
            return np.array([])
        
        # COCO-17 关节权重定义
        # 0: 鼻子, 1-2: 眼, 3-4: 耳, 5-6: 肩, 7-8: 肘, 9-10: 腕, 11-12: 髋, 13-14: 膝, 15-16: 踝
        weights = np.ones(17, dtype=np.float32)
        
        if self.config.velocity_metric == "weighted":
            # 末端执行器
            weights[[9, 10]] = self.config.end_effector_weight  # 手腕
            weights[[15, 16]] = self.config.end_effector_weight  # 脚踝
            # 躯干
            weights[[5, 6, 11, 12]] = self.config.torso_weight  # 肩、髋
            # 四肢
            weights[[7, 8, 13, 14]] = self.config.limb_weight  # 肘、膝
        
        # 归一化权重
        weights = weights / weights.sum()
        
        velocities = []
        for i in range(1, T):
            delta = kp_arr[i] - kp_arr[i-1]  # (17, 2)
            speed_per_joint = np.sqrt((delta ** 2).sum(axis=-1))  # (17,)
            weighted_speed = np.sum(speed_per_joint * weights)
            velocities.append(weighted_speed)
        
        return np.array(velocities, dtype=np.float32)
    
    def _compute_rotation_metric(self, rot_arr: np.ndarray) -> np.ndarray:
        """
        计算旋转指标（排除根关节 joint 0 / global_orient，避免全局朝向漂移干扰）
        - axis_angle_diff: 轴角差分
        - angular_velocity: 角速度（李代数）
        """
        T = len(rot_arr)
        if T < 2:
            return np.array([])
        
        # 排除根关节（index 0），仅用 body joints 1-23
        body_rot = rot_arr[:, 1:, :] if rot_arr.ndim == 3 and rot_arr.shape[1] > 1 else rot_arr
        
        metrics = []
        
        for i in range(1, T):
            if self.config.rotation_metric == "angular_velocity":
                # 使用李代数计算角速度
                metric = self._compute_angular_velocity(body_rot[i-1], body_rot[i])
            else:
                # 简单的轴角差分
                diff = body_rot[i] - body_rot[i-1]
                metric = float((diff ** 2).sum())
            
            metrics.append(metric)
        
        return np.array(metrics, dtype=np.float32)
    
    def _compute_angular_velocity(self, rot_prev: np.ndarray, rot_curr: np.ndarray) -> float:
        """
        计算角速度（李代数表示）
        将轴角转换为旋转矩阵，计算相对旋转，再转回李代数
        """
        total_vel = 0.0
        
        for j in range(len(rot_prev)):
            # 轴角 -> 旋转矩阵
            R_prev = R.from_rotvec(rot_prev[j]).as_matrix()
            R_curr = R.from_rotvec(rot_curr[j]).as_matrix()
            
            # 相对旋转
            R_rel = R_prev.T @ R_curr
            
            # 旋转矩阵 -> 李代数（角速度）
            try:
                omega = R.from_matrix(R_rel).as_rotvec()
                total_vel += np.linalg.norm(omega) ** 2
            except Exception:
                # 退化情况回退到轴角差分
                diff = rot_curr[j] - rot_prev[j]
                total_vel += float((diff ** 2).sum())
        
        return total_vel
    
    def _detect_beats(
        self,
        velocities: Optional[np.ndarray],
        rot_grads: Optional[np.ndarray],
        joint_coords: Optional[np.ndarray],
        joint_rotations: Optional[np.ndarray],
        fps: float,
    ) -> Tuple[List[int], List[float], List[str]]:
        """
        多策略节拍/突变帧检测
        
        返回: (keyframe_indices, beat_times, keyframe_sources)
        keyframe_sources: 每个关键帧的来源 ('peak'/'valley'/'boundary')
        """
        strategy = self.config.fusion_strategy
        
        if strategy == "vitpose_only" or (strategy == "weighted_sum" and rot_grads is None):
            return self._detect_multiscale(velocities, fps, "velocity")
        
        elif strategy == "hmr2_only" or (strategy == "weighted_sum" and velocities is None):
            return self._detect_multiscale(rot_grads, fps, "rotation")
        
        elif strategy == "weighted_sum":
            return self._detect_weighted_fusion(velocities, rot_grads, fps)
        
        elif strategy == "adaptive_alpha":
            return self._detect_adaptive_fusion(velocities, rot_grads, fps)
        
        else:
            return self._detect_weighted_fusion(velocities, rot_grads, fps)
    
    def _detect_multiscale(
        self, 
        signal: np.ndarray, 
        fps: float, 
        signal_type: str,
        return_diagnostics: bool = False,
    ) -> Tuple[List[int], List[float]]:
        """
        多尺度峰值检测 + 加速度零交叉
        
        双通道检测：
        - 通道A: 低平滑(σ=1)捕捉细粒度动作突变
        - 通道B: 高平滑(σ=3)捕捉大动作转折
        - 通道C: 加速度过零点（动作方向反转时刻）
        
        投票合并：至少2/3通道同时触发才认定为突变帧
        
        当 return_diagnostics=True 时，返回第三个元素为诊断字典。
        """
        if signal is None or len(signal) < 5:
            if return_diagnostics:
                return [0], [0.0], {}
            return [0], [0.0]
        
        N = len(signal)
        
        # ── 自适应 σ: 周期性检测 → 分策略估计最优平滑参数 ──
        sigma = self.config.gaussian_sigma
        is_periodic = False
        if getattr(self.config, 'auto_sigma', False) and N >= 16:
            sigma, is_periodic = self._estimate_sigma(signal, fps)
        
        # ── 通道 A: 细粒度峰值 ──
        smooth_fine = gaussian_filter1d(signal, sigma=max(0.5, sigma * 0.5))
        peaks_fine = self._adaptive_peak_detect(smooth_fine, fps, scale="fine")
        
        # ── 通道 B: 粗粒度峰值 ──
        smooth_coarse = gaussian_filter1d(signal, sigma=max(1.5, sigma * 2.0))
        peaks_coarse = self._adaptive_peak_detect(smooth_coarse, fps, scale="coarse")
        
        # ── 通道 C: 加速度过零点 ──
        smooth_mid = gaussian_filter1d(signal, sigma=sigma)
        velocity = np.diff(smooth_mid, prepend=smooth_mid[0])
        acceleration = np.diff(velocity, prepend=velocity[0])
        
        # 过零点: 加速度从正变负 → 速度达到峰值（动作极速时刻）
        zero_crossings = []
        for i in range(1, len(acceleration)):
            if acceleration[i-1] > 0 and acceleration[i] <= 0:
                # 正→负: 速度达到局部最大值
                if smooth_mid[i] > np.percentile(smooth_mid, 30):
                    zero_crossings.append(i)
        
        # ── 投票合并 ──
        # 创建投票热力图
        vote_map = np.zeros(N, dtype=np.float32)
        
        vote_radius = max(2, int(fps * 0.05))  # 50ms容差窗口
        
        for p in peaks_fine:
            lo, hi = max(0, p - vote_radius), min(N, p + vote_radius + 1)
            vote_map[lo:hi] += 1.0
        
        for p in peaks_coarse:
            lo, hi = max(0, p - vote_radius), min(N, p + vote_radius + 1)
            vote_map[lo:hi] += 1.2  # 粗粒度权重稍高
        
        for p in zero_crossings:
            lo, hi = max(0, p - vote_radius), min(N, p + vote_radius + 1)
            vote_map[lo:hi] += 0.8
        
        # 票数 ≥ 1.5 视为确认突变帧（至少2个通道相近位置触发）
        confirmed_mask = vote_map >= 1.5
        
        # 从确认区域提取精确峰值位置
        min_dist = max(self.config.min_peak_distance, int(fps * 0.15))
        
        # 在 vote_map 上找峰值
        final_peaks, _ = find_peaks(vote_map, distance=min_dist, height=1.5)
        
        if len(final_peaks) == 0:
            # 回退: 降低阈值到单通道
            final_peaks, _ = find_peaks(
                smooth_mid,
                height=np.mean(smooth_mid) + 0.3 * np.std(smooth_mid),
                distance=min_dist
            )
        
        # 精化: 将每个峰值对齐到原始信号的局部最大值
        refined_peaks = []
        for p in final_peaks:
            lo = max(0, p - vote_radius)
            hi = min(N, p + vote_radius + 1)
            local_max_idx = lo + np.argmax(signal[lo:hi])
            refined_peaks.append(int(local_max_idx))
        refined_peaks = sorted(set(refined_peaks))
        
        # ── 谷值检测 ──
        refined_valleys = []
        peak_mode = getattr(self.config, 'peak_mode', 'peaks_only')
        if peak_mode in ('valleys_only', 'peaks_and_valleys'):
            # 在 smooth_mid 上检测谷值
            valley_indices, _ = find_peaks(-smooth_mid, distance=min_dist)
            # 用低于均值的阈值过滤：仅保留真正低洼的谷
            valley_threshold = np.mean(smooth_mid) - self.config.peak_height_factor * np.std(smooth_mid)
            for vi in valley_indices:
                if smooth_mid[vi] < valley_threshold:
                    refined_valleys.append(int(vi))
            refined_valleys = sorted(set(refined_valleys))
        
        # ── 合并并标记来源 ──
        tagged: list  # [(frame_idx, 'peak'|'valley')]
        if peak_mode == 'peaks_only':
            tagged = [(p, 'peak') for p in refined_peaks]
        elif peak_mode == 'valleys_only':
            tagged = [(v, 'valley') for v in refined_valleys]
        else:  # peaks_and_valleys
            tagged = [(p, 'peak') for p in refined_peaks] + \
                     [(v, 'valley') for v in refined_valleys]
        tagged.sort(key=lambda x: x[0])
        
        # 去重（峰和谷距离太近时保留峰）
        if len(tagged) > 1:
            deduped = [tagged[0]]
            for item in tagged[1:]:
                if abs(item[0] - deduped[-1][0]) >= min_dist:
                    deduped.append(item)
                elif item[1] == 'peak' and deduped[-1][1] == 'valley':
                    deduped[-1] = item  # 峰优先
            tagged = deduped
        
        # ── 同模态最小间隔约束（基于音频节拍间隔）──
        audio_interval = getattr(self.config, 'audio_beat_interval', 0.0)
        interval_ratio = getattr(self.config, 'min_interval_ratio', 0.8)
        if audio_interval > 0 and interval_ratio > 0 and len(tagged) > 1:
            min_same_dist = int(audio_interval * interval_ratio * fps)
            filtered = [tagged[0]]
            # 记录每种模态上一次通过的帧索引
            last_by_mode = {tagged[0][1]: tagged[0][0]}
            for item in tagged[1:]:
                frame_idx_item, mode = item
                prev_same = last_by_mode.get(mode)
                if prev_same is not None and (frame_idx_item - prev_same) < min_same_dist:
                    continue  # 同模态间隔过小，跳过
                filtered.append(item)
                last_by_mode[mode] = frame_idx_item
            if len(filtered) < len(tagged):
                print(f"[KeyframeFilter] 同模态最小间隔 {min_same_dist} 帧 "
                      f"({audio_interval*interval_ratio:.2f}s): "
                      f"{len(tagged)} → {len(filtered)} 关键帧")
            tagged = filtered
        
        beat_indices = [t[0] for t in tagged]
        beat_sources = [t[1] for t in tagged]
        beat_times = [p / fps for p in beat_indices]
        
        # 关键帧: 不再强制加首尾帧（首尾区间由匹配/导出时按原速处理）
        keyframe_indices = list(beat_indices)
        keyframe_sources = list(beat_sources)
        
        # 如果没检测到任何关键帧，保底取首帧
        if not keyframe_indices:
            keyframe_indices = [0]
            keyframe_sources = ['peak']
        
        result = (sorted(set(keyframe_indices)), beat_times)
        # 同步排序 sources
        idx_src = sorted(zip(keyframe_indices, keyframe_sources))
        # 去重后保留 source
        seen = set()
        unique_src = []
        for idx, src in idx_src:
            if idx not in seen:
                seen.add(idx)
                unique_src.append(src)
        keyframe_sources = unique_src
        
        if return_diagnostics:
            diagnostics = {
                "smooth_fine": smooth_fine,
                "smooth_coarse": smooth_coarse,
                "smooth_mid": smooth_mid,
                "acceleration": acceleration,
                "peaks_fine": peaks_fine,
                "peaks_coarse": peaks_coarse,
                "zero_crossings": zero_crossings,
                "vote_map": vote_map,
                "vote_radius": vote_radius,
                "signal_type": signal_type,
                "refined_peaks": refined_peaks,
                "refined_valleys": refined_valleys,
                "keyframe_sources": keyframe_sources,
                "sigma": sigma,
                "is_periodic": is_periodic,
            }
            return result[0], result[1], diagnostics
        
        return result[0], result[1], keyframe_sources
    
    def _estimate_sigma(
        self, signal: np.ndarray, fps: float
    ) -> tuple:
        """
        自适应平滑参数 σ 估计（周期性感知）。

        策略:
        1. 自相关分析检测信号是否具有周期性
        2. 若存在周期性 → σ = period / (2π)，基于主周期
        3. 若无周期性  → σ 基于信号特征尺度（典型峰间距）
           3a. 先用保守 σ 做初步峰值检测
           3b. 统计典型峰间距 → σ = median_spacing × 0.2
           3c. 若峰太少 → 基于噪声水平的保守 σ

        :return: (sigma, is_periodic)
        """
        N = len(signal)

        # ── Step 1: 自相关检测周期性 ──
        sig = signal - signal.mean()
        acf = np.correlate(sig, sig, mode='full')
        acf = acf[N - 1:]  # 正 lag [0..N-1]
        acf_norm = acf / (acf[0] + 1e-12)

        min_lag = max(3, int(fps * 0.15))   # 最小 ~150ms
        max_lag = min(N // 2, int(fps * 3.0))  # 最大 ~3s

        is_periodic = False
        period_frames = 0

        acf_search = acf_norm[min_lag:max_lag]
        if len(acf_search) > 3:
            acf_peaks, _ = find_peaks(acf_search, height=0.25, distance=min_lag)
            if len(acf_peaks) > 0:
                peak_height = acf_search[acf_peaks[0]]
                # ACF 第一峰高度 > 0.3 → 认定为周期性
                if peak_height > 0.3:
                    is_periodic = True
                    period_frames = acf_peaks[0] + min_lag

        if is_periodic:
            # ── 周期性信号: σ ≈ period / (2π) ──
            f0 = fps / period_frames
            sigma = np.clip(period_frames / (2 * np.pi), 1.0, 15.0)
            print(f"[σ] 周期信号: T={period_frames}帧 ({period_frames/fps:.2f}s), "
                  f"f₀={f0:.2f}Hz → σ={sigma:.2f}")
        else:
            # ── 非周期性信号: 基于特征尺度的 σ ──
            # 初步峰值检测（保守 σ=2）→ 统计典型峰间距
            prelim_smooth = gaussian_filter1d(signal, sigma=2.0)
            prelim_peaks, _ = find_peaks(
                prelim_smooth,
                height=np.mean(prelim_smooth) + 0.2 * np.std(prelim_smooth),
                distance=max(3, int(fps * 0.1)),
            )

            if len(prelim_peaks) >= 2:
                spacings = np.diff(prelim_peaks)
                median_spacing = float(np.median(spacings))
                # σ = 典型间距 × 0.2 (在 1/5 间距内平滑子峰噪声)
                sigma = np.clip(median_spacing * 0.2, 1.0, 15.0)
                print(f"[σ] 非周期信号: 典型峰间距={median_spacing:.1f}帧 "
                      f"({median_spacing/fps:.2f}s) → σ={sigma:.2f}")
            else:
                # 峰太少: 基于噪声水平的保守估计
                deriv = np.diff(signal)
                noise_mad = float(np.median(np.abs(deriv - np.median(deriv)))) * 1.4826
                sig_range = float(np.percentile(signal, 95) - np.percentile(signal, 5))
                if sig_range > 1e-8 and noise_mad > 1e-8:
                    snr = sig_range / noise_mad
                    sigma = np.clip(3.0 / np.log1p(snr), 1.0, 8.0)
                else:
                    sigma = 3.0
                print(f"[σ] 非周期信号(峰值过少): 噪声估计 → σ={sigma:.2f}")

        return float(sigma), is_periodic

    def _adaptive_peak_detect(
        self, signal: np.ndarray, fps: float, scale: str = "fine"
    ) -> List[int]:
        """
        自适应阈值峰值检测
        
        使用滑动窗口局部统计代替全局阈值，
        对变速运动（如慢动作段+快速段混合）更鲁棒
        """
        N = len(signal)
        
        if scale == "fine":
            min_dist = max(2, int(fps * 0.1))   # 100ms
            window_sec = 1.0
        else:
            min_dist = max(3, int(fps * 0.2))   # 200ms
            window_sec = 2.0
        
        window_frames = max(min_dist * 4, int(fps * window_sec))
        half_w = window_frames // 2
        
        # 计算局部自适应阈值
        local_threshold = np.zeros(N, dtype=np.float32)
        for i in range(N):
            lo = max(0, i - half_w)
            hi = min(N, i + half_w + 1)
            local_win = signal[lo:hi]
            mu = np.mean(local_win)
            std = np.std(local_win)
            local_threshold[i] = mu + self.config.peak_height_factor * std
        
        # 两阶段检测: 先找所有候选峰值，再按自适应阈值过滤
        all_peaks, properties = find_peaks(signal, distance=min_dist)
        
        # 保留超过局部阈值的峰值
        filtered = [p for p in all_peaks if signal[p] > local_threshold[p]]
        
        return filtered
    
    def _detect_from_signal(
        self, 
        signal: np.ndarray, 
        fps: float, 
        signal_type: str
    ) -> Tuple[List[int], List[float]]:
        """从单一信号检测节拍（保留作为内部接口）"""
        kf, bt, _ = self._detect_multiscale(signal, fps, signal_type)
        return kf, bt
    
    def _detect_weighted_fusion(
        self,
        velocities: np.ndarray,
        rot_grads: np.ndarray,
        fps: float,
        return_diagnostics: bool = False,
    ) -> Tuple[List[int], List[float]]:
        """
        改进加权融合检测
        
        改进点:
        1. 鲁棒归一化 (percentile 而非 min-max，抗异常值)
        2. 时序对齐（对齐两个信号的主要峰值）
        3. 逐帧置信度加权（信号质量差的帧权重降低）
        """
        # ── 鲁棒归一化: 用 2%-98% percentile 代替 min-max ──
        v_norm = self._robust_normalize(velocities)
        r_norm = self._robust_normalize(rot_grads)
        
        # ── 长度对齐 ──
        min_len = min(len(v_norm), len(r_norm))
        v_norm = v_norm[:min_len]
        r_norm = r_norm[:min_len]
        
        # ── 逐帧置信度: 基于局部信噪比 ──
        v_conf = self._compute_local_snr(v_norm)
        r_conf = self._compute_local_snr(r_norm)
        
        # ── 动态逐帧融合 ──
        alpha_base = self.config.vitpose_weight
        
        # 逐帧自适应权重: 在 alpha_base 基础上按置信度调整
        total_conf = v_conf + r_conf + 1e-8
        alpha_per_frame = alpha_base * (v_conf / total_conf) / max(alpha_base, 1e-8)
        alpha_per_frame = np.clip(alpha_per_frame, 0.1, 0.9)
        
        fused = alpha_per_frame * v_norm + (1 - alpha_per_frame) * r_norm
        
        result = self._detect_multiscale(fused, fps, "fused", return_diagnostics=return_diagnostics)
        
        if return_diagnostics:
            kf_indices, beat_times, ms_diag = result
            ms_diag["v_norm"] = v_norm
            ms_diag["r_norm"] = r_norm
            ms_diag["v_conf"] = v_conf
            ms_diag["r_conf"] = r_conf
            ms_diag["alpha_per_frame"] = alpha_per_frame
            ms_diag["fused"] = fused
            return kf_indices, beat_times, ms_diag
        
        kf_indices, beat_times, kf_sources = result
        return kf_indices, beat_times, kf_sources
    
    @staticmethod
    def _robust_normalize(signal: np.ndarray) -> np.ndarray:
        """鲁棒归一化: percentile 2%-98% 截断后归一化到 [0,1]"""
        lo = np.percentile(signal, 2)
        hi = np.percentile(signal, 98)
        if hi - lo < 1e-8:
            return np.zeros_like(signal)
        normed = np.clip((signal - lo) / (hi - lo), 0.0, 1.0)
        return normed
    
    @staticmethod
    def _compute_local_snr(signal: np.ndarray, window: int = 15) -> np.ndarray:
        """
        计算逐帧局部信噪比 (Signal-to-Noise Ratio)
        SNR = |local_mean| / (local_std + eps)
        
        高 SNR → 信号在该区域明确，置信度高
        低 SNR → 信号模糊/噪声大，置信度低
        """
        half_w = window // 2
        N = len(signal)
        snr = np.ones(N, dtype=np.float32)
        
        for i in range(N):
            lo = max(0, i - half_w)
            hi = min(N, i + half_w + 1)
            local = signal[lo:hi]
            mu = np.mean(local)
            std = np.std(local)
            snr[i] = abs(mu) / (std + 1e-6)
        
        # 归一化 SNR 到 [0.1, 1.0]
        if snr.max() > snr.min():
            snr = 0.1 + 0.9 * (snr - snr.min()) / (snr.max() - snr.min())
        
        return snr
    
    def _detect_adaptive_fusion(
        self,
        velocities: np.ndarray,
        rot_grads: np.ndarray,
        fps: float,
    ) -> Tuple[List[int], List[float], List[str]]:
        """
        自适应权重融合（改进版）
        
        改进: 使用滑动窗口局部方差做时变权重，
        而非全局方差的单一权重 — 处理视频中动作节奏变化
        """
        min_len = min(len(velocities), len(rot_grads))
        v = velocities[:min_len]
        r = rot_grads[:min_len]
        
        v_norm = self._robust_normalize(v)
        r_norm = self._robust_normalize(r)
        
        # 滑动窗口局部方差计算时变权重
        window = max(15, int(fps * 0.5))  # 0.5秒窗口
        half_w = window // 2
        
        fused = np.zeros(min_len, dtype=np.float32)
        
        for i in range(min_len):
            lo = max(0, i - half_w)
            hi = min(min_len, i + half_w + 1)
            
            v_local_var = np.var(v_norm[lo:hi])
            r_local_var = np.var(r_norm[lo:hi])
            
            total = v_local_var + r_local_var + 1e-8
            alpha_i = v_local_var / total
            
            # 混合: 一部分来自局部自适应，一部分来自用户配置
            alpha_i = 0.6 * alpha_i + 0.4 * self.config.vitpose_weight
            
            fused[i] = alpha_i * v_norm[i] + (1 - alpha_i) * r_norm[i]
        
        return self._detect_multiscale(fused, fps, "adaptive_fused")
    
    def _generate_visualizations(
        self,
        joint_coords: Optional[np.ndarray],
        joint_rotations: Optional[np.ndarray],
        fps: float,
    ):
        """生成可视化 GIF"""
        output_dir = os.path.join(os.path.dirname(self.video_path), "visualizations")
        os.makedirs(output_dir, exist_ok=True)
        
        if joint_coords is not None:
            self._generate_vitpose_gif(joint_coords, fps, output_dir)
        
        if joint_rotations is not None:
            self._generate_hmr2_gif(joint_rotations, fps, output_dir)
    
    def _generate_vitpose_gif(self, coords: np.ndarray, fps: float, output_dir: str):
        """
        生成 ViTPose 骨架动画 GIF
        在深色背景上绘制 COCO-17 骨架连线 + 关节点
        """
        try:
            from PIL import Image, ImageDraw
            
            # COCO-17 骨架连接
            skeleton = [
                (0, 1), (0, 2), (1, 3), (2, 4),      # 头部
                (5, 6),                                 # 双肩
                (5, 7), (7, 9),                         # 左臂
                (6, 8), (8, 10),                        # 右臂
                (5, 11), (6, 12), (11, 12),             # 躯干
                (11, 13), (13, 15),                     # 左腿
                (12, 14), (14, 16),                     # 右腿
            ]
            
            # 关节颜色分组
            JOINT_COLORS = {
                'head': (255, 200, 87),     # 金黄
                'shoulder': (87, 199, 255),  # 天蓝
                'elbow': (87, 199, 255),
                'wrist': (255, 135, 87),     # 橙色
                'torso': (150, 255, 150),    # 绿色
                'hip': (150, 255, 150),
                'knee': (150, 150, 255),     # 紫蓝
                'ankle': (255, 87, 199),     # 粉红
            }
            joint_color_map = [
                'head', 'head', 'head', 'head', 'head',      # 0-4
                'shoulder', 'shoulder', 'elbow', 'elbow',    # 5-8
                'wrist', 'wrist', 'hip', 'hip',              # 9-12
                'knee', 'knee', 'ankle', 'ankle',            # 13-16
            ]
            
            # 连线颜色
            LINE_COLORS = {
                'head': (255, 200, 87, 180),
                'arm': (87, 199, 255, 180),
                'torso': (150, 255, 150, 180),
                'leg': (150, 150, 255, 180),
            }
            line_color_map = [
                'head', 'head', 'head', 'head', 'head',
                'arm', 'arm', 'arm',
                'torso', 'torso', 'torso',
                'leg', 'leg', 'leg', 'leg',
            ]
            
            frames = []
            T = min(len(coords), int(self.config.gif_duration * fps))
            step = max(1, int(fps / self.config.gif_fps))
            
            # 计算全局坐标范围（用于固定视角）
            all_x = coords[:T, :, 0]
            all_y = coords[:T, :, 1]
            x_min, x_max = all_x.min(), all_x.max()
            y_min, y_max = all_y.min(), all_y.max()
            
            # 留边距
            pad = 20
            canvas_w, canvas_h = 400, 500
            
            def kp_to_canvas(kp):
                """将关节坐标映射到画布"""
                scale = min(
                    (canvas_w - 2 * pad) / max(x_max - x_min, 1),
                    (canvas_h - 2 * pad) / max(y_max - y_min, 1)
                )
                ox = (canvas_w - (x_max - x_min) * scale) / 2 - x_min * scale
                oy = (canvas_h - (y_max - y_min) * scale) / 2 - y_min * scale
                return kp * scale + np.array([ox, oy])
            
            for i in range(0, T, step):
                img = Image.new('RGB', (canvas_w, canvas_h), (30, 30, 30))
                draw = ImageDraw.Draw(img)
                
                kp = kp_to_canvas(coords[i])
                
                # 画骨架连线
                for j, (a, b) in enumerate(skeleton):
                    pt1 = tuple(kp[a].astype(int))
                    pt2 = tuple(kp[b].astype(int))
                    color_key = line_color_map[j] if j < len(line_color_map) else 'torso'
                    color = LINE_COLORS.get(color_key, (150, 255, 150, 180))[:3]
                    draw.line([pt1, pt2], fill=color, width=3)
                
                # 画关节点
                for j in range(17):
                    x, y = int(kp[j, 0]), int(kp[j, 1])
                    color_key = joint_color_map[j]
                    color = JOINT_COLORS[color_key]
                    r = 5 if j in [9, 10, 15, 16] else 4  # 末端执行器稍大
                    draw.ellipse([x-r, y-r, x+r, y+r], fill=color)
                
                # 帧号
                draw.text((10, 10), f"F{i}", fill=(200, 200, 200))
                
                frames.append(img)
            
            if frames:
                gif_path = os.path.join(output_dir, "vitpose_skeleton.gif")
                frames[0].save(
                    gif_path,
                    save_all=True,
                    append_images=frames[1:],
                    duration=int(1000 / self.config.gif_fps),
                    loop=0
                )
                self.gif_saved.emit(gif_path)
                
        except Exception as e:
            print(f"[GIF] ViTPose 生成失败: {e}")
    
    def _generate_hmr2_gif(self, rotations: np.ndarray, fps: float, output_dir: str):
        """
        生成 HMR2 SMPL 人体模型动画 GIF
        使用 SMPLRenderer 将旋转参数渲染为 3D 人体模型
        """
        try:
            from engines.smpl_renderer import SMPLRenderer
            from pose_extraction.weights_config import SMPL_NEUTRAL
            
            renderer = SMPLRenderer(SMPL_NEUTRAL)
            
            T = min(len(rotations), int(self.config.gif_duration * fps))
            step = max(1, int(fps / self.config.gif_fps))
            
            # 抽取帧
            sampled_poses = rotations[:T:step]  # (N, 24, 3)
            
            gif_path = os.path.join(output_dir, "hmr2_body_model.gif")
            renderer.render_gif(
                sampled_poses,
                gif_path,
                fps=self.config.gif_fps,
                azimuth=90.0,
                elevation=5.0,
                size=(400, 500),
                rotate_view=False,
                flip_y=True,
                stabilize_root=True,
            )
            self.gif_saved.emit(gif_path)
                
        except Exception as e:
            import traceback
            print(f"[GIF] HMR2 SMPL 渲染失败: {e}")
            traceback.print_exc()
            # 回退：使用简化的旋转量可视化
            self._generate_hmr2_fallback_gif(rotations, fps, output_dir)
    
    def _generate_hmr2_fallback_gif(self, rotations: np.ndarray, fps: float, output_dir: str):
        """HMR2 渲染失败的降级方案：旋转量条形图"""
        try:
            from PIL import Image
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import io
            
            frames = []
            T = min(len(rotations), int(self.config.gif_duration * fps))
            step = max(1, int(fps / self.config.gif_fps))
            
            for i in range(0, T, step):
                fig, ax = plt.subplots(figsize=(4, 4))
                rot_magnitudes = np.linalg.norm(rotations[i], axis=1)
                ax.barh(range(24), rot_magnitudes, color='steelblue')
                ax.set_xlabel('Rotation Magnitude (rad)')
                ax.set_ylabel('Joint Index')
                ax.set_title(f'Frame {i}')
                ax.set_xlim(0, np.pi)
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight')
                buf.seek(0)
                frames.append(Image.open(buf).copy())
                buf.close()
                plt.close()
            
            if frames:
                gif_path = os.path.join(output_dir, "hmr2_rotation.gif")
                frames[0].save(
                    gif_path,
                    save_all=True,
                    append_images=frames[1:],
                    duration=int(1000 / self.config.gif_fps),
                    loop=0
                )
                self.gif_saved.emit(gif_path)
                
        except Exception as e:
            print(f"[GIF] HMR2 降级渲染也失败: {e}")
