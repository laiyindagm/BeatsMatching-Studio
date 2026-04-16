# engines/algorithm_proxy.py
"""
算法代理：管理帧差法（快速）和深度学习（精确）两种关键帧提取策略
通过 AlgorithmProxy.set_use_deep_learning(True/False) 切换
"""

from PySide6.QtCore import QThread, Signal, QObject
from .algorithms import AlgorithmEngine
from core.data_model import ProjectModel
from core.structures import Keyframe, KeyframeSource


# ─────────────────────────────────────────────────────────────────────────────
# 帧差法提取（原版，快速，不需要 GPU）
# ─────────────────────────────────────────────────────────────────────────────
class KeyframeExtractWorker(QThread):
    finished = Signal(list)   # List[int]: 关键帧索引
    progress = Signal(int)
    error    = Signal(str)

    def __init__(self, frames, threshold=0.5):
        super().__init__()
        self.frames    = frames
        self.threshold = threshold

    def run(self):
        try:
            kfs = AlgorithmEngine.extract_motion_keyframes(
                self.frames, threshold=self.threshold
            )
            self.finished.emit(kfs)
        except Exception as e:
            self.error.emit(str(e))


# ─────────────────────────────────────────────────────────────────────────────
# 深度学习提取（精确，需要 GPU + pose_unified 环境）
# ─────────────────────────────────────────────────────────────────────────────
class DeepPoseWorker(QThread):
    """
    深度学习姿态提取工作线程
    
    直接在 run() 中调用 EnhancedPoseExtractionWorker 的核心逻辑，
    不再使用信号桥接，避免跨线程信号问题。
    """
    finished   = Signal(list)    # List[int]: 关键帧索引
    progress   = Signal(int)
    status_msg = Signal(str)
    error      = Signal(str)
    pose_ready = Signal(object)  # PoseData
    gif_saved  = Signal(str)     # GIF 保存路径

    def __init__(self, video_path: str, config=None):
        super().__init__()
        self.video_path = video_path
        self.config = config  # AlgorithmConfig 实例

    def run(self):
        try:
            from engines.enhanced_pose_worker import EnhancedPoseExtractionWorker
            from core.algorithm_config import AlgorithmConfig

            config = self.config or AlgorithmConfig()
            worker = EnhancedPoseExtractionWorker(
                self.video_path, config
            )

            # 连接信号（在子线程中直接连接，信号会排队到主线程）
            worker.progress.connect(self.progress.emit)
            worker.status.connect(self.status_msg.emit)
            worker.finished.connect(self.finished.emit)
            worker.pose_ready.connect(self.pose_ready.emit)
            worker.error.connect(self.error.emit)
            worker.gif_saved.connect(self.gif_saved.emit)

            # 直接在当前线程执行核心逻辑
            worker._run_impl()

        except Exception as e:
            import traceback
            self.error.emit(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


# ─────────────────────────────────────────────────────────────────────────────
# 匹配 Worker（不变）
# ─────────────────────────────────────────────────────────────────────────────
class MatchWorker(QThread):
    finished = Signal(list)
    status = Signal(str)

    def __init__(self, kf_indices, beats, fps, config=None, algorithm="dp"):
        super().__init__()
        self.kf_indices = kf_indices
        self.beats      = beats
        self.fps        = fps
        self.config     = config
        self.algorithm  = algorithm

    def run(self):
        n_kf = len(self.kf_indices)
        n_bt = len(self.beats)
        self.status.emit(f"匹配中... ({n_kf} 关键帧 × {n_bt} 节拍, 算法: {self.algorithm})")
        new_times = AlgorithmEngine.match_beats(
            self.kf_indices, self.beats, self.fps, config=self.config
        )
        self.status.emit(f"匹配完成，{len(new_times)} 个时间点已对齐")
        self.finished.emit(new_times)


# ─────────────────────────────────────────────────────────────────────────────
# AlgorithmProxy（主控，支持模式切换）
# ─────────────────────────────────────────────────────────────────────────────
class AlgorithmProxy(QObject):
    """
    管理算法线程，连接 UI 和 Model。

    use_deep_learning=False → 帧差法（原版）
    use_deep_learning=True  → ViTPose/HMR2（需要 GPU + pose_unified 环境）
    """

    # 透传进度和状态给主窗口
    extraction_progress = Signal(int)
    extraction_status   = Signal(str)

    def __init__(self, model: ProjectModel):
        super().__init__()
        self.model             = model
        self.extract_worker    = None
        self.match_worker      = None
        self._use_deep         = False   # 默认帧差法
        self._deep_mode        = "vitpose"
        self._config           = None    # AlgorithmConfig

    def set_use_deep_learning(self, enabled: bool, mode: str = "vitpose",
                              config=None):
        """切换算法模式"""
        self._use_deep  = enabled
        self._deep_mode = mode
        self._config    = config

    def start_keyframe_extraction(self):
        # 如果有正在运行的 worker，先停止
        if self.extract_worker is not None and self.extract_worker.isRunning():
            if hasattr(self.extract_worker, 'cancel'):
                self.extract_worker.cancel()
            self.extract_worker = None

        if self._use_deep:
            self._start_deep_extraction()
        else:
            self._start_fast_extraction()

    # ── 帧差法 ────────────────────────────────────────────────────────────────
    def _start_fast_extraction(self):
        frames = self.model.frames
        if not frames:
            print("No video frames cached! Please wait for loading to finish.")
            return
        self.extract_worker = KeyframeExtractWorker(frames)
        self.extract_worker.finished.connect(self.on_extraction_finished)
        self.extract_worker.start()
        self.extraction_status.emit("帧差法提取中...")
        print("Starting frame-diff keyframe extraction...")

    # ── 深度学习 ──────────────────────────────────────────────────────────────
    def _start_deep_extraction(self):
        video_path = self.model.video_path
        if not video_path:
            print("No video path set in model.")
            return

        # 优先使用 model 上的 algo_config（来自 GUI 算法配置面板），否则使用手动传入的
        config = getattr(self.model, 'algo_config', None) or self._config

        # 将音频节拍平均间隔注入 config，用于关键帧最小间隔约束
        if config and self.model.beats and len(self.model.beats) >= 2:
            import numpy as np
            intervals = np.diff(self.model.beats)
            config.audio_beat_interval = float(np.mean(intervals))
            print(f"[Proxy] 音频节拍平均间隔: {config.audio_beat_interval:.3f}s")

        self.extract_worker = DeepPoseWorker(
            video_path,
            config=config,
        )
        self.extract_worker.progress.connect(self.extraction_progress)
        self.extract_worker.status_msg.connect(self.extraction_status)
        self.extract_worker.finished.connect(self.on_extraction_finished)
        self.extract_worker.pose_ready.connect(self._on_pose_ready)
        self.extract_worker.error.connect(self._on_error)
        self.extract_worker.start()
        self.extraction_status.emit("深度学习提取启动...")
        print(f"Starting deep pose extraction (mode={self._deep_mode})...")

    def on_extraction_finished(self, indices):
        print(f"Extraction done. Found {len(indices)} keyframes.")
        # 清空旧关键帧，重新创建
        self.model.keyframes.clear()
        fps = self.model.fps or 30.0
        # 保存提取索引，用于后续 _on_pose_ready 匹配 source
        self._last_extracted_indices = list(indices)
        indices_set = set(indices)
        for idx in indices:
            t = idx / fps
            self.model.add_keyframe(idx, t)

        # ── 插入首尾 BOUNDARY 关键帧（业务用途：TimeMapper/播放循环/导出） ──
        # 仅当首尾帧未被提取算法检出时才标记为 BOUNDARY
        last_frame_idx = max(0, len(self.model.frames) - 1) if self.model.frames else \
                         max(0, round(self.model.duration * fps) - 1)
        if 0 not in indices_set:
            self.model.add_keyframe(0, 0.0, source=KeyframeSource.BOUNDARY)
        if last_frame_idx > 0 and last_frame_idx not in indices_set:
            self.model.add_keyframe(last_frame_idx, last_frame_idx / fps,
                                    source=KeyframeSource.BOUNDARY)

        self.model.data_loaded.emit()
        self.extraction_status.emit(f"提取完成，{len(indices)} 个关键帧")
        # 清理 worker 引用
        self.extract_worker = None

    def _on_pose_ready(self, pose_data):
        """接收完整姿态数据，存入 model 供可视化面板使用"""
        print(f"PoseData ready: {len(pose_data.keyframe_indices)} kf, "
              f"{len(pose_data.beat_timestamps)} beats")
        # 可选：把节拍时间点存入 model
        if pose_data.beat_timestamps and not self.model.beats:
            self.model.beats = pose_data.beat_timestamps
            self.model.audio_data_loaded.emit()

        # 用关键帧来源信息补充已创建的 Keyframe 对象
        kf_sources = getattr(pose_data, 'keyframe_sources', None)
        extracted = getattr(self, '_last_extracted_indices', [])
        if kf_sources and extracted:
            from core.structures import KeyframeSource
            _map = {
                'peak': KeyframeSource.PEAK,
                'valley': KeyframeSource.VALLEY,
                'boundary': KeyframeSource.BOUNDARY,
            }
            # 建立 idx → source 的映射
            idx_to_source = {}
            for idx, src_str in zip(extracted, kf_sources):
                idx_to_source[idx] = _map.get(src_str, KeyframeSource.PEAK)
            # 应用到 model.keyframes（可能比 extracted 多出首尾锚点）
            for kf in self.model.keyframes:
                if kf.frame_idx in idx_to_source:
                    kf.source = idx_to_source[kf.frame_idx]
            self.model.keyframes_changed.emit()

        # 存储姿态数据到 model（供位姿可视化面板使用）
        self.model.set_pose_data(
            coords=pose_data.joint_coords,
            rotations=pose_data.joint_rotations,
            coords_raw=getattr(pose_data, 'joint_coords_raw', None),
            rotations_raw=getattr(pose_data, 'joint_rotations_raw', None),
            valid_vit=getattr(pose_data, 'valid_mask_vit', None),
            valid_hmr=getattr(pose_data, 'valid_mask_hmr', None),
        )

    def _on_error(self, msg):
        print(f"[PoseWorker ERROR] {msg}")
        self.extraction_status.emit(f"错误: {msg[:80]}")

    # ── 匹配 ──────────────────────────────────────────────────────────────────
    def start_auto_match(self):
        if not self.model.keyframes or not self.model.beats:
            self.extraction_status.emit("⚠️ 缺少关键帧或节拍数据，无法匹配")
            return

        # ── 只匹配非 BOUNDARY 关键帧（BOUNDARY 用于业务但不参与节拍对齐） ──
        match_kfs = [k for k in self.model.keyframes
                     if k.source != KeyframeSource.BOUNDARY]
        if not match_kfs:
            self.extraction_status.emit("⚠️ 无可匹配的关键帧（仅有首尾边界帧）")
            return

        kf_indices = [k.frame_idx for k in match_kfs]
        beats      = self.model.beats
        fps        = self.model.fps or 30.0

        # 应用音频偏移：节拍时间 + offset = 视频时间轴上的实际位置
        offset = self.model.audio_offset
        beats_shifted = [b + offset for b in beats]

        config = getattr(self.model, 'algo_config', None) or self._config
        algorithm = getattr(config, 'match_algorithm', 'dp') if config else 'dp'
        self.match_worker = MatchWorker(kf_indices, beats_shifted, fps, config=config, algorithm=algorithm)
        self.match_worker.finished.connect(self.on_match_finished)
        self.match_worker.status.connect(self.extraction_status)
        self.match_worker.start()
        self.extraction_status.emit("正在匹配关键帧与节拍...")

    def on_match_finished(self, new_times):
        fps = self.model.fps or 30.0

        # ── 只更新非 BOUNDARY 关键帧的 output_time ──
        match_kfs = [k for k in self.model.keyframes
                     if k.source != KeyframeSource.BOUNDARY]
        for i, t in enumerate(new_times):
            if i < len(match_kfs):
                match_kfs[i].output_time = t

        # ── 重算 BOUNDARY 关键帧：保持与相邻提取帧的原速关系 ──
        if match_kfs:
            first_match = match_kfs[0]
            last_match  = match_kfs[-1]
            for bkf in self.model.keyframes:
                if bkf.source != KeyframeSource.BOUNDARY:
                    continue
                if bkf.frame_idx <= first_match.frame_idx:
                    # 头部 BOUNDARY: output_time = first_match.t - Δframes/fps
                    bkf.output_time = first_match.output_time - \
                        (first_match.frame_idx - bkf.frame_idx) / fps
                else:
                    # 尾部 BOUNDARY: output_time = last_match.t + Δframes/fps
                    bkf.output_time = last_match.output_time + \
                        (bkf.frame_idx - last_match.frame_idx) / fps

        self.model.keyframes.sort()
        self.model.keyframes_changed.emit()
        self.extraction_status.emit(f"✅ 匹配完成，{len(new_times)} 个关键帧已对齐")
        self.match_worker = None

    def reset_speed(self):
        """重置所有关键帧的变速——将 output_time 恢复为原始时间 (frame_idx / fps)"""
        fps = self.model.fps or 30.0
        if not self.model.keyframes:
            return
        for kf in self.model.keyframes:
            kf.output_time = kf.frame_idx / fps
        self.model.keyframes_changed.emit()
        self.extraction_status.emit("↻ 已重置所有变速，恢复原始时间")
