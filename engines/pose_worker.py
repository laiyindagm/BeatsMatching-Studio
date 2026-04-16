# engines/pose_worker.py
"""
基于深度学习的姿态提取工作线程
接口与 algorithms.py 中的 AlgorithmEngine.extract_motion_keyframes 同级

用法（在 algorithm_proxy.py 中替换原有提取逻辑）：
    from engines.pose_worker import PoseExtractionWorker
    self.pose_worker = PoseExtractionWorker(video_path, mode="auto")
    self.pose_worker.progress.connect(on_progress)
    self.pose_worker.finished.connect(on_finished)  # List[int]: keyframe indices
    self.pose_worker.pose_ready.connect(on_pose_ready)  # PoseData namedtuple
    self.pose_worker.start()

依赖：pose_unified conda 环境 + d:/毕业设计/weights/ 目录中的权重
"""

from __future__ import annotations

import os
import sys
import time
from typing import List, NamedTuple, Optional

import numpy as np
from PySide6.QtCore import QThread, Signal

# ─── 路径：让 pose_extraction 包可导入 ────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))          # BeatsMatching/engines/
_REPO = os.path.dirname(os.path.dirname(_HERE))             # d:/毕业设计/
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


# ─── 返回值结构 ───────────────────────────────────────────────────────────────
class PoseData(NamedTuple):
    """姿态提取完整结果，供高级消费方使用"""
    keyframe_indices: List[int]         # 检测到的动作关键帧帧号
    beat_timestamps:  List[float]       # 节拍时间点（秒）
    joint_coords:     Optional[np.ndarray]  # (T, 17, 2) float32，ViTPose 关节坐标
    joint_rotations:  Optional[np.ndarray]  # (T, 24, 3) float32，HMR2 SMPL 轴角
    fps: float
    total_frames: int


class PoseExtractionWorker(QThread):
    """
    后台线程：加载模型 → GPU推理 → 节拍检测 → 返回结果

    信号：
        progress(int)          : 0-100 进度百分比
        status(str)            : 状态文字（用于状态栏）
        finished(list)         : 关键帧帧号列表（与原 KeyframeExtractWorker 兼容）
        pose_ready(PoseData)   : 完整姿态数据（可选消费）
        error(str)             : 错误信息
    """

    progress  = Signal(int)
    status    = Signal(str)
    finished  = Signal(list)       # List[int]  keyframe indices
    pose_ready = Signal(object)    # PoseData
    error     = Signal(str)

    def __init__(
        self,
        video_path: str,
        mode: str = "auto",
        vitpose_frames: int = 60,
        hmr2_frames:    int = 30,
        parent=None,
    ):
        """
        Args:
            video_path:     视频文件路径
            mode:           "vitpose" | "hmr2" | "joint" | "auto"
                            auto = 先尝试 vitpose，失败则用 hmr2
            vitpose_frames: 参与推理的帧数（均匀抽样），速度 vs 精度权衡
            hmr2_frames:    HMR2 参与推理的帧数
        """
        super().__init__(parent)
        self.video_path     = video_path
        self.mode           = mode
        self.vitpose_frames = vitpose_frames
        self.hmr2_frames    = hmr2_frames
        self._cancelled     = False

    def cancel(self):
        self._cancelled = True

    # ──────────────────────────────────────────────────────────────────────────
    def run(self):
        try:
            self._run_impl()
        except Exception as e:
            import traceback
            self.error.emit(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

    def _run_impl(self):
        import cv2

        self.status.emit("正在初始化权重配置...")
        self.progress.emit(2)

        # ─── 权重配置 ───────────────────────────────────────────────────────
        from pose_extraction.weights_config import (
            HF_HOME, HMR2_CHECKPOINT, YOLO_WEIGHTS, VITPOSE_MODEL_ID
        )
        os.environ["HF_HOME"]              = HF_HOME
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"]       = "1"

        # ─── 读取视频元数据 ─────────────────────────────────────────────────
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.error.emit(f"无法打开视频: {self.video_path}")
            return
        fps         = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if self._cancelled: return

        # ─── 确定实际运行模式 ───────────────────────────────────────────────
        mode = self.mode
        if mode == "auto":
            mode = "vitpose"   # 默认用 vitpose（更快）

        joint_coords_arr   = None
        joint_rotations_arr = None
        keyframe_indices   = []
        beat_timestamps    = []

        # ══════════════════════════════════════════════════════════════════════
        # 管线 A：ViTPose
        # ══════════════════════════════════════════════════════════════════════
        if mode in ("vitpose", "joint"):
            self.status.emit("正在加载 ViTPose 模型...")
            self.progress.emit(5)

            from pose_extraction.vitpose.detector import ViTPoseDetector
            detector = ViTPoseDetector(
                vitpose_model=VITPOSE_MODEL_ID,
                yolo_model=YOLO_WEIGHTS,
            )
            self.progress.emit(15)
            if self._cancelled: return

            # 均匀抽帧
            n_frames = min(self.vitpose_frames, total_frames)
            frame_idx_list = [int(i * total_frames / n_frames) for i in range(n_frames)]

            self.status.emit(f"ViTPose 推理中 (0/{n_frames})...")
            kp_list, valid_indices = [], []

            cap = cv2.VideoCapture(self.video_path)
            for step, fidx in enumerate(frame_idx_list):
                if self._cancelled: break
                cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
                ret, frame = cap.read()
                if not ret: continue

                result = detector.detect_frame(frame, frame_idx=fidx, timestamp=fidx/fps)
                if result.persons:
                    kp = result.persons[0]   # (17,3)
                    kp_list.append(kp[:, :2].astype(np.float32))
                    valid_indices.append(fidx)
                else:
                    kp_list.append(np.zeros((17, 2), dtype=np.float32))
                    valid_indices.append(fidx)

                pct = 15 + int(step / n_frames * 40)
                self.progress.emit(pct)
                if step % 10 == 0:
                    self.status.emit(f"ViTPose 推理中 ({step}/{n_frames})...")
            cap.release()

            joint_coords_arr = np.stack(kp_list) if kp_list else None  # (T,17,2)

            # 节拍检测
            if joint_coords_arr is not None and len(joint_coords_arr) >= 2:
                kf_idx, beats = self._detect_beats_vitpose(
                    joint_coords_arr, valid_indices, fps
                )
                keyframe_indices = kf_idx
                beat_timestamps  = beats
            self.progress.emit(60)

        # ══════════════════════════════════════════════════════════════════════
        # 管线 B：HMR2
        # ══════════════════════════════════════════════════════════════════════
        if mode in ("hmr2", "joint"):
            self.status.emit("正在加载 HMR2 模型...")
            self.progress.emit(62)

            from pose_extraction.hmr2.reconstructor import HMR2Reconstructor
            reconstructor = HMR2Reconstructor(
                checkpoint_path=HMR2_CHECKPOINT,
                yolo_model=YOLO_WEIGHTS,
            )
            self.progress.emit(72)
            if self._cancelled: return

            n_hmr = min(self.hmr2_frames, total_frames)
            frame_idx_hmr = [int(i * total_frames / n_hmr) for i in range(n_hmr)]

            self.status.emit(f"HMR2 推理中 (0/{n_hmr})...")
            rot_list, valid_hmr = [], []

            cap = cv2.VideoCapture(self.video_path)
            for step, fidx in enumerate(frame_idx_hmr):
                if self._cancelled: break
                cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
                ret, frame = cap.read()
                if not ret: continue

                res_list = reconstructor.reconstruct_frame(frame, frame_idx=fidx, timestamp=fidx/fps)
                if res_list:
                    r = res_list[0]
                    go = r.global_orient  # (1,3)
                    bp = r.body_pose      # (23,3)
                    if go is not None and bp is not None:
                        full = np.vstack([go.reshape(1,3), bp])  # (24,3)
                        rot_list.append(full.astype(np.float32))
                        valid_hmr.append(fidx)
                    else:
                        rot_list.append(np.zeros((24,3), dtype=np.float32))
                        valid_hmr.append(fidx)
                else:
                    rot_list.append(np.zeros((24,3), dtype=np.float32))
                    valid_hmr.append(fidx)

                pct = 72 + int(step / n_hmr * 20)
                self.progress.emit(pct)
                if step % 5 == 0:
                    self.status.emit(f"HMR2 推理中 ({step}/{n_hmr})...")
            cap.release()

            joint_rotations_arr = np.stack(rot_list) if rot_list else None  # (T,24,3)

            if mode == "hmr2" and joint_rotations_arr is not None:
                kf_idx, beats = self._detect_beats_hmr2(
                    joint_rotations_arr, valid_hmr, fps
                )
                keyframe_indices = kf_idx
                beat_timestamps  = beats

        # ══════════════════════════════════════════════════════════════════════
        # 融合（joint 模式）
        # ══════════════════════════════════════════════════════════════════════
        if mode == "joint" and joint_coords_arr is not None and joint_rotations_arr is not None:
            kf_idx, beats = self._detect_beats_joint(
                joint_coords_arr, valid_indices,
                joint_rotations_arr, valid_hmr,
                fps
            )
            keyframe_indices = kf_idx
            beat_timestamps  = beats

        # ─── 汇总 ──────────────────────────────────────────────────────────
        self.progress.emit(98)
        self.status.emit(f"检测到 {len(keyframe_indices)} 个运动关键帧，{len(beat_timestamps)} 个节拍")

        pose_data = PoseData(
            keyframe_indices=keyframe_indices,
            beat_timestamps=beat_timestamps,
            joint_coords=joint_coords_arr,
            joint_rotations=joint_rotations_arr,
            fps=fps,
            total_frames=total_frames,
        )

        self.progress.emit(100)
        self.finished.emit(keyframe_indices)
        self.pose_ready.emit(pose_data)

    # ──────────────────────────────────────────────────────────────────────────
    # 节拍检测算法
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _detect_beats_vitpose(
        kp_arr: np.ndarray,    # (T, 17, 2)
        frame_indices: list,
        fps: float,
        sigma: float = 1.5,
    ):
        """ViTPose 管线：关节位移速度峰值检测"""
        from scipy.ndimage import gaussian_filter1d
        from scipy.signal import find_peaks

        speeds, times = [], []
        for i in range(1, len(kp_arr)):
            delta = kp_arr[i] - kp_arr[i-1]
            speed = float(np.sqrt((delta**2).sum(axis=-1)).mean())
            speeds.append(speed)
            times.append(frame_indices[i] / fps)

        speeds = np.array(speeds)
        smoothed = gaussian_filter1d(speeds, sigma=sigma)

        mu, std = smoothed.mean(), smoothed.std()
        thr = mu + 0.5 * std

        peaks, _ = find_peaks(smoothed, height=thr, distance=3)
        beat_t = [times[p] for p in peaks]
        kf_idx = [frame_indices[p+1] for p in peaks]  # +1 因为 speeds 从第1帧开始

        # 补充首尾关键帧
        if not kf_idx or kf_idx[0] != frame_indices[0]:
            kf_idx.insert(0, frame_indices[0])
        if kf_idx[-1] != frame_indices[-1]:
            kf_idx.append(frame_indices[-1])

        return sorted(set(kf_idx)), beat_t

    @staticmethod
    def _detect_beats_hmr2(
        rot_arr: np.ndarray,   # (T, 24, 3)
        frame_indices: list,
        fps: float,
        sigma: float = 1.5,
    ):
        """HMR2 管线：SMPL 旋转梯度谷值检测"""
        from scipy.ndimage import gaussian_filter1d
        from scipy.signal import find_peaks

        grads, times = [], []
        for i in range(1, len(rot_arr)):
            diff = rot_arr[i] - rot_arr[i-1]
            grad = float((diff**2).sum())
            grads.append(grad)
            times.append(frame_indices[i] / fps)

        grads = np.array(grads)
        smoothed = gaussian_filter1d(grads, sigma=sigma)

        # 找极大值（运动剧烈帧）
        mu, std = smoothed.mean(), smoothed.std()
        thr = mu + 0.3 * std
        peaks, _ = find_peaks(smoothed, height=thr, distance=3)
        beat_t = [times[p] for p in peaks]
        kf_idx = [frame_indices[p+1] for p in peaks]

        if not kf_idx or kf_idx[0] != frame_indices[0]:
            kf_idx.insert(0, frame_indices[0])
        if kf_idx[-1] != frame_indices[-1]:
            kf_idx.append(frame_indices[-1])

        return sorted(set(kf_idx)), beat_t

    @staticmethod
    def _detect_beats_joint(
        kp_arr,  frame_idx_vit,
        rot_arr, frame_idx_hmr,
        fps: float,
        alpha: float = 0.6,   # vitpose 权重
    ):
        """
        融合信号：将 vitpose 速度信号（对齐到公共时间轴）和
        hmr2 旋转梯度信号加权融合后检测峰值。
        """
        from scipy.ndimage import gaussian_filter1d
        from scipy.signal import find_peaks

        # vitpose 速度序列
        vit_speeds, vit_times = [], []
        for i in range(1, len(kp_arr)):
            delta = kp_arr[i] - kp_arr[i-1]
            vit_speeds.append(float(np.sqrt((delta**2).sum(axis=-1)).mean()))
            vit_times.append(frame_idx_vit[i] / fps)

        # hmr2 梯度序列
        hmr_grads, hmr_times = [], []
        for i in range(1, len(rot_arr)):
            diff = rot_arr[i] - rot_arr[i-1]
            hmr_grads.append(float((diff**2).sum()))
            hmr_times.append(frame_idx_hmr[i] / fps)

        if not vit_speeds or not hmr_grads:
            # 回退到 vitpose
            return PoseExtractionWorker._detect_beats_vitpose(
                kp_arr, frame_idx_vit, fps)

        # 各自归一化到 [0,1]
        def normalize(x):
            x = np.array(x, dtype=np.float32)
            r = x.max() - x.min()
            return (x - x.min()) / (r + 1e-8)

        vs_n = normalize(vit_speeds)
        hg_n = normalize(hmr_grads)

        # 插值 hmr 到 vit 的时间轴
        hg_interp = np.interp(vit_times, hmr_times, hg_n,
                              left=hg_n[0], right=hg_n[-1])

        fused = alpha * vs_n + (1-alpha) * hg_interp
        smoothed = gaussian_filter1d(fused, sigma=1.5)
        mu, std = smoothed.mean(), smoothed.std()
        thr = mu + 0.5 * std
        peaks, _ = find_peaks(smoothed, height=thr, distance=3)

        beat_t = [vit_times[p] for p in peaks]
        kf_idx = [frame_idx_vit[p+1] for p in peaks]

        if not kf_idx or kf_idx[0] != frame_idx_vit[0]:
            kf_idx.insert(0, frame_idx_vit[0])
        if kf_idx[-1] != frame_idx_vit[-1]:
            kf_idx.append(frame_idx_vit[-1])

        return sorted(set(kf_idx)), beat_t
