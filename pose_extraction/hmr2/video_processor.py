"""
pose_extraction/hmr2/video_processor.py

HMR2 视频处理器：逐帧重建三维人体，提取 SMPL 关节旋转量时序序列。

输出 θ_j(t)：第 t 帧、第 j 个 SMPL 关节的旋转量（轴角形式），
这是论文算法的核心输入之一。

使用示例：
    from pose_extraction.hmr2 import HMR2VideoProcessor
    proc = HMR2VideoProcessor()
    sequence = proc.process_video("dance.mp4")
    # 保存
    proc.save(sequence, "output/dance_hmr2.pkl")
    # 转为 BeatsMatching 接口格式
    pts = proc.to_pose_time_series(sequence, person_id=0)
    # 获取关节 18（left_elbow）的旋转量时序
    ts, rot = sequence.get_joint_rotation_sequence(joint_idx=18, person_id=0)
"""

from __future__ import annotations

import os
import pickle
import time
import warnings
from pathlib import Path
from typing import Callable, List, Optional, Union

import cv2
import numpy as np
from tqdm import tqdm

from .reconstructor import HMR2Reconstructor
from ..core.data_types import (
    HMR2FrameResult,
    HMR2Sequence,
    PoseTimeSeries,
    SMPL_JOINT_NAMES,
)


class HMR2VideoProcessor:
    """HMR2 视频处理器。

    Args:
        model_name:        "hmr2b" 或 "hmr2l"
        checkpoint_path:   本地 .ckpt 文件路径，None 则自动下载
        device:            "cuda" / "cpu" / "auto"
        yolo_model:        YOLO 权重路径，None 则不使用检测器
        conf_threshold:    YOLO 检测置信度阈值
        skip_frames:       每隔 N 帧处理一次（0 = 逐帧）
        max_persons:       每帧最多保留 N 个人（按检测顺序）
        save_vertices:     是否保存 SMPL 网格顶点（内存较大，默认 False）
    """

    def __init__(
        self,
        model_name: str = "hmr2b",
        checkpoint_path: Optional[str] = None,
        device: str = "auto",
        yolo_model: Optional[str] = "yolov8n.pt",
        conf_threshold: float = 0.35,
        skip_frames: int = 0,
        max_persons: int = 5,
        save_vertices: bool = False,
    ) -> None:
        self.reconstructor = HMR2Reconstructor(
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            device=device,
            yolo_model=yolo_model,
            conf_threshold=conf_threshold,
            save_vertices=save_vertices,
        )
        self.skip_frames = skip_frames
        self.max_persons = max_persons

    # ──────────────────────────────────────────────────────────────────────────
    # 主接口：视频 → HMR2Sequence
    # ──────────────────────────────────────────────────────────────────────────

    def process_video(
        self,
        video_path: Union[str, Path],
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        person_id: int = 0,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> HMR2Sequence:
        """逐帧处理视频，返回完整的 HMR2Sequence。

        Args:
            video_path:        输入视频路径
            start_frame:       起始帧（0-based）
            end_frame:         结束帧（None 则处理到末尾）
            person_id:         仅保留指定 person_id 的结果（0 = 首个检测到的人）
                               若需要多人，请设为 None（保留所有人）
            progress_callback: 进度回调 f(current, total)

        Returns:
            HMR2Sequence
        """
        video_path = str(video_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"无法打开视频文件：{video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if end_frame is None:
            end_frame = total
        end_frame = min(end_frame, total)

        print(f"[HMR2VideoProcessor] 视频：{video_path}")
        print(f"  帧率：{fps:.2f}，总帧数：{total}，"
              f"处理范围：[{start_frame}, {end_frame})")

        sequence = HMR2Sequence(
            video_path=video_path,
            fps=fps,
            total_frames=total,
        )

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        process_range = range(start_frame, end_frame)
        t0 = time.time()

        with tqdm(total=len(process_range), desc="HMR2 三维重建", unit="帧") as pbar:
            for global_idx in process_range:
                ret, bgr_frame = cap.read()
                if not ret:
                    break

                # 跳帧
                if self.skip_frames > 0 and (global_idx - start_frame) % (self.skip_frames + 1) != 0:
                    pbar.update(1)
                    continue

                timestamp = global_idx / fps

                try:
                    frame_results = self.reconstructor.reconstruct_frame(
                        bgr_frame,
                        frame_idx=global_idx,
                        timestamp=timestamp,
                    )
                except Exception as e:
                    warnings.warn(f"帧 {global_idx} 重建失败：{e}，跳过。")
                    pbar.update(1)
                    continue

                # 过滤人物
                for fr in frame_results:
                    if person_id is not None and fr.person_id != person_id:
                        continue
                    if fr.person_id > self.max_persons:
                        continue
                    sequence.frames.append(fr)
                    if fr.person_id not in sequence.person_ids:
                        sequence.person_ids.append(fr.person_id)

                if progress_callback is not None:
                    progress_callback(global_idx - start_frame + 1, len(process_range))

                pbar.update(1)

        cap.release()

        elapsed = time.time() - t0
        print(f"[HMR2VideoProcessor] 完成！处理帧数：{len(sequence.frames)}，"
              f"耗时：{elapsed:.1f}s，平均 {elapsed / max(1, len(sequence.frames)):.3f}s/帧")

        return sequence

    # ──────────────────────────────────────────────────────────────────────────
    # 转换为 PoseTimeSeries
    # ──────────────────────────────────────────────────────────────────────────

    def to_pose_time_series(
        self,
        sequence: HMR2Sequence,
        person_id: int = 0,
    ) -> PoseTimeSeries:
        """将 HMR2Sequence 转换为 PoseTimeSeries，供 BeatsMatching 使用。

        提取所有帧关节旋转量，构造 (T, 24, 3) 的 joint_rotations 数组。
        每一行 joint_rotations[t, j] = θ_j(t)，即第 t 帧第 j 个 SMPL 关节的
        旋转轴角量（对应论文式1/式2的输入）。

        Args:
            sequence:  process_video 的返回值
            person_id: 目标人物 ID

        Returns:
            PoseTimeSeries
        """
        timestamps, rotations = sequence.get_all_joints_rotation_sequence(
            person_id=person_id
        )

        return PoseTimeSeries(
            source="hmr2",
            video_path=sequence.video_path,
            fps=sequence.fps,
            timestamps=timestamps,
            joint_rotations=rotations.astype(np.float32) if len(rotations) > 0 else None,
            joint_names=SMPL_JOINT_NAMES,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 序列化 / 反序列化
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def save(
        sequence: HMR2Sequence,
        output_path: str,
        format: str = "pkl",
    ) -> str:
        """保存 HMR2Sequence 到文件。

        Args:
            sequence:    HMR2Sequence 对象
            output_path: 输出路径（.pkl 或 .npz）
            format:      "pkl" 或 "npz"

        Returns:
            实际写入的文件路径
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        if format == "pkl":
            with open(output_path, "wb") as f:
                pickle.dump(sequence, f, protocol=pickle.HIGHEST_PROTOCOL)

        elif format == "npz":
            # 按 person_id=0 导出为 npz
            ts_list, rot_list = [], []
            for fr in sequence.frames:
                if fr.person_id != 0:
                    continue
                ts_list.append(fr.timestamp)
                pose = fr.full_pose
                if pose is not None:
                    rot_list.append(pose)
                else:
                    rot_list.append(np.full((24, 3), np.nan, dtype=np.float32))

            if rot_list:
                np.savez_compressed(
                    output_path,
                    timestamps=np.array(ts_list, dtype=np.float64),
                    joint_rotations=np.array(rot_list, dtype=np.float32),  # (T, 24, 3)
                    joint_names=np.array(SMPL_JOINT_NAMES),
                    fps=sequence.fps,
                    total_frames=sequence.total_frames,
                )
            else:
                warnings.warn("序列中没有 person_id=0 的帧，npz 为空。")
        else:
            raise ValueError(f"不支持的格式：{format}")

        print(f"[HMR2VideoProcessor] 结果已保存：{output_path}")
        return output_path

    @staticmethod
    def load(input_path: str) -> HMR2Sequence:
        """从 .pkl 文件加载 HMR2Sequence。"""
        with open(input_path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def load_npz(input_path: str) -> "PoseTimeSeries":
        """从 .npz 文件加载并直接返回 PoseTimeSeries。"""
        data = np.load(input_path, allow_pickle=True)
        return PoseTimeSeries(
            source="hmr2",
            video_path=str(data.get("video_path", "")),
            fps=float(data["fps"]),
            timestamps=data["timestamps"],
            joint_rotations=data["joint_rotations"],
            joint_names=list(data["joint_names"]),
        )
