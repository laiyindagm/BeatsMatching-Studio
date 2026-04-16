"""
pose_extraction/vitpose/video_processor.py

ViTPose 视频处理器：逐帧读取视频，调用 ViTPoseDetector，
输出包含完整时序关节坐标的 ViTPoseSequence 对象，
再转化为对接 BeatsMatching 的 PoseTimeSeries。

使用示例：
    from pose_extraction.vitpose import ViTPoseVideoProcessor
    proc = ViTPoseVideoProcessor()
    sequence = proc.process_video("dance.mp4")
    # 保存
    proc.save(sequence, "output/dance_vitpose.pkl")
    # 转为 BeatsMatching 接口格式
    pts = proc.to_pose_time_series(sequence, person_id=0)
"""

from __future__ import annotations

import json
import os
import pickle
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from tqdm import tqdm

from .detector import ViTPoseDetector
from ..core.data_types import (
    ViTPoseFrameResult,
    ViTPoseSequence,
    PoseTimeSeries,
)

# 从 detector 导入关节名称
from .detector import COCO17_JOINT_NAMES, COCO17_SKELETON


class ViTPoseVideoProcessor:
    """ViTPose 视频处理器。

    Args:
        vitpose_model:  ViTPose 模型名称或本地路径
        device:         "cuda" / "cpu" / "auto"
        yolo_model:     YOLO 权重路径，None 则不使用检测器（全图）
        conf_threshold: YOLO 检测置信度阈值
        batch_size:     逐帧推理的批大小（当前为 1，预留扩展）
        skip_frames:    每隔 N 帧处理一次（0 = 逐帧）
        max_persons:    每帧最多保留 N 个人（按面积排序）
    """

    def __init__(
        self,
        vitpose_model: str = "usyd-community/vitpose-base-simple",
        device: str = "auto",
        yolo_model: Optional[str] = "yolov8n.pt",
        conf_threshold: float = 0.35,
        skip_frames: int = 0,
        max_persons: int = 5,
    ) -> None:
        self.detector = ViTPoseDetector(
            vitpose_model=vitpose_model,
            device=device,
            yolo_model=yolo_model,
            conf_threshold=conf_threshold,
        )
        self.skip_frames = skip_frames
        self.max_persons = max_persons

    # ──────────────────────────────────────────────────────────────────────────
    # 主接口：视频 → ViTPoseSequence
    # ──────────────────────────────────────────────────────────────────────────

    def process_video(
        self,
        video_path: Union[str, Path],
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        visualize: bool = False,
        vis_output_path: Optional[str] = None,
    ) -> ViTPoseSequence:
        """逐帧处理视频，返回完整的 ViTPoseSequence。

        Args:
            video_path:        输入视频路径
            start_frame:       起始帧（0-based）
            end_frame:         结束帧（None 则处理到末尾）
            progress_callback: 进度回调 f(current_frame, total_frames)
            visualize:         是否生成可视化视频
            vis_output_path:   可视化视频输出路径（默认在原视频旁）

        Returns:
            ViTPoseSequence
        """
        video_path = str(video_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"无法打开视频文件：{video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if end_frame is None:
            end_frame = total
        end_frame = min(end_frame, total)

        print(f"[ViTPoseVideoProcessor] 视频：{video_path}")
        print(f"  分辨率：{W}x{H}，帧率：{fps:.2f}，总帧数：{total}，"
              f"处理范围：[{start_frame}, {end_frame})")

        # 可视化输出
        vis_writer = None
        if visualize:
            if vis_output_path is None:
                p = Path(video_path)
                vis_output_path = str(p.parent / (p.stem + "_vitpose_vis.mp4"))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            vis_writer = cv2.VideoWriter(vis_output_path, fourcc, fps, (W, H))

        sequence = ViTPoseSequence(
            video_path=video_path,
            fps=fps,
            total_frames=total,
            joint_names=COCO17_JOINT_NAMES,
            skeleton=COCO17_SKELETON,
        )

        # 定位到起始帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        process_range = range(start_frame, end_frame)
        t0 = time.time()

        with tqdm(total=len(process_range), desc="ViTPose 关节检测", unit="帧") as pbar:
            for global_idx in process_range:
                ret, bgr_frame = cap.read()
                if not ret:
                    break

                # skip_frames 跳帧
                if self.skip_frames > 0 and (global_idx - start_frame) % (self.skip_frames + 1) != 0:
                    # 写原帧到可视化视频（若启用）
                    if vis_writer is not None:
                        vis_writer.write(bgr_frame)
                    pbar.update(1)
                    continue

                timestamp = global_idx / fps

                # 推理
                frame_result = self.detector.detect_frame(
                    bgr_frame,
                    frame_idx=global_idx,
                    timestamp=timestamp,
                )

                # 限制人数（按关节置信度之和降序）
                if len(frame_result.persons) > self.max_persons:
                    scored = sorted(
                        frame_result.persons.items(),
                        key=lambda kv: float(np.sum(kv[1][:, 2])),
                        reverse=True,
                    )
                    frame_result.persons = {
                        new_id: old_kpts
                        for new_id, (_, old_kpts) in enumerate(scored[:self.max_persons])
                    }

                sequence.frames.append(frame_result)

                # 可视化
                if vis_writer is not None:
                    vis_bgr = self.detector.visualize_frame(bgr_frame, frame_result)
                    vis_writer.write(vis_bgr)

                # 进度回调
                if progress_callback is not None:
                    progress_callback(global_idx - start_frame + 1, len(process_range))

                pbar.update(1)

        cap.release()
        if vis_writer is not None:
            vis_writer.release()
            print(f"[ViTPoseVideoProcessor] 可视化视频已保存：{vis_output_path}")

        elapsed = time.time() - t0
        print(f"[ViTPoseVideoProcessor] 完成！处理帧数：{len(sequence.frames)}，"
              f"耗时：{elapsed:.1f}s，平均 {elapsed / max(1, len(sequence.frames)):.3f}s/帧")

        return sequence

    # ──────────────────────────────────────────────────────────────────────────
    # 转换为 PoseTimeSeries（对接 BeatsMatching）
    # ──────────────────────────────────────────────────────────────────────────

    def to_pose_time_series(
        self,
        sequence: ViTPoseSequence,
        person_id: int = 0,
        conf_threshold: float = 0.3,
    ) -> PoseTimeSeries:
        """将 ViTPoseSequence 转换为 PoseTimeSeries，供 BeatsMatching 使用。

        提取指定人物（person_id）在所有帧的关节坐标，
        构造 (T, J, 2) 的 joint_coords 和 (T, J) 的 joint_coords_conf。

        Args:
            sequence:        process_video 的返回值
            person_id:       目标人物 ID（通常 0 = 最大置信度的人）
            conf_threshold:  置信度阈值

        Returns:
            PoseTimeSeries
        """
        timestamps, coords_with_conf = sequence.get_all_joints_trajectory(
            person_id=person_id,
            conf_threshold=0.0,   # 暂不过滤，conf 保存在单独字段
        )

        # coords_with_conf 实际由 get_all_joints_trajectory 返回 (T, J, 2)，
        # 置信度已被置 NaN。我们需要重新提取 conf。
        # 重新遍历，同时提取 (x,y) 和 conf
        T = len(sequence.frames)
        J = len(sequence.joint_names) if sequence.joint_names else 17
        joint_coords = np.full((T, J, 2), np.nan, dtype=np.float32)
        joint_conf = np.zeros((T, J), dtype=np.float32)

        for i, frame in enumerate(sequence.frames):
            if person_id in frame.persons:
                kpts = frame.persons[person_id]  # (J, 3): [x, y, conf]
                if kpts.shape[-1] >= 3:
                    joint_coords[i] = kpts[:J, :2]
                    joint_conf[i] = kpts[:J, 2]
                else:
                    joint_coords[i] = kpts[:J, :2]
                    joint_conf[i] = 1.0

        # 收集时间戳
        ts_arr = np.array([f.timestamp for f in sequence.frames], dtype=np.float64)

        return PoseTimeSeries(
            source="vitpose",
            video_path=sequence.video_path,
            fps=sequence.fps,
            timestamps=ts_arr,
            joint_coords=joint_coords,
            joint_coords_conf=joint_conf,
            joint_names=list(sequence.joint_names),
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 序列化 / 反序列化
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def save(
        sequence: ViTPoseSequence,
        output_path: str,
        format: str = "pkl",
    ) -> str:
        """保存 ViTPoseSequence 到文件。

        Args:
            sequence:    ViTPoseSequence 对象
            output_path: 输出文件路径（.pkl 或 .npz）
            format:      "pkl" 或 "npz"

        Returns:
            实际写入的文件路径
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        if format == "pkl":
            with open(output_path, "wb") as f:
                pickle.dump(sequence, f, protocol=pickle.HIGHEST_PROTOCOL)

        elif format == "npz":
            # 将 ViTPoseSequence 序列化为 npz（所有人物合并为单人模式）
            # 假设只保存 person_id=0
            T = len(sequence.frames)
            J = len(sequence.joint_names) if sequence.joint_names else 17
            coords = np.full((T, J, 3), np.nan, dtype=np.float32)
            timestamps = np.zeros(T, dtype=np.float64)
            for i, frame in enumerate(sequence.frames):
                timestamps[i] = frame.timestamp
                if 0 in frame.persons:
                    kpts = frame.persons[0]
                    coords[i, :min(kpts.shape[0], J)] = kpts[:J]
            np.savez_compressed(
                output_path,
                timestamps=timestamps,
                keypoints=coords,       # (T, J, 3): [x, y, conf]
                fps=sequence.fps,
                total_frames=sequence.total_frames,
                joint_names=np.array(sequence.joint_names or []),
            )
        else:
            raise ValueError(f"不支持的格式：{format}，请选择 pkl 或 npz")

        print(f"[ViTPoseVideoProcessor] 结果已保存：{output_path}")
        return output_path

    @staticmethod
    def load(input_path: str) -> ViTPoseSequence:
        """从文件加载 ViTPoseSequence。

        Args:
            input_path: .pkl 文件路径

        Returns:
            ViTPoseSequence
        """
        with open(input_path, "rb") as f:
            return pickle.load(f)
