"""
pose_extraction

人体姿态提取模块，为 BeatsMatching 提供两条关节数据管线：

管线 A — ViTPose（2D 关节坐标）：
    通过 ViTPose 逐帧检测视频中的人体关节点，
    输出 J 个关节的 (x, y) 坐标时序序列，
    作为节拍动作帧检测的 2D 特征信号。

管线 B — HMR2 / 4D-Humans（3D SMPL 旋转量）：
    通过 HMR2 逐帧重建视频中的三维人体，
    提取 SMPL 模型的 24 个关节旋转量 θ_j(t)（轴角形式），
    作为节拍动作帧检测的 3D 旋转特征信号（对应论文式 1、式 2）。

管线 C — 联合管线：
    同时运行 A + B，融合 2D+3D 特征进行节拍检测（论文改进方法）。

公开接口：
    extract_pose(video_path, mode, ...)  → PoseTimeSeries
    extract_and_detect(video_path, ...) → PoseTimeSeries（含节拍帧）
"""

from .pipeline import extract_pose, extract_and_detect, PoseExtractionConfig
from .core.data_types import (
    ViTPoseFrameResult,
    ViTPoseSequence,
    HMR2FrameResult,
    HMR2Sequence,
    PoseTimeSeries,
    SMPL_JOINT_NAMES,
    SMPL_PARENT_IDX,
)
from .core.beat_detector import detect_beats
from .core.smoothing import (
    smooth_sequence,
    smooth_coords,
    compute_rotation_gradient_vectorized,
)

__all__ = [
    # 顶层接口
    "extract_pose",
    "extract_and_detect",
    "PoseExtractionConfig",
    # 数据类型
    "ViTPoseFrameResult",
    "ViTPoseSequence",
    "HMR2FrameResult",
    "HMR2Sequence",
    "PoseTimeSeries",
    "SMPL_JOINT_NAMES",
    "SMPL_PARENT_IDX",
    # 算法工具
    "detect_beats",
    "smooth_sequence",
    "smooth_coords",
    "compute_rotation_gradient_vectorized",
]
