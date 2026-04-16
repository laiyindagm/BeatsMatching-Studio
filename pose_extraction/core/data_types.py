"""
pose_extraction/core/data_types.py

统一的数据类型定义，用于描述两种模型的输出格式，
以及对接 BeatsMatching 的节拍检测接口所需的时序序列格式。
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# 基础类型别名
# ──────────────────────────────────────────────────────────────────────────────

# 单帧关节点坐标数组，形状 (J, 2) 或 (J, 3)，列序：[x, y] 或 [x, y, conf]
JointCoords2D = np.ndarray  # shape (J, 2) or (J, 3)

# 单帧 SMPL 关节旋转量（轴角形式），形状 (J, 3)；24 个 SMPL 关节
JointAxisAngle = np.ndarray  # shape (J, 3)

# 单帧 SMPL 关节旋转量（旋转矩阵），形状 (J, 3, 3)
JointRotMat = np.ndarray     # shape (J, 3, 3)


# ──────────────────────────────────────────────────────────────────────────────
# ViTPose 输出数据类
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ViTPoseFrameResult:
    """单帧的 ViTPose 检测结果。

    Attributes:
        frame_idx:  视频帧编号（0-based）
        timestamp:  该帧对应的视频时间戳（秒）
        persons:    字典，key 为 person_id（跟踪 ID），
                    value 为形状 (J, 3) 的数组 [x, y, confidence]
        num_joints: 关节数（COCO=17，Wholebody=133 等）
    """
    frame_idx: int
    timestamp: float
    persons: Dict[int, JointCoords2D]     # person_id -> (J, 3) array
    num_joints: int = 17


@dataclass
class ViTPoseSequence:
    """整段视频的 ViTPose 时序结果。

    Attributes:
        video_path:     处理的视频路径
        fps:            视频帧率
        total_frames:   视频总帧数
        frames:         按帧索引排序的帧结果列表
        joint_names:    关节名称列表，与 persons 数组列索引对应
        skeleton:       骨架连接关系，用于可视化，格式 [(src_idx, dst_idx), ...]
    """
    video_path: str
    fps: float
    total_frames: int
    frames: List[ViTPoseFrameResult] = field(default_factory=list)
    joint_names: List[str] = field(default_factory=list)
    skeleton: List[Tuple[int, int]] = field(default_factory=list)

    def get_joint_trajectory(
        self,
        joint_idx: int,
        person_id: int = 0,
        conf_threshold: float = 0.3,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """获取指定关节、指定人物的 (x, y) 时序轨迹。

        Args:
            joint_idx:       关节索引
            person_id:       人物跟踪 ID
            conf_threshold:  置信度阈值，低于此值用 NaN 填充

        Returns:
            timestamps:  形状 (T,) 的时间戳数组（秒）
            coords:      形状 (T, 2) 的坐标数组，不可信帧为 NaN
        """
        timestamps = []
        coords = []
        for frame in self.frames:
            timestamps.append(frame.timestamp)
            if person_id in frame.persons:
                kpt = frame.persons[person_id][joint_idx]  # (3,): x, y, conf
                if kpt[2] >= conf_threshold:
                    coords.append(kpt[:2])
                else:
                    coords.append(np.array([np.nan, np.nan]))
            else:
                coords.append(np.array([np.nan, np.nan]))
        return np.array(timestamps), np.array(coords)

    def get_all_joints_trajectory(
        self,
        person_id: int = 0,
        conf_threshold: float = 0.3,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """获取所有关节的时序轨迹。

        Returns:
            timestamps:  形状 (T,) 的时间戳数组
            coords:      形状 (T, J, 2) 的坐标数组
        """
        timestamps = []
        all_coords = []
        for frame in self.frames:
            timestamps.append(frame.timestamp)
            if person_id in frame.persons:
                kpts = frame.persons[person_id]  # (J, 3)
                xy = kpts[:, :2].copy()
                conf = kpts[:, 2]
                xy[conf < conf_threshold] = np.nan
                all_coords.append(xy)
            else:
                J = len(self.joint_names) if self.joint_names else 17
                all_coords.append(np.full((J, 2), np.nan))
        return np.array(timestamps), np.array(all_coords)


# ──────────────────────────────────────────────────────────────────────────────
# HMR2 / 4D-Humans SMPL 输出数据类
# ──────────────────────────────────────────────────────────────────────────────

# SMPL 24 个关节的标准名称（官方顺序）
SMPL_JOINT_NAMES = [
    "pelvis",           # 0  根节点
    "left_hip",         # 1
    "right_hip",        # 2
    "spine1",           # 3
    "left_knee",        # 4
    "right_knee",       # 5
    "spine2",           # 6
    "left_ankle",       # 7
    "right_ankle",      # 8
    "spine3",           # 9
    "left_foot",        # 10
    "right_foot",       # 11
    "neck",             # 12
    "left_collar",      # 13
    "right_collar",     # 14
    "head",             # 15
    "left_shoulder",    # 16
    "right_shoulder",   # 17
    "left_elbow",       # 18
    "right_elbow",      # 19
    "left_wrist",       # 20
    "right_wrist",      # 21
    "left_hand",        # 22
    "right_hand",       # 23
]

# SMPL 父节点索引（-1 表示根节点）
SMPL_PARENT_IDX = [
    -1,  # 0 pelvis (root)
    0,   # 1 left_hip
    0,   # 2 right_hip
    0,   # 3 spine1
    1,   # 4 left_knee
    2,   # 5 right_knee
    3,   # 6 spine2
    4,   # 7 left_ankle
    5,   # 8 right_ankle
    6,   # 9 spine3
    7,   # 10 left_foot
    8,   # 11 right_foot
    9,   # 12 neck
    9,   # 13 left_collar
    9,   # 14 right_collar
    12,  # 15 head
    13,  # 16 left_shoulder
    14,  # 17 right_shoulder
    16,  # 18 left_elbow
    17,  # 19 right_elbow
    18,  # 20 left_wrist
    19,  # 21 right_wrist
    20,  # 22 left_hand
    21,  # 23 right_hand
]


@dataclass
class HMR2FrameResult:
    """单帧的 HMR2 重建结果。

    Attributes:
        frame_idx:        视频帧编号（0-based）
        timestamp:        该帧对应的时间戳（秒）
        person_id:        人物跟踪 ID
        global_orient:    全局方向（轴角），形状 (1, 3)
        body_pose:        身体关节旋转（轴角），形状 (23, 3)，不含根节点
        betas:            形体系数，形状 (10,)
        pred_cam:         相机参数 [s, tx, ty]，形状 (3,)
        pred_cam_t_full:  相机平移（像素坐标系下），形状 (3,)
        joints_3d:        3D 关节位置（相机坐标系），形状 (J, 3)，J=44
        joints_2d:        2D 投影关节点（图像坐标系），形状 (J, 2)
        vertices:         SMPL 网格顶点（可选），形状 (6890, 3)
    """
    frame_idx: int
    timestamp: float
    person_id: int = 0
    global_orient: Optional[JointAxisAngle] = None    # (1, 3)
    body_pose: Optional[JointAxisAngle] = None         # (23, 3)
    betas: Optional[np.ndarray] = None                  # (10,)
    pred_cam: Optional[np.ndarray] = None               # (3,)
    pred_cam_t_full: Optional[np.ndarray] = None        # (3,)
    joints_3d: Optional[np.ndarray] = None              # (J, 3)
    joints_2d: Optional[np.ndarray] = None              # (J, 2)
    vertices: Optional[np.ndarray] = None               # (6890, 3)，内存较大，默认 None

    @property
    def full_pose(self) -> Optional[np.ndarray]:
        """返回完整的 24 关节轴角旋转量，形状 (24, 3)。
        索引 0 为 global_orient，索引 1-23 为 body_pose。
        """
        if self.global_orient is None or self.body_pose is None:
            return None
        go = self.global_orient.reshape(1, 3)
        bp = self.body_pose.reshape(23, 3)
        return np.concatenate([go, bp], axis=0)  # (24, 3)

    def get_joint_rotation(self, joint_idx: int) -> Optional[np.ndarray]:
        """获取指定关节的旋转轴角量，形状 (3,)。"""
        pose = self.full_pose
        if pose is None:
            return None
        return pose[joint_idx]  # (3,)


@dataclass
class HMR2Sequence:
    """整段视频的 HMR2 时序重建结果。

    Attributes:
        video_path:    处理的视频路径
        fps:           视频帧率
        total_frames:  视频总帧数
        frames:        按帧索引排序的帧结果列表
        person_ids:    出现的所有人物跟踪 ID
    """
    video_path: str
    fps: float
    total_frames: int
    frames: List[HMR2FrameResult] = field(default_factory=list)
    person_ids: List[int] = field(default_factory=list)

    def get_joint_rotation_sequence(
        self,
        joint_idx: int,
        person_id: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """获取关节 j 的旋转量时序序列 θ_j(t)。

        对应论文公式：θ_j(t) 为第 t 帧关节 j 的轴角旋转量。

        Args:
            joint_idx:  关节索引（0=pelvis/global_orient, 1-23=body joints）
            person_id:  人物跟踪 ID

        Returns:
            timestamps:   形状 (T,) 的时间戳数组（秒）
            rotations:    形状 (T, 3) 的轴角旋转量数组，缺失帧为 NaN
        """
        timestamps = []
        rotations = []
        for frame in self.frames:
            if frame.person_id != person_id:
                continue
            timestamps.append(frame.timestamp)
            rot = frame.get_joint_rotation(joint_idx)
            if rot is not None:
                rotations.append(rot)
            else:
                rotations.append(np.full(3, np.nan))
        return np.array(timestamps), np.array(rotations)

    def get_all_joints_rotation_sequence(
        self,
        person_id: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """获取所有 24 个关节的旋转量时序序列。

        Returns:
            timestamps:  形状 (T,) 的时间戳数组
            rotations:   形状 (T, 24, 3) 的轴角旋转量数组
        """
        timestamps = []
        rotations = []
        for frame in self.frames:
            if frame.person_id != person_id:
                continue
            timestamps.append(frame.timestamp)
            pose = frame.full_pose
            if pose is not None:
                rotations.append(pose)
            else:
                rotations.append(np.full((24, 3), np.nan))
        return np.array(timestamps), np.array(rotations)


# ──────────────────────────────────────────────────────────────────────────────
# 节拍检测接口所需的统一输出格式（对接 BeatsMatching）
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PoseTimeSeries:
    """统一的姿态时序数据，对接 BeatsMatching 算法接口。

    可由 ViTPoseSequence 或 HMR2Sequence 转化而来。

    Attributes:
        source:       数据来源标识，"vitpose" 或 "hmr2"
        video_path:   源视频路径
        fps:          视频帧率
        timestamps:   帧时间戳数组，形状 (T,)
        # ViTPose 字段（2D 关节坐标）
        joint_coords:        形状 (T, J, 2)，ViTPose 输出时填充
        joint_coords_conf:   形状 (T, J)，ViTPose 置信度
        joint_names:         长度 J 的关节名称列表
        # HMR2 字段（3D 关节旋转）
        joint_rotations:     形状 (T, 24, 3)，HMR2 轴角旋转量 θ_j(t)
        # 后处理结果（由平滑模块填充）
        smoothed_rotations:  形状 (T, 24, 3)，平滑后的 θ_j(t) hat
        rotation_gradient:   形状 (T, 24)，旋转梯度模长 m_j(t)，见论文式(2)
        beat_frames:         节拍动作帧索引列表
        beat_timestamps:     节拍动作帧时间戳列表（秒）
    """
    source: str                                            # "vitpose" | "hmr2"
    video_path: str
    fps: float
    timestamps: np.ndarray                                 # (T,)
    # ViTPose
    joint_coords: Optional[np.ndarray] = None              # (T, J, 2)
    joint_coords_conf: Optional[np.ndarray] = None         # (T, J)
    joint_names: Optional[List[str]] = None
    # HMR2
    joint_rotations: Optional[np.ndarray] = None           # (T, 24, 3)
    # 后处理
    smoothed_rotations: Optional[np.ndarray] = None        # (T, 24, 3)
    rotation_gradient: Optional[np.ndarray] = None         # (T, 24)
    beat_frames: Optional[List[int]] = None
    beat_timestamps: Optional[List[float]] = None
