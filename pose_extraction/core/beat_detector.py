"""
pose_extraction/core/beat_detector.py

基于关节时序信息的节拍动作帧检测算法。

支持两种策略：
1. ViTPose 策略：利用关节累计位移的峰值检测节拍动作帧
2. HMR2 策略：利用关节旋转梯度 m_j(t) 的零点（低谷）检测节拍动作帧
3. 联合策略：综合坐标-旋转量联合分析（论文提出的改进方法）

所有策略均利用父子关节加权。
"""

from __future__ import annotations
from typing import List, Optional, Tuple

import numpy as np
from scipy.signal import find_peaks, peak_prominences

from .data_types import (
    PoseTimeSeries,
    SMPL_JOINT_NAMES,
    SMPL_PARENT_IDX,
)
from .smoothing import (
    smooth_sequence_fast,
    compute_rotation_gradient_vectorized,
    compute_cumulative_displacement,
    smooth_coords,
)


# ──────────────────────────────────────────────────────────────────────────────
# 关节权重：优先关注四肢末端与躯干节点
# ──────────────────────────────────────────────────────────────────────────────

# SMPL 关节重要性权重（手动设定，可调整）
SMPL_JOINT_WEIGHTS = np.array([
    0.1,   # 0  pelvis         —— 根节点
    0.3,   # 1  left_hip
    0.3,   # 2  right_hip
    0.2,   # 3  spine1
    0.4,   # 4  left_knee
    0.4,   # 5  right_knee
    0.2,   # 6  spine2
    0.8,   # 7  left_ankle     —— 足踝（重要）
    0.8,   # 8  right_ankle
    0.2,   # 9  spine3
    0.5,   # 10 left_foot
    0.5,   # 11 right_foot
    0.3,   # 12 neck
    0.3,   # 13 left_collar
    0.3,   # 14 right_collar
    0.3,   # 15 head
    0.6,   # 16 left_shoulder  —— 肩膀（重要）
    0.6,   # 17 right_shoulder
    0.9,   # 18 left_elbow     —— 肘关节（非常重要）
    0.9,   # 19 right_elbow
    1.0,   # 20 left_wrist     —— 腕关节（最重要）
    1.0,   # 21 right_wrist
    0.7,   # 22 left_hand
    0.7,   # 23 right_hand
], dtype=float)


# ──────────────────────────────────────────────────────────────────────────────
# 策略 1：基于 ViTPose 累计位移峰值检测
# ──────────────────────────────────────────────────────────────────────────────

def detect_beats_from_displacement(
    pose_ts: PoseTimeSeries,
    person_id: int = 0,
    smooth_K: int = 3,
    peak_distance_sec: float = 0.2,
    peak_prominence_ratio: float = 0.1,
    conf_threshold: float = 0.3,
) -> List[int]:
    """基于关节累计位移峰值检测节拍动作帧（ViTPose 策略）。

    步骤：
    1. 计算所有关节（置信度 > conf_threshold）的逐帧位移增量
    2. 对坐标进行高斯平滑（式1）
    3. 计算各关节累计位移，COCO 关节按权重加权求和
    4. 峰值检测 → 节拍动作帧

    Args:
        pose_ts:              包含 joint_coords (T, J, 2) 的 PoseTimeSeries
        person_id:            目标人物 ID（ViTPose 为 key，这里忽略，coords 已是单人）
        smooth_K:             平滑窗口半径
        peak_distance_sec:    两个峰值之间的最小时间间隔（秒）
        peak_prominence_ratio: 峰值显著性阈值（占全局峰谷差的比例）
        conf_threshold:       置信度阈值，低于此值的关节不参与计算

    Returns:
        beat_frames: 节拍动作帧索引列表
    """
    assert pose_ts.joint_coords is not None, "joint_coords 不能为空"

    coords = pose_ts.joint_coords.copy()     # (T, J, 2)
    conf = pose_ts.joint_coords_conf          # (T, J) or None

    # 置信度过滤
    if conf is not None:
        mask = conf < conf_threshold         # (T, J)
        coords[mask] = np.nan

    # 坐标平滑（Savitzky-Golay）
    coords_smooth = smooth_coords(coords, method="savgol", window=7, polyorder=2)

    # 计算累计位移距离
    cum_disp = compute_cumulative_displacement(coords_smooth)   # (T, J)

    # COCO 关节权重（若无对应 SMPL 权重，均匀使用）
    J = coords.shape[1]
    if J == 17:
        # COCO-17 权重：手腕、脚踝、膝盖最重要
        coco_weights = np.array([
            0.1,  # 0 nose
            0.1,  # 1 left_eye
            0.1,  # 2 right_eye
            0.1,  # 3 left_ear
            0.1,  # 4 right_ear
            0.5,  # 5 left_shoulder
            0.5,  # 6 right_shoulder
            0.8,  # 7 left_elbow
            0.8,  # 8 right_elbow
            1.0,  # 9 left_wrist
            1.0,  # 10 right_wrist
            0.5,  # 11 left_hip
            0.5,  # 12 right_hip
            0.8,  # 13 left_knee
            0.8,  # 14 right_knee
            1.0,  # 15 left_ankle
            1.0,  # 16 right_ankle
        ], dtype=float)
    else:
        coco_weights = np.ones(J, dtype=float)

    # 加权合并各关节累计位移
    w = coco_weights / coco_weights.sum()
    # NaN 替换为 0 后加权
    cum_filled = np.nan_to_num(cum_disp, nan=0.0)
    combined = (cum_filled * w[np.newaxis, :]).sum(axis=1)   # (T,)

    # 逐帧速度（累计位移的一阶差分）
    velocity = np.diff(combined, prepend=combined[0])         # (T,)

    # 平滑速度
    velocity_smooth = np.convolve(
        velocity,
        np.ones(max(1, smooth_K * 2 + 1)) / (smooth_K * 2 + 1),
        mode="same",
    )

    # 峰值检测
    fps = pose_ts.fps
    min_dist = max(1, int(peak_distance_sec * fps))
    prom_threshold = (velocity_smooth.max() - velocity_smooth.min()) * peak_prominence_ratio

    peaks, props = find_peaks(
        velocity_smooth,
        distance=min_dist,
        prominence=prom_threshold,
    )

    return sorted(peaks.tolist())


# ──────────────────────────────────────────────────────────────────────────────
# 策略 2：基于 HMR2 旋转梯度零点检测
# ──────────────────────────────────────────────────────────────────────────────

def detect_beats_from_rotation(
    pose_ts: PoseTimeSeries,
    smooth_K: int = 5,
    gradient_threshold_ratio: float = 0.05,
    min_beat_distance_sec: float = 0.2,
    joint_weights: Optional[np.ndarray] = None,
) -> List[int]:
    """基于关节旋转梯度零点检测节拍动作帧（HMR2 策略）。

    步骤：
    1. 平滑旋转量序列 θ_j(t)_hat（式1）
    2. 计算旋转梯度 m_j(t)（式2）
    3. 各关节梯度加权求和得到综合曲线
    4. 梯度曲线的局部极小值（低于阈值）= 动作方向变化时刻 = 节拍帧

    Args:
        pose_ts:                含 joint_rotations (T, 24, 3) 的 PoseTimeSeries
        smooth_K:               平滑窗口半径 K
        gradient_threshold_ratio: 梯度阈值（占最大梯度的比例），低于此为零点
        min_beat_distance_sec:  相邻节拍最小间隔（秒）
        joint_weights:          24 个关节权重，默认使用 SMPL_JOINT_WEIGHTS

    Returns:
        beat_frames: 节拍动作帧索引列表
    """
    assert pose_ts.joint_rotations is not None, "joint_rotations 不能为空"

    rotations = pose_ts.joint_rotations.copy()   # (T, 24, 3)

    # 平滑（式1）
    rot_smooth = smooth_sequence_fast(
        rotations, K=smooth_K, mode="gaussian"
    )
    # 保存到 pose_ts
    pose_ts.smoothed_rotations = rot_smooth

    # 旋转梯度（式2）
    grad = compute_rotation_gradient_vectorized(rot_smooth)   # (T, 24)
    pose_ts.rotation_gradient = grad

    # 关节权重
    if joint_weights is None:
        joint_weights = SMPL_JOINT_WEIGHTS

    w = joint_weights / joint_weights.sum()
    combined_grad = (grad * w[np.newaxis, :]).sum(axis=1)   # (T,)

    # 再次平滑综合曲线
    combined_smooth = np.convolve(
        combined_grad,
        np.ones(3) / 3,
        mode="same",
    )

    # 检测极小值（对应梯度的零点/低谷）
    fps = pose_ts.fps
    min_dist = max(1, int(min_beat_distance_sec * fps))
    threshold = combined_smooth.max() * gradient_threshold_ratio

    # 找低谷（负号后找峰值）
    neg_curve = -combined_smooth
    valleys, _ = find_peaks(neg_curve, distance=min_dist)

    # 过滤：仅保留低于阈值的低谷
    beat_frames = [int(v) for v in valleys if combined_smooth[v] < threshold]

    return sorted(beat_frames)


# ──────────────────────────────────────────────────────────────────────────────
# 策略 3：坐标-旋转量联合分析
# ──────────────────────────────────────────────────────────────────────────────

def detect_beats_joint(
    pose_ts: PoseTimeSeries,
    weight_coord: float = 0.5,
    weight_rot: float = 0.5,
    smooth_K: int = 4,
    peak_distance_sec: float = 0.2,
    peak_prominence_ratio: float = 0.08,
) -> List[int]:
    """坐标-旋转量联合分析检测节拍动作帧（论文提出的改进方法）。

    将 ViTPose 的位移信号与 HMR2 的旋转梯度信号加权融合，
    同时利用 2D 图像坐标和 3D 广义坐标进行节拍判断。

    Args:
        pose_ts:               同时含 joint_coords 和 joint_rotations 的 PoseTimeSeries
        weight_coord:          2D 坐标信号权重
        weight_rot:            3D 旋转信号权重
        smooth_K:              平滑半窗口
        peak_distance_sec:     最小节拍间隔（秒）
        peak_prominence_ratio: 峰值显著性阈值

    Returns:
        beat_frames: 节拍动作帧索引列表
    """
    T = len(pose_ts.timestamps)
    fps = pose_ts.fps
    combined = np.zeros(T)

    # —— 坐标信号 ——
    if pose_ts.joint_coords is not None:
        coords = pose_ts.joint_coords.copy()
        if pose_ts.joint_coords_conf is not None:
            mask = pose_ts.joint_coords_conf < 0.3
            coords[mask] = np.nan
        coords_smooth = smooth_coords(coords, method="savgol")
        cum_disp = compute_cumulative_displacement(coords_smooth)
        velocity = np.diff(np.nan_to_num(cum_disp).mean(axis=1), prepend=0)
        velocity = np.convolve(velocity, np.ones(5) / 5, mode="same")
        # 归一化
        v_range = velocity.max() - velocity.min()
        if v_range > 1e-6:
            velocity_norm = (velocity - velocity.min()) / v_range
        else:
            velocity_norm = velocity
        combined += weight_coord * velocity_norm

    # —— 旋转信号 ——
    if pose_ts.joint_rotations is not None:
        rot = pose_ts.joint_rotations.copy()
        rot_smooth = smooth_sequence_fast(rot, K=smooth_K)
        grad = compute_rotation_gradient_vectorized(rot_smooth)
        w = SMPL_JOINT_WEIGHTS / SMPL_JOINT_WEIGHTS.sum()
        grad_combined = (grad * w[np.newaxis, :]).sum(axis=1)
        grad_combined = np.convolve(grad_combined, np.ones(3) / 3, mode="same")
        # 归一化
        g_range = grad_combined.max() - grad_combined.min()
        if g_range > 1e-6:
            grad_norm = (grad_combined - grad_combined.min()) / g_range
        else:
            grad_norm = grad_combined
        # 旋转梯度低谷 → 将信号反转后作为正向特征
        combined += weight_rot * (1.0 - grad_norm)

    # 峰值检测
    min_dist = max(1, int(peak_distance_sec * fps))
    prom = (combined.max() - combined.min()) * peak_prominence_ratio
    peaks, _ = find_peaks(combined, distance=min_dist, prominence=prom)

    return sorted(peaks.tolist())


# ──────────────────────────────────────────────────────────────────────────────
# 顶层接口
# ──────────────────────────────────────────────────────────────────────────────

def detect_beats(
    pose_ts: PoseTimeSeries,
    strategy: str = "auto",
    **kwargs,
) -> PoseTimeSeries:
    """检测节拍动作帧，返回填充了 beat_frames 和 beat_timestamps 的 PoseTimeSeries。

    Args:
        pose_ts:   输入的姿态时序数据
        strategy:  检测策略：
                   "vitpose"  —— 仅用关节坐标（需要 joint_coords）
                   "hmr2"     —— 仅用旋转量（需要 joint_rotations）
                   "joint"    —— 坐标+旋转联合（需要两者）
                   "auto"     —— 自动选择：若有两种则用 "joint"，否则退化

    Returns:
        更新了 beat_frames 与 beat_timestamps 的 pose_ts（原地修改后返回）
    """
    if strategy == "auto":
        has_coords = pose_ts.joint_coords is not None
        has_rot = pose_ts.joint_rotations is not None
        if has_coords and has_rot:
            strategy = "joint"
        elif has_coords:
            strategy = "vitpose"
        elif has_rot:
            strategy = "hmr2"
        else:
            raise ValueError("pose_ts 中既没有 joint_coords 也没有 joint_rotations")

    if strategy == "vitpose":
        beat_frames = detect_beats_from_displacement(pose_ts, **kwargs)
    elif strategy == "hmr2":
        beat_frames = detect_beats_from_rotation(pose_ts, **kwargs)
    elif strategy == "joint":
        beat_frames = detect_beats_joint(pose_ts, **kwargs)
    else:
        raise ValueError(f"未知策略: {strategy}")

    pose_ts.beat_frames = beat_frames
    pose_ts.beat_timestamps = [float(pose_ts.timestamps[i]) for i in beat_frames]
    return pose_ts
