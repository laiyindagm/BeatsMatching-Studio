"""
pose_extraction/core/smoothing.py

实现论文中的关节旋转量平滑处理与梯度计算：

  θ_j(t)_hat = Σ_{k=-K}^{K} w_k · θ_j(t+k),   Σw_k = 1  ...(1)
  m_j(t)     = ||θ_j(t)_hat - θ_j(t-1)_hat||²            ...(2)

同时提供 ViTPose 关节坐标的平滑与位移梯度计算。
"""

from __future__ import annotations
from typing import Optional, Tuple, Union

import numpy as np
from scipy.signal import savgol_filter


# ──────────────────────────────────────────────────────────────────────────────
# 通用加权平均平滑
# ──────────────────────────────────────────────────────────────────────────────

def gaussian_weights(K: int, sigma: float = 1.0) -> np.ndarray:
    """生成以 0 为中心的高斯核权重，长度 2K+1，归一化。"""
    t = np.arange(-K, K + 1, dtype=float)
    w = np.exp(-t ** 2 / (2 * sigma ** 2))
    return w / w.sum()


def uniform_weights(K: int) -> np.ndarray:
    """生成均匀权重，长度 2K+1。"""
    n = 2 * K + 1
    return np.ones(n) / n


def smooth_sequence(
    seq: np.ndarray,
    K: int = 3,
    weights: Optional[np.ndarray] = None,
    mode: str = "gaussian",
) -> np.ndarray:
    """对时序数据进行加权滑动平均平滑，对应论文式(1)。

    Args:
        seq:     输入序列，任意形状，第 0 维为时间轴，形状 (T, ...)
        K:       半窗口大小，窗口为 [-K, K]，共 2K+1 个点
        weights: 自定义权重，长度须为 2K+1，若为 None 则按 mode 自动生成
        mode:    "gaussian" 或 "uniform"

    Returns:
        smoothed: 与 seq 形状相同的平滑结果
    """
    if weights is None:
        if mode == "gaussian":
            weights = gaussian_weights(K)
        else:
            weights = uniform_weights(K)

    T = seq.shape[0]
    orig_shape = seq.shape[1:]
    seq_flat = seq.reshape(T, -1)   # (T, D)
    out_flat = np.zeros_like(seq_flat)

    for t in range(T):
        wsum = 0.0
        val = np.zeros(seq_flat.shape[1])
        for ki, k in enumerate(range(-K, K + 1)):
            idx = t + k
            if 0 <= idx < T and not np.any(np.isnan(seq_flat[idx])):
                val += weights[ki] * seq_flat[idx]
                wsum += weights[ki]
        if wsum > 0:
            out_flat[t] = val / wsum
        else:
            out_flat[t] = seq_flat[t]

    return out_flat.reshape(seq.shape)


def smooth_sequence_fast(
    seq: np.ndarray,
    K: int = 3,
    weights: Optional[np.ndarray] = None,
    mode: str = "gaussian",
) -> np.ndarray:
    """利用 np.convolve 批量加速平滑（对 NaN 不做特殊处理，速度更快）。

    适用于完整无缺失的序列。
    """
    if weights is None:
        weights = gaussian_weights(K, sigma=K / 2.0) if mode == "gaussian" else uniform_weights(K)

    T = seq.shape[0]
    orig_shape = seq.shape[1:]
    seq_flat = seq.reshape(T, -1)   # (T, D)
    D = seq_flat.shape[1]
    out = np.zeros_like(seq_flat)

    for d in range(D):
        out[:, d] = np.convolve(seq_flat[:, d], weights, mode="same")

    return out.reshape(seq.shape)


# ──────────────────────────────────────────────────────────────────────────────
# HMR2 旋转梯度（论文式2）
# ──────────────────────────────────────────────────────────────────────────────

def compute_rotation_gradient(
    rotations: np.ndarray,
) -> np.ndarray:
    """计算关节旋转量梯度的模长 m_j(t)，对应论文式(2)。

    m_j(t) = ||θ_j(t) - θ_j(t-1)||²

    Args:
        rotations: 形状 (T, J, 3) 的轴角旋转量序列

    Returns:
        gradient: 形状 (T, J) 的梯度模长序列，t=0 时为 0
    """
    T, J = rotations.shape[:2]
    gradient = np.zeros((T, J), dtype=float)
    for t in range(1, T):
        diff = rotations[t] - rotations[t - 1]   # (J, 3)
        gradient[t] = np.sum(diff ** 2, axis=-1)  # (J,)
    return gradient


def compute_rotation_gradient_vectorized(
    rotations: np.ndarray,
) -> np.ndarray:
    """向量化版本，更快。"""
    # rotations: (T, J, 3)
    diff = np.diff(rotations, axis=0)           # (T-1, J, 3)
    mag = np.sum(diff ** 2, axis=-1)            # (T-1, J)
    gradient = np.zeros((rotations.shape[0], rotations.shape[1]))
    gradient[1:] = mag
    return gradient


# ──────────────────────────────────────────────────────────────────────────────
# ViTPose 坐标位移计算
# ──────────────────────────────────────────────────────────────────────────────

def compute_displacement(
    coords: np.ndarray,
    ref_frame: int = 0,
) -> np.ndarray:
    """计算每帧关节相对于参考帧的矢量位移量。

    对应论文：计算每一节点相较于上一个节拍动作帧（初始帧）的矢量位移变化。

    Args:
        coords:    形状 (T, J, 2) 的关节坐标序列
        ref_frame: 参考帧索引

    Returns:
        displacement: 形状 (T, J, 2) 的位移数组
    """
    return coords - coords[ref_frame : ref_frame + 1]   # 广播


def compute_cumulative_displacement(
    coords: np.ndarray,
) -> np.ndarray:
    """计算每个关节相对于上一帧的累计位移距离（模长）。

    Args:
        coords: 形状 (T, J, 2) 的坐标序列

    Returns:
        cum_dist: 形状 (T, J) 的累计位移距离序列
    """
    T, J = coords.shape[:2]
    cum_dist = np.zeros((T, J))
    for t in range(1, T):
        delta = coords[t] - coords[t - 1]               # (J, 2)
        step = np.linalg.norm(delta, axis=-1)            # (J,)
        # 处理 NaN
        nan_mask = np.any(np.isnan(delta), axis=-1)
        step[nan_mask] = 0.0
        cum_dist[t] = cum_dist[t - 1] + step
    return cum_dist


def compute_velocity(
    coords: np.ndarray,
    fps: float,
) -> np.ndarray:
    """计算关节速度（帧间差分除以时间间隔）。

    Returns:
        velocity: 形状 (T, J, 2) 的速度数组，t=0 时为 0
    """
    diff = np.zeros_like(coords)
    diff[1:] = (coords[1:] - coords[:-1]) * fps
    return diff


def smooth_coords(
    coords: np.ndarray,
    method: str = "savgol",
    window: int = 7,
    polyorder: int = 2,
    K: int = 3,
) -> np.ndarray:
    """对关节坐标进行平滑，忽略 NaN 区域。

    Args:
        coords:    形状 (T, J, 2) 的坐标
        method:    "savgol" 使用 Savitzky-Golay，"gaussian" 使用高斯滑动平均
        window:    savgol 窗口大小（须为奇数）
        polyorder: savgol 多项式阶数
        K:         gaussian 平滑半窗口大小

    Returns:
        smoothed: 形状 (T, J, 2) 的平滑坐标
    """
    T, J, _ = coords.shape
    out = coords.copy()
    if method == "savgol":
        for j in range(J):
            for d in range(2):
                col = coords[:, j, d]
                nan_mask = np.isnan(col)
                if nan_mask.all():
                    continue
                # 插值填充 NaN
                x_valid = np.where(~nan_mask)[0]
                y_valid = col[~nan_mask]
                col_interp = np.interp(np.arange(T), x_valid, y_valid)
                try:
                    col_smooth = savgol_filter(col_interp, window_length=window, polyorder=polyorder)
                except ValueError:
                    col_smooth = col_interp
                # 还原 NaN
                col_smooth[nan_mask] = np.nan
                out[:, j, d] = col_smooth
    else:
        out = smooth_sequence(coords, K=K, mode="gaussian")
    return out
