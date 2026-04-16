"""
pose_extraction/hmr2

4D-Humans / HMR2 三维人体重建模块。
从视频中逐帧提取 SMPL 参数（global_orient, body_pose, betas），
并计算每个关节 j 的旋转量时序序列 θ_j(t)。
"""

from .reconstructor import HMR2Reconstructor
from .video_processor import HMR2VideoProcessor

__all__ = ["HMR2Reconstructor", "HMR2VideoProcessor"]
