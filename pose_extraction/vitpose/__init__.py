"""
pose_extraction/vitpose

ViTPose 2D 关节检测模块。
基于 HuggingFace transformers 的 ViTPoseForPoseEstimation 接口，
支持通过 YOLOv8 / ViTDet 进行人体框检测，再逐帧提取关节坐标。
"""

from .detector import ViTPoseDetector
from .video_processor import ViTPoseVideoProcessor

__all__ = ["ViTPoseDetector", "ViTPoseVideoProcessor"]
