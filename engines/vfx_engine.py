import cv2
import numpy as np
import random
from core.structures import EffectType


class VFXEngine:
    @staticmethod
    def apply_effect(frame: np.ndarray, effect_type: EffectType, progress: float, params: dict) -> np.ndarray:
        """
        对单帧图像应用动效。
        :param frame: 原始图像 (RGB)
        :param effect_type: 动效类型
        :param progress: 动效进度 0.0 -> 1.0 (0是开始，0.5是峰值，1是结束)
        :param params: 参数字典
        :return: 处理后的图像
        """
        if effect_type == EffectType.NONE:
            return frame

        h, w = frame.shape[:2]
        center = (w // 2, h // 2)

        if effect_type == EffectType.ZOOM_PUNCH:
            # 逻辑：从 1.0 放大到 max_scale，再回缩到 1.0
            # 进度 p: 0 -> 0.5 (放大) -> 1.0 (缩小)
            max_scale = params.get('scale', 1.2)

            if progress < 0.5:
                # 0 -> 0.5 映射到 1.0 -> max
                s = 1.0 + (max_scale - 1.0) * (progress / 0.5)
            else:
                # 0.5 -> 1.0 映射到 max -> 1.0
                s = max_scale - (max_scale - 1.0) * ((progress - 0.5) / 0.5)

            # 使用 affine 变换进行缩放 (以中心为锚点)
            M = cv2.getRotationMatrix2D(center, 0, s)
            return cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        elif effect_type == EffectType.ROTATE_SHAKE:
            # 逻辑：左右快速旋转震动
            # 使用 sin 函数模拟震动
            intensity = params.get('intensity', 5.0)  # 角度
            decay = 1.0 - progress  # 震动逐渐减弱

            angle = np.sin(progress * np.pi * 8) * intensity * decay

            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            return cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        return frame