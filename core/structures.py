# 定义数据类 (Keyframe, BeatInfo)
# core/structures.py

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Any, Optional


class KeyframeSource(Enum):
    """关键帧来源类型（用于可视化区分）"""
    PEAK = auto()       # 峰值（动作最激烈）
    VALLEY = auto()     # 谷值（动作停顿/转折）
    MANUAL = auto()     # 手动添加
    BOUNDARY = auto()   # 首尾边界帧


class EaseType(Enum):
    """
    变速函数类型枚举
    对应需求：提供多个可选变速函数
    """
    LINEAR = auto()  # 线性 (匀速)
    EASE_IN_QUAD = auto()  # 缓入 (先慢后快)
    EASE_OUT_QUAD = auto()  # 缓出 (先快后慢)
    EASE_IN_OUT = auto()  # 缓入缓出 (中间快，两头慢)
    # 可以根据后续需求轻松扩展更多类型


class EffectType(Enum):
    """
    动效类型枚举
    对应需求：关键帧处添加动效
    """
    NONE = auto()
    ZOOM_PUNCH = auto()  # 放大再缩小
    ROTATE_SHAKE = auto()  # 旋转再反转
    DISTORTION = auto()  # 扭曲
    FLASH = auto()  # 闪白
    # ... 更多动效


@dataclass(order=True)
class Keyframe:
    """
    关键帧核心数据结构
    对应需求：包含源帧索引、输出时间、变速函数、动效参数
    """
    # [排序键] 输出时间轴上的时间点 (秒)
    # 设置 order=True 后，Keyframe 列表会自动根据 output_time 排序
    output_time: float

    # 对应的源视频帧索引 (source frame index)
    frame_idx: int = field(compare=False)

    # 该关键帧与其**下一个**关键帧之间的变速函数类型
    ease_type: EaseType = field(default=EaseType.LINEAR, compare=False)

    # 在该关键帧处触发的动效类型
    effect_type: EffectType = field(default=EffectType.NONE, compare=False)

    # 动效的具体参数 (例如：放大倍数=1.2, 持续时间=0.5s)
    effect_params: Dict[str, Any] = field(default_factory=dict, compare=False)

    # 关键帧来源类型（峰/谷/手动/边界）
    source: 'KeyframeSource' = field(default=KeyframeSource.PEAK, compare=False)

    # UI 状态：是否被选中
    selected: bool = field(default=False, compare=False)

    def __repr__(self):
        return (f"<Keyframe T={self.output_time:.2f}s -> Frame={self.frame_idx} "
                f"Ease={self.ease_type.name}>")

    def get_effect_duration(self):
        """获取动效持续时间 (默认 0.2s)"""
        return self.effect_params.get('duration', 0.2)


@dataclass
class BeatInfo:
    """
    音频节拍信息
    """
    time: float  # 节拍发生的时间 (秒)
    strength: float  # 节拍强度 (用于绘制不同高度的线条)
    is_downbeat: bool = False  # 是否为重音/强拍

