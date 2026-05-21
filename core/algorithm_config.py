"""
算法配置系统
支持从配置文件加载 + GUI 实时修改
包含选项冲突检测
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from enum import Enum, auto
import json
import os


class FusionStrategy(Enum):
    """融合策略"""
    VITPOSE_ONLY = "vitpose_only"      # 仅 ViTPose
    HMR2_ONLY = "hmr2_only"            # 仅 HMR2
    WEIGHTED_SUM = "weighted_sum"      # 加权叠加（默认）
    ADAPTIVE_ALPHA = "adaptive_alpha"  # 自适应权重
    PCA_FUSION = "pca_fusion"          # PCA 多特征融合（实验性）


class VelocityMetric(Enum):
    """速度指标类型"""
    UNIFORM = "uniform"                # 均匀权重
    WEIGHTED = "weighted"              # 加权（末端执行器权重更高）


class RotationMetric(Enum):
    """旋转指标类型"""
    AXIS_ANGLE_DIFF = "axis_angle_diff"   # 轴角差分
    ANGULAR_VELOCITY = "angular_velocity" # 角速度（李代数）


class PeakMode(Enum):
    """关键帧提取模式"""
    PEAKS_ONLY = "peaks_only"          # 仅峰值（动作最激烈时刻）
    VALLEYS_ONLY = "valleys_only"      # 仅谷值（动作停顿/转折时刻）
    PEAKS_AND_VALLEYS = "peaks_and_valleys"  # 峰 + 谷


class AuxiliaryFeature(Enum):
    """辅助特征"""
    NONE = "none"
    ACCELERATION = "acceleration"      # 加速度
    OPTICAL_FLOW = "optical_flow"      # 光流（预留）


@dataclass
class AlgorithmConfig:
    """
    算法配置（可序列化）
    
    所有参数都有默认值，支持从 JSON 加载
    """
    # === 基础设置 ===
    fusion_strategy: str = "weighted_sum"
    velocity_metric: str = "weighted"
    rotation_metric: str = "angular_velocity"
    auxiliary_feature: str = "none"
    
    # === 融合权重（WEIGHTED_SUM 和 ADAPTIVE_ALPHA 用）===
    vitpose_weight: float = 0.6          # α: ViTPose 权重
    hmr2_weight: float = 0.4             # (1-α): HMR2 权重
    
    # === 加权关节速度参数 ===
    # 末端执行器权重更高：手腕、脚踝
    end_effector_weight: float = 2.0     # 末端执行器权重倍数
    torso_weight: float = 1.0            # 躯干权重
    limb_weight: float = 1.2             # 四肢权重
    
    # === 异常检测与插值 ===
    # 异常判定阈值
    outlier_threshold: float = 3.0       # 异常值阈值（标准差倍数）
    min_valid_ratio: float = 0.5         # 最小有效帧比例（低于此值丢弃区间）
    
    # === 节拍检测参数 ===
    gaussian_sigma: float = 5.0          # 高斯平滑 σ（理论最优 ≈ 0.83/(2π·f₀)·fps）
    auto_sigma: bool = True              # 自适应 σ（FFT 估计主频后自动计算）
    peak_height_factor: float = 0.3      # 峰值高度阈值因子（mean + factor * std）
    min_peak_distance: int = 8           # 最小峰值间距（帧，≈0.13s@60fps）
    peak_mode: str = "peaks_and_valleys"        # 关键帧提取模式: peaks_only / valleys_only / peaks_and_valleys
    min_interval_ratio: float = 0.8      # 同模态关键帧最小间距 = ratio × 音频节拍平均间距
    audio_beat_interval: float = 0.0     # 音频节拍平均间距（秒，0=未设置，由提取前自动计算）
    
    # === 匹配约束 ===
    match_algorithm: str = "dp_subset"    # 匹配算法: "greedy" | "dp_full" | "dp_subset"
    speed_ratio_min: float = 0.8         # 最小速度比（输出Δt/输入Δt 下限）
    speed_ratio_max: float = 1.25        # 最大速度比（输出Δt/输入Δt 上限）
    duration_scale_min: float = 0.75     # 输出总时长/输入总时长 下限
    duration_scale_max: float = 1.35     # 输出总时长/输入总时长 上限
    speed_smoothness: float = 4.0        # 速度平滑权重（惩罚相邻段速度比突变，越大越平滑）
    skip_cost: float = 0.1               # 子集 DP 跳过惩罚（越大越鼓励匹配更多关键帧）
    
    # === 自适应阈值 ===
    use_adaptive_threshold: bool = False
    target_beat_count: Optional[int] = None  # 目标节拍数（None 表示自动）
    
    # === 实验性功能 ===
    enable_beat_attraction: bool = False # 节拍吸引力势场
    beat_attraction_sigma: float = 0.15  # 势场宽度（秒）
    beat_attraction_beta: float = 0.3    # 势场强度
    
    # === 可视化 ===
    generate_gif: bool = True            # 是否生成 GIF 动图
    gif_fps: int = 15                    # GIF 帧率
    gif_duration: float = 3.0            # GIF 时长（秒）
    
    def to_dict(self) -> dict:
        """转换为字典（用于 JSON 序列化）"""
        return asdict(self)
    
    def to_json(self, indent: int = 2) -> str:
        """转换为 JSON 字符串"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    @classmethod
    def from_dict(cls, d: dict) -> "AlgorithmConfig":
        """从字典创建"""
        # 过滤掉不存在的字段
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)
    
    @classmethod
    def from_json(cls, s: str) -> "AlgorithmConfig":
        """从 JSON 字符串创建"""
        return cls.from_dict(json.loads(s))
    
    @classmethod
    def load_from_file(cls, path: str) -> "AlgorithmConfig":
        """从文件加载"""
        if not os.path.exists(path):
            return cls()  # 返回默认配置
        with open(path, 'r', encoding='utf-8') as f:
            return cls.from_json(f.read())
    
    def save_to_file(self, path: str):
        """保存到文件"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.to_json())
    
    def check_conflicts(self) -> List[Tuple[str, str]]:
        """
        检查配置冲突
        返回: [(冲突字段, 冲突描述), ...]
        """
        conflicts = []
        
        # 冲突 1: 融合策略与权重设置
        if self.fusion_strategy in ("vitpose_only", "hmr2_only"):
            if self.vitpose_weight != 1.0 and self.fusion_strategy == "vitpose_only":
                conflicts.append(("vitpose_weight", 
                    f"融合策略为 vitpose_only 时，vitpose_weight 应为 1.0，当前为 {self.vitpose_weight}"))
            if self.hmr2_weight != 1.0 and self.fusion_strategy == "hmr2_only":
                conflicts.append(("hmr2_weight",
                    f"融合策略为 hmr2_only 时，hmr2_weight 应为 1.0，当前为 {self.hmr2_weight}"))
        
        # 冲突 2: 权重归一化检查
        if self.fusion_strategy == "weighted_sum":
            total = self.vitpose_weight + self.hmr2_weight
            if abs(total - 1.0) > 0.01:
                conflicts.append(("weights",
                    f"加权融合时权重之和应接近 1.0，当前为 {total:.2f}"))
        
        # 冲突 3: 实验性功能依赖
        if self.enable_beat_attraction and self.target_beat_count is None:
            conflicts.append(("beat_attraction",
                "节拍吸引力势场需要设置 target_beat_count（目标节拍数）"))
        
        # 冲突 4: 辅助特征与融合策略
        if self.auxiliary_feature == "optical_flow" and self.fusion_strategy != "pca_fusion":
            conflicts.append(("optical_flow",
                "光流辅助当前仅在 pca_fusion 策略下有效"))
        
        # 冲突 5: 旋转指标与 HMR2
        if self.rotation_metric == "angular_velocity" and self.fusion_strategy == "vitpose_only":
            conflicts.append(("rotation_metric",
                "角速度指标仅在启用 HMR2 时有效（vitpose_only 模式下被忽略）"))
        
        # 冲突 6: 速度比范围
        if self.speed_ratio_min >= self.speed_ratio_max:
            conflicts.append(("speed_ratio",
                f"速度比下限({self.speed_ratio_min})必须小于上限({self.speed_ratio_max})"))
        
        return conflicts
    
    def auto_fix(self):
        """自动修复可修复的冲突"""
        # 修复权重归一化
        if self.fusion_strategy == "weighted_sum":
            total = self.vitpose_weight + self.hmr2_weight
            if abs(total - 1.0) > 0.01 and total > 0:
                self.vitpose_weight /= total
                self.hmr2_weight /= total
        
        # 修复单模式权重
        if self.fusion_strategy == "vitpose_only":
            self.vitpose_weight = 1.0
            self.hmr2_weight = 0.0
        elif self.fusion_strategy == "hmr2_only":
            self.vitpose_weight = 0.0
            self.hmr2_weight = 1.0


# 默认配置文件路径
DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config", "algorithm_config.json"
)


def load_config(path: Optional[str] = None) -> AlgorithmConfig:
    """加载配置（如果不存在则创建默认）"""
    path = path or DEFAULT_CONFIG_PATH
    config = AlgorithmConfig.load_from_file(path)
    
    # 检查并报告冲突
    conflicts = config.check_conflicts()
    if conflicts:
        print("[AlgorithmConfig] 检测到配置冲突，尝试自动修复...")
        for field, desc in conflicts:
            print(f"  - {field}: {desc}")
        config.auto_fix()
        config.save_to_file(path)
        print("[AlgorithmConfig] 配置已自动修复并保存")
    
    return config


def save_config(config: AlgorithmConfig, path: Optional[str] = None):
    """保存配置"""
    path = path or DEFAULT_CONFIG_PATH
    config.save_to_file(path)
