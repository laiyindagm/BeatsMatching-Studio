"""
算法配置面板
集成到 PropertyPanel 或作为独立对话框
支持实时修改配置、冲突检测、预设保存/加载
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QDoubleSpinBox, QSpinBox, QCheckBox, QGroupBox, QFormLayout,
    QPushButton, QMessageBox, QFileDialog, QTabWidget, QScrollArea,
    QSizePolicy
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor

import os
import sys

# 路径设置
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from core.algorithm_config import (
    AlgorithmConfig, FusionStrategy, VelocityMetric, 
    RotationMetric, AuxiliaryFeature, load_config, save_config
)


class AlgorithmConfigPanel(QWidget):
    """
    算法配置面板
    
    信号：
        config_changed(AlgorithmConfig): 配置发生变化
    """
    config_changed = Signal(object)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.config = AlgorithmConfig()
        self._updating = False  # 防止递归更新
        
        self.init_ui()
        self.load_current_config()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignTop)
        
        # 标题
        title = QLabel("算法配置")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)
        
        # 使用 TabWidget 组织配置
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        
        # === Tab 1: 基础设置 ===
        self.tab_basic = self._create_basic_tab()
        self.tabs.addTab(self.tab_basic, "基础")
        
        # === Tab 2: 高级参数 ===
        self.tab_advanced = self._create_advanced_tab()
        self.tabs.addTab(self.tab_advanced, "高级")
        
        # === Tab 3: 实验性 ===
        self.tab_experimental = self._create_experimental_tab()
        self.tabs.addTab(self.tab_experimental, "实验性")
        
        # 冲突警告
        self.conflict_label = QLabel("")
        self.conflict_label.setStyleSheet("color: #ff6b6b; font-size: 11px;")
        self.conflict_label.setWordWrap(True)
        layout.addWidget(self.conflict_label)
        
        # 按钮组
        btn_layout = QHBoxLayout()
        
        self.btn_save = QPushButton("保存配置")
        self.btn_save.clicked.connect(self.save_config_to_file)
        btn_layout.addWidget(self.btn_save)
        
        self.btn_load = QPushButton("加载配置")
        self.btn_load.clicked.connect(self.load_config_from_file)
        btn_layout.addWidget(self.btn_load)
        
        self.btn_reset = QPushButton("恢复默认")
        self.btn_reset.clicked.connect(self.reset_to_default)
        btn_layout.addWidget(self.btn_reset)
        
        self.btn_apply = QPushButton("应用")
        self.btn_apply.setStyleSheet("background-color: #4CAF50; color: white;")
        self.btn_apply.clicked.connect(self.apply_config)
        btn_layout.addWidget(self.btn_apply)
        
        layout.addLayout(btn_layout)
        
        # 设置面板大小策略
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        self.setMaximumWidth(400)
    
    def _create_basic_tab(self) -> QWidget:
        """创建基础设置标签页"""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setAlignment(Qt.AlignTop)
        
        # 融合策略
        self.group_fusion = QGroupBox("融合策略")
        form_fusion = QFormLayout(self.group_fusion)
        
        self.combo_fusion = QComboBox()
        self.combo_fusion.addItem("仅 ViTPose", "vitpose_only")
        self.combo_fusion.addItem("仅 HMR2", "hmr2_only")
        self.combo_fusion.addItem("加权融合", "weighted_sum")
        self.combo_fusion.addItem("自适应权重", "adaptive_alpha")
        self.combo_fusion.addItem("PCA 融合 (实验性)", "pca_fusion")
        self.combo_fusion.currentIndexChanged.connect(self._on_fusion_changed)
        form_fusion.addRow("策略:", self.combo_fusion)
        
        # 权重设置（仅在加权融合时启用）
        self.spin_vitpose_weight = QDoubleSpinBox()
        self.spin_vitpose_weight.setRange(0.0, 1.0)
        self.spin_vitpose_weight.setSingleStep(0.1)
        self.spin_vitpose_weight.setDecimals(2)
        self.spin_vitpose_weight.valueChanged.connect(self._on_weight_changed)
        form_fusion.addRow("ViTPose 权重:", self.spin_vitpose_weight)
        
        self.spin_hmr2_weight = QDoubleSpinBox()
        self.spin_hmr2_weight.setRange(0.0, 1.0)
        self.spin_hmr2_weight.setSingleStep(0.1)
        self.spin_hmr2_weight.setDecimals(2)
        self.spin_hmr2_weight.setEnabled(False)  # 自动计算
        form_fusion.addRow("HMR2 权重:", self.spin_hmr2_weight)
        
        layout.addWidget(self.group_fusion)
        
        # 速度指标
        self.group_velocity = QGroupBox("速度指标")
        form_vel = QFormLayout(self.group_velocity)
        
        self.combo_velocity = QComboBox()
        self.combo_velocity.addItem("均匀权重", "uniform")
        self.combo_velocity.addItem("加权（末端执行器优先）", "weighted")
        form_vel.addRow("类型:", self.combo_velocity)
        
        self.spin_end_weight = QDoubleSpinBox()
        self.spin_end_weight.setRange(1.0, 5.0)
        self.spin_end_weight.setSingleStep(0.5)
        self.spin_end_weight.setValue(2.0)
        form_vel.addRow("末端权重:", self.spin_end_weight)
        
        layout.addWidget(self.group_velocity)
        
        # 旋转指标
        self.group_rotation = QGroupBox("旋转指标")
        form_rot = QFormLayout(self.group_rotation)
        
        self.combo_rotation = QComboBox()
        self.combo_rotation.addItem("轴角差分", "axis_angle_diff")
        self.combo_rotation.addItem("角速度（李代数）", "angular_velocity")
        form_rot.addRow("类型:", self.combo_rotation)
        
        layout.addWidget(self.group_rotation)
        
        # 辅助特征
        self.group_aux = QGroupBox("辅助特征")
        form_aux = QFormLayout(self.group_aux)
        
        self.combo_aux = QComboBox()
        self.combo_aux.addItem("无", "none")
        self.combo_aux.addItem("加速度", "acceleration")
        self.combo_aux.addItem("光流 (预留)", "optical_flow")
        form_aux.addRow("特征:", self.combo_aux)
        
        layout.addWidget(self.group_aux)
        
        scroll.setWidget(widget)
        return scroll
    
    def _create_advanced_tab(self) -> QWidget:
        """创建高级参数标签页"""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setAlignment(Qt.AlignTop)
        
        # 异常检测
        self.group_outlier = QGroupBox("异常检测")
        form_out = QFormLayout(self.group_outlier)
        
        self.spin_outlier_thr = QDoubleSpinBox()
        self.spin_outlier_thr.setRange(1.0, 10.0)
        self.spin_outlier_thr.setSingleStep(0.5)
        self.spin_outlier_thr.setValue(3.0)
        form_out.addRow("异常阈值 (σ):", self.spin_outlier_thr)
        
        self.spin_min_valid = QDoubleSpinBox()
        self.spin_min_valid.setRange(0.1, 1.0)
        self.spin_min_valid.setSingleStep(0.1)
        self.spin_min_valid.setValue(0.5)
        form_out.addRow("最小有效比例:", self.spin_min_valid)
        
        layout.addWidget(self.group_outlier)
        
        # 节拍检测
        self.group_beat = QGroupBox("节拍检测")
        form_beat = QFormLayout(self.group_beat)
        
        self.check_auto_sigma = QCheckBox("自适应 σ（FFT 估计）")
        self.check_auto_sigma.setChecked(True)
        self.check_auto_sigma.toggled.connect(self._on_auto_sigma_toggled)
        form_beat.addRow(self.check_auto_sigma)
        
        self.spin_gaussian_sigma = QDoubleSpinBox()
        self.spin_gaussian_sigma.setRange(0.5, 15.0)
        self.spin_gaussian_sigma.setSingleStep(0.5)
        self.spin_gaussian_sigma.setValue(5.0)
        self.spin_gaussian_sigma.setEnabled(False)  # 默认自适应时禁用
        form_beat.addRow("手动 σ:", self.spin_gaussian_sigma)
        
        self.spin_peak_factor = QDoubleSpinBox()
        self.spin_peak_factor.setRange(0.0, 2.0)
        self.spin_peak_factor.setSingleStep(0.1)
        self.spin_peak_factor.setValue(0.3)
        form_beat.addRow("峰值因子:", self.spin_peak_factor)
        
        self.spin_min_distance = QSpinBox()
        self.spin_min_distance.setRange(1, 30)
        self.spin_min_distance.setValue(8)
        form_beat.addRow("最小峰值间距:", self.spin_min_distance)
        
        self.combo_peak_mode = QComboBox()
        self.combo_peak_mode.addItem("仅峰值 (动作最激烈)", "peaks_only")
        self.combo_peak_mode.addItem("仅谷值 (动作停顿点)", "valleys_only")
        self.combo_peak_mode.addItem("峰 + 谷 (全部关键帧)", "peaks_and_valleys")
        form_beat.addRow("关键帧模式:", self.combo_peak_mode)
        
        self.spin_min_interval_ratio = QDoubleSpinBox()
        self.spin_min_interval_ratio.setRange(0.0, 3.0)
        self.spin_min_interval_ratio.setSingleStep(0.1)
        self.spin_min_interval_ratio.setDecimals(1)
        self.spin_min_interval_ratio.setValue(0.8)
        self.spin_min_interval_ratio.setToolTip(
            "同模态关键帧最小间距 = 此值 × 音频节拍平均间距\n"
            "0 = 禁用，仅当音频已加载时生效")
        form_beat.addRow("最小间距比例:", self.spin_min_interval_ratio)
        
        layout.addWidget(self.group_beat)
        
        # 匹配约束
        self.group_match = QGroupBox("匹配约束")
        form_match = QFormLayout(self.group_match)
        
        self.combo_match_algo = QComboBox()
        self.combo_match_algo.addItem("贪心最近邻", "greedy")
        self.combo_match_algo.addItem("全匹配 DP（所有KF→beat）", "dp_full")
        self.combo_match_algo.addItem("子集 DP + 比例插值", "dp_subset")
        self.combo_match_algo.setToolTip(
            "贪心: 简单快速，无参数\n"
            "全匹配 DP: 每个关键帧都严格对齐到 beat\n"
            "子集 DP: 锚点帧严格对齐，非锚点帧等比例插值")
        self.combo_match_algo.currentIndexChanged.connect(self._on_match_algo_changed)
        form_match.addRow("匹配算法:", self.combo_match_algo)
        
        self.spin_speed_ratio_min = QDoubleSpinBox()
        self.spin_speed_ratio_min.setRange(0.1, 1.0)
        self.spin_speed_ratio_min.setSingleStep(0.1)
        self.spin_speed_ratio_min.setDecimals(1)
        self.spin_speed_ratio_min.setValue(0.8)
        form_match.addRow("速度比下限:", self.spin_speed_ratio_min)
        
        self.spin_speed_ratio_max = QDoubleSpinBox()
        self.spin_speed_ratio_max.setRange(1.0, 5.0)
        self.spin_speed_ratio_max.setSingleStep(0.05)
        self.spin_speed_ratio_max.setDecimals(2)
        self.spin_speed_ratio_max.setValue(1.25)
        form_match.addRow("速度比上限:", self.spin_speed_ratio_max)
        
        self.spin_dur_scale_min = QDoubleSpinBox()
        self.spin_dur_scale_min.setRange(0.1, 1.0)
        self.spin_dur_scale_min.setSingleStep(0.05)
        self.spin_dur_scale_min.setDecimals(2)
        self.spin_dur_scale_min.setValue(0.75)
        form_match.addRow("总时长缩放下限:", self.spin_dur_scale_min)
        
        self.spin_dur_scale_max = QDoubleSpinBox()
        self.spin_dur_scale_max.setRange(1.0, 10.0)
        self.spin_dur_scale_max.setSingleStep(0.05)
        self.spin_dur_scale_max.setDecimals(2)
        self.spin_dur_scale_max.setValue(1.35)
        form_match.addRow("总时长缩放上限:", self.spin_dur_scale_max)
        
        self.spin_speed_smoothness = QDoubleSpinBox()
        self.spin_speed_smoothness.setRange(0.0, 20.0)
        self.spin_speed_smoothness.setSingleStep(0.5)
        self.spin_speed_smoothness.setDecimals(1)
        self.spin_speed_smoothness.setValue(4.0)
        self.spin_speed_smoothness.setToolTip(
            "惩罚相邻段速度比突变，越大越平滑\n"
            "0 = 不惩罚，仅约束 ratio 范围")
        self.label_speed_smoothness = QLabel("速度平滑权重:")
        form_match.addRow(self.label_speed_smoothness, self.spin_speed_smoothness)
        
        self.spin_skip_cost = QDoubleSpinBox()
        self.spin_skip_cost.setRange(0.0, 2.0)
        self.spin_skip_cost.setSingleStep(0.05)
        self.spin_skip_cost.setDecimals(2)
        self.spin_skip_cost.setValue(0.1)
        self.spin_skip_cost.setToolTip(
            "每跳过一个未匹配关键帧的代价\n"
            "越大 → 尽可能匹配更多关键帧\n"
            "越小 → 允许更多跳过，整体更平滑")
        self.label_skip_cost = QLabel("跳过惩罚:")
        form_match.addRow(self.label_skip_cost, self.spin_skip_cost)
        
        layout.addWidget(self.group_match)
        
        # 初始状态: 根据默认算法更新可见性
        self._on_match_algo_changed()
        
        # 自适应阈值
        self.group_adaptive = QGroupBox("自适应阈值")
        form_adapt = QFormLayout(self.group_adaptive)
        
        self.check_adaptive = QCheckBox("启用自适应阈值")
        form_adapt.addRow(self.check_adaptive)
        
        self.spin_target_beats = QSpinBox()
        self.spin_target_beats.setRange(0, 1000)
        self.spin_target_beats.setSpecialValueText("自动")
        self.spin_target_beats.setValue(0)
        form_adapt.addRow("目标节拍数:", self.spin_target_beats)
        
        layout.addWidget(self.group_adaptive)
        
        # 可视化
        self.group_viz = QGroupBox("可视化")
        form_viz = QFormLayout(self.group_viz)
        
        self.check_generate_gif = QCheckBox("生成 GIF 动图")
        self.check_generate_gif.setChecked(True)
        form_viz.addRow(self.check_generate_gif)
        
        self.spin_gif_fps = QSpinBox()
        self.spin_gif_fps.setRange(5, 60)
        self.spin_gif_fps.setValue(15)
        form_viz.addRow("GIF 帧率:", self.spin_gif_fps)
        
        self.spin_gif_duration = QDoubleSpinBox()
        self.spin_gif_duration.setRange(1.0, 10.0)
        self.spin_gif_duration.setValue(3.0)
        form_viz.addRow("GIF 时长 (秒):", self.spin_gif_duration)
        
        layout.addWidget(self.group_viz)
        
        scroll.setWidget(widget)
        return scroll
    
    def _create_experimental_tab(self) -> QWidget:
        """创建实验性功能标签页"""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setAlignment(Qt.AlignTop)
        
        # 节拍吸引力
        self.group_attraction = QGroupBox("节拍吸引力势场")
        form_attr = QFormLayout(self.group_attraction)
        
        self.check_attraction = QCheckBox("启用")
        form_attr.addRow(self.check_attraction)
        
        self.spin_attr_sigma = QDoubleSpinBox()
        self.spin_attr_sigma.setRange(0.05, 1.0)
        self.spin_attr_sigma.setSingleStep(0.05)
        self.spin_attr_sigma.setValue(0.15)
        form_attr.addRow("势场宽度 (秒):", self.spin_attr_sigma)
        
        self.spin_attr_beta = QDoubleSpinBox()
        self.spin_attr_beta.setRange(0.0, 1.0)
        self.spin_attr_beta.setSingleStep(0.1)
        self.spin_attr_beta.setValue(0.3)
        form_attr.addRow("势场强度:", self.spin_attr_beta)
        
        layout.addWidget(self.group_attraction)
        
        # 警告标签
        warning = QLabel("⚠️ 实验性功能可能不稳定，建议先在小视频上测试")
        warning.setStyleSheet("color: #ffa500; font-size: 11px;")
        warning.setWordWrap(True)
        layout.addWidget(warning)
        
        scroll.setWidget(widget)
        return scroll
    
    def _on_fusion_changed(self, index):
        """融合策略改变时更新 UI 状态"""
        strategy = self.combo_fusion.currentData()
        
        # 根据策略启用/禁用权重设置
        if strategy in ("vitpose_only", "hmr2_only"):
            self.spin_vitpose_weight.setEnabled(False)
            self.spin_hmr2_weight.setEnabled(False)
            if strategy == "vitpose_only":
                self.spin_vitpose_weight.setValue(1.0)
            else:
                self.spin_vitpose_weight.setValue(0.0)
        else:
            self.spin_vitpose_weight.setEnabled(True)
        
        self._check_conflicts()
    
    def _on_weight_changed(self, value):
        """权重改变时自动更新另一项"""
        self.spin_hmr2_weight.setValue(1.0 - value)
        self._check_conflicts()
    
    def _on_match_algo_changed(self, index=None):
        """匹配算法切换时更新参数可见性"""
        algo = self.combo_match_algo.currentData()
        is_dp = algo in ('dp_full', 'dp_subset')
        is_subset = algo == 'dp_subset'
        
        # DP 参数: 仅 dp_full / dp_subset 可见
        self.spin_speed_ratio_min.setEnabled(is_dp)
        self.spin_speed_ratio_max.setEnabled(is_dp)
        self.spin_dur_scale_min.setEnabled(is_dp)
        self.spin_dur_scale_max.setEnabled(is_dp)
        self.spin_speed_smoothness.setEnabled(is_dp)
        self.label_speed_smoothness.setEnabled(is_dp)
        
        # 跳过惩罚: 仅 dp_subset 可见
        self.spin_skip_cost.setEnabled(is_subset)
        self.label_skip_cost.setEnabled(is_subset)
    
    def _on_auto_sigma_toggled(self, checked):
        """自适应 σ 切换时禁用/启用手动 σ"""
        self.spin_gaussian_sigma.setEnabled(not checked)
    
    def _check_conflicts(self):
        """检查配置冲突并显示警告"""
        config = self._gather_config()
        conflicts = config.check_conflicts()
        
        if conflicts:
            text = "⚠️ 配置冲突:\n"
            for field, desc in conflicts:
                text += f"  • {desc}\n"
            self.conflict_label.setText(text)
            self.btn_apply.setEnabled(False)
        else:
            self.conflict_label.setText("")
            self.btn_apply.setEnabled(True)
    
    def _gather_config(self) -> AlgorithmConfig:
        """从 UI 收集配置"""
        return AlgorithmConfig(
            fusion_strategy=self.combo_fusion.currentData(),
            velocity_metric=self.combo_velocity.currentData(),
            rotation_metric=self.combo_rotation.currentData(),
            auxiliary_feature=self.combo_aux.currentData(),
            vitpose_weight=self.spin_vitpose_weight.value(),
            hmr2_weight=self.spin_hmr2_weight.value(),
            end_effector_weight=self.spin_end_weight.value(),
            outlier_threshold=self.spin_outlier_thr.value(),
            min_valid_ratio=self.spin_min_valid.value(),
            gaussian_sigma=self.spin_gaussian_sigma.value(),
            auto_sigma=self.check_auto_sigma.isChecked(),
            peak_height_factor=self.spin_peak_factor.value(),
            min_peak_distance=self.spin_min_distance.value(),
            peak_mode=self.combo_peak_mode.currentData(),
            min_interval_ratio=self.spin_min_interval_ratio.value(),
            match_algorithm=self.combo_match_algo.currentData(),
            speed_ratio_min=self.spin_speed_ratio_min.value(),
            speed_ratio_max=self.spin_speed_ratio_max.value(),
            duration_scale_min=self.spin_dur_scale_min.value(),
            duration_scale_max=self.spin_dur_scale_max.value(),
            speed_smoothness=self.spin_speed_smoothness.value(),
            skip_cost=self.spin_skip_cost.value(),
            use_adaptive_threshold=self.check_adaptive.isChecked(),
            target_beat_count=self.spin_target_beats.value() if self.spin_target_beats.value() > 0 else None,
            enable_beat_attraction=self.check_attraction.isChecked(),
            beat_attraction_sigma=self.spin_attr_sigma.value(),
            beat_attraction_beta=self.spin_attr_beta.value(),
            generate_gif=self.check_generate_gif.isChecked(),
            gif_fps=self.spin_gif_fps.value(),
            gif_duration=self.spin_gif_duration.value(),
        )
    
    def _apply_config_to_ui(self, config: AlgorithmConfig):
        """将配置应用到 UI"""
        self._updating = True
        
        # 基础设置
        idx = self.combo_fusion.findData(config.fusion_strategy)
        if idx >= 0:
            self.combo_fusion.setCurrentIndex(idx)
        
        idx = self.combo_velocity.findData(config.velocity_metric)
        if idx >= 0:
            self.combo_velocity.setCurrentIndex(idx)
        
        idx = self.combo_rotation.findData(config.rotation_metric)
        if idx >= 0:
            self.combo_rotation.setCurrentIndex(idx)
        
        idx = self.combo_aux.findData(config.auxiliary_feature)
        if idx >= 0:
            self.combo_aux.setCurrentIndex(idx)
        
        # 权重
        self.spin_vitpose_weight.setValue(config.vitpose_weight)
        self.spin_hmr2_weight.setValue(config.hmr2_weight)
        self.spin_end_weight.setValue(config.end_effector_weight)
        
        # 高级参数
        self.spin_outlier_thr.setValue(config.outlier_threshold)
        self.spin_min_valid.setValue(config.min_valid_ratio)
        self.check_auto_sigma.setChecked(config.auto_sigma)
        self.spin_gaussian_sigma.setValue(config.gaussian_sigma)
        self.spin_gaussian_sigma.setEnabled(not config.auto_sigma)
        self.spin_peak_factor.setValue(config.peak_height_factor)
        self.spin_min_distance.setValue(config.min_peak_distance)
        
        idx = self.combo_peak_mode.findData(getattr(config, 'peak_mode', 'peaks_and_valleys'))
        if idx >= 0:
            self.combo_peak_mode.setCurrentIndex(idx)
        
        # 匹配约束
        algo = getattr(config, 'match_algorithm', 'dp_subset')
        if algo == 'dp':
            algo = 'dp_subset'  # 兼容旧配置
        algo_idx = self.combo_match_algo.findData(algo)
        if algo_idx >= 0:
            self.combo_match_algo.setCurrentIndex(algo_idx)
        self.spin_speed_ratio_min.setValue(config.speed_ratio_min)
        self.spin_speed_ratio_max.setValue(config.speed_ratio_max)
        self.spin_dur_scale_min.setValue(getattr(config, 'duration_scale_min', 0.75))
        self.spin_dur_scale_max.setValue(getattr(config, 'duration_scale_max', 1.35))
        self.spin_speed_smoothness.setValue(getattr(config, 'speed_smoothness', 4.0))
        self.spin_skip_cost.setValue(getattr(config, 'skip_cost', 0.1))
        self.spin_min_interval_ratio.setValue(getattr(config, 'min_interval_ratio', 0.8))
        self._on_match_algo_changed()
        
        # 自适应阈值
        self.check_adaptive.setChecked(config.use_adaptive_threshold)
        if config.target_beat_count:
            self.spin_target_beats.setValue(config.target_beat_count)
        else:
            self.spin_target_beats.setValue(0)
        
        # 可视化
        self.check_generate_gif.setChecked(config.generate_gif)
        self.spin_gif_fps.setValue(config.gif_fps)
        self.spin_gif_duration.setValue(config.gif_duration)
        
        # 实验性
        self.check_attraction.setChecked(config.enable_beat_attraction)
        self.spin_attr_sigma.setValue(config.beat_attraction_sigma)
        self.spin_attr_beta.setValue(config.beat_attraction_beta)
        
        self._updating = False
        self._check_conflicts()
    
    def load_current_config(self):
        """加载当前配置文件"""
        try:
            self.config = load_config()
            self._apply_config_to_ui(self.config)
        except Exception as e:
            QMessageBox.warning(self, "加载失败", f"无法加载配置: {e}")
    
    def save_config_to_file(self):
        """保存配置到文件"""
        config = self._gather_config()
        
        # 检查冲突
        conflicts = config.check_conflicts()
        if conflicts:
            reply = QMessageBox.question(
                self, "配置冲突",
                "检测到配置冲突，是否自动修复后保存？",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                config.auto_fix()
            else:
                return
        
        try:
            save_config(config)
            self.config = config
            QMessageBox.information(self, "保存成功", "配置已保存")
        except Exception as e:
            QMessageBox.critical(self, "保存失败", str(e))
    
    def load_config_from_file(self):
        """从文件加载配置"""
        path, _ = QFileDialog.getOpenFileName(
            self, "加载配置", "", "JSON Files (*.json)"
        )
        if path:
            try:
                self.config = AlgorithmConfig.load_from_file(path)
                self._apply_config_to_ui(self.config)
                QMessageBox.information(self, "加载成功", f"已加载: {path}")
            except Exception as e:
                QMessageBox.critical(self, "加载失败", str(e))
    
    def reset_to_default(self):
        """恢复默认配置"""
        reply = QMessageBox.question(
            self, "确认",
            "确定要恢复默认配置吗？",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.config = AlgorithmConfig()
            self._apply_config_to_ui(self.config)
    
    def apply_config(self):
        """应用配置并发射信号"""
        config = self._gather_config()
        
        # 自动修复冲突
        conflicts = config.check_conflicts()
        if conflicts:
            config.auto_fix()
            self._apply_config_to_ui(config)  # 更新 UI 显示修复后的值
        
        self.config = config
        self.config_changed.emit(config)
        QMessageBox.information(self, "已应用", "配置已应用到当前会话")
    
    def get_config(self) -> AlgorithmConfig:
        """获取当前配置"""
        return self.config


# 独立对话框版本
class AlgorithmConfigDialog(QWidget):
    """算法配置对话框（可浮动）"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("算法配置")
        self.resize(450, 600)
        
        layout = QVBoxLayout(self)
        self.panel = AlgorithmConfigPanel()
        layout.addWidget(self.panel)
