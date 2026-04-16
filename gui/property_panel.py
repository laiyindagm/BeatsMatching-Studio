from PySide6.QtGui import QPixmap, QIcon, QColor
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QComboBox,
                               QGroupBox, QFormLayout, QDoubleSpinBox,
                               QTabWidget, QSizePolicy)
from PySide6.QtCore import Qt, Signal
from core.data_model import ProjectModel
from core.structures import EaseType, EffectType
from gui.styles import DesignTokens


class PropertyPanel(QWidget):
    def __init__(self, model: ProjectModel, parent=None):
        super().__init__(parent)
        self.model = model
        self.model.keyframes_changed.connect(self.update_panel)  # 监听变化

        self.init_ui()




    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setAlignment(Qt.AlignTop)

        # 标题
        self.lbl_title = QLabel("未选中")
        self.lbl_title.setStyleSheet(f"""
            font-weight: 600;
            font-size: 14px;
            color: {DesignTokens.TEXT_PRIMARY};
            padding: 4px 0;
        """)
        layout.addWidget(self.lbl_title)

        # ── 使用 TabWidget 组织：属性 | 算法配置 ──
        self.tabs = QTabWidget()
        self.tabs.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        layout.addWidget(self.tabs)

        # Tab 1: 关键帧属性
        tab_props = QWidget()
        props_layout = QVBoxLayout(tab_props)
        props_layout.setAlignment(Qt.AlignTop)
        props_layout.setContentsMargins(0, 4, 0, 0)

        # --- 变速设置 (Easing) ---
        self.grp_easing = QGroupBox("变速设置")
        form_ease = QFormLayout(self.grp_easing)

        self.combo_ease = QComboBox()
        # 添加枚举项
        for e in EaseType:
            # 获取该类型对应的颜色
            color = ProjectModel.get_ease_color(e)
            icon = self._create_color_icon(color)

            # 添加带图标的选项
            self.combo_ease.addItem(icon, e.name, e)

        self.combo_ease.currentIndexChanged.connect(self.on_ease_changed)
        form_ease.addRow("函数:", self.combo_ease)

        props_layout.addWidget(self.grp_easing)

        # --- 动效设置 (VFX) ---
        self.grp_vfx = QGroupBox("视觉效果")
        form_vfx = QFormLayout(self.grp_vfx)

        self.combo_vfx = QComboBox()
        for e in EffectType:
            color = ProjectModel.get_effect_color(e)
            icon = self._create_color_icon(color)

            self.combo_vfx.addItem(icon, e.name, e)
        self.combo_vfx.currentIndexChanged.connect(self.on_vfx_changed)

        form_vfx.addRow("效果:", self.combo_vfx)

        props_layout.addWidget(self.grp_vfx)

        # 默认禁用
        self.set_enabled(False)

        # 增加参数控件 (默认隐藏)
        self.grp_params = QGroupBox("效果参数")
        self.layout_params = QFormLayout(self.grp_params)

        self.spin_duration = QDoubleSpinBox()
        self.spin_duration.setRange(0.1, 2.0)
        self.spin_duration.setSingleStep(0.1)
        self.spin_duration.setValue(0.2)
        self.spin_duration.valueChanged.connect(self.on_param_changed)

        self.spin_scale = QDoubleSpinBox()
        self.spin_scale.setRange(1.0, 3.0)
        self.spin_scale.setSingleStep(0.1)
        self.spin_scale.setValue(1.2)
        self.spin_scale.valueChanged.connect(self.on_param_changed)

        self.layout_params.addRow("时长 (s):", self.spin_duration)
        self.layout_params.addRow("强度/缩放:", self.spin_scale)

        props_layout.addWidget(self.grp_params)
        self.grp_params.setVisible(False)  # 初始隐藏

        # --- 音频偏移设置 ---
        self.grp_audio_offset = QGroupBox("音频偏移")
        form_offset = QFormLayout(self.grp_audio_offset)
        self.spin_audio_offset = QDoubleSpinBox()
        self.spin_audio_offset.setRange(-60.0, 60.0)
        self.spin_audio_offset.setSingleStep(0.01)
        self.spin_audio_offset.setDecimals(3)
        self.spin_audio_offset.setSuffix(" s")
        self.spin_audio_offset.setValue(self.model.audio_offset)
        self.spin_audio_offset.setToolTip("音频轴相对于视频轴的时间差（正=音频延后）")
        self.spin_audio_offset.valueChanged.connect(self._on_audio_offset_changed)
        form_offset.addRow("时间差:", self.spin_audio_offset)
        props_layout.addWidget(self.grp_audio_offset)
        # 监听 model 偏移变化（拖拽音频条时同步）
        self.model.audio_data_loaded.connect(self._sync_audio_offset_ui)

        props_layout.addStretch()

        # Tab 2: 算法配置
        from gui.algorithm_config_panel import AlgorithmConfigPanel
        self.algo_config_panel = AlgorithmConfigPanel()
        self.algo_config_panel.config_changed.connect(self._on_algo_config_changed)

        # 添加 Tab
        self.tabs.addTab(tab_props, "属性")
        self.tabs.addTab(self.algo_config_panel, "算法")

    def _on_algo_config_changed(self, config):
        """算法配置变更时的回调，将配置传递给 AlgorithmProxy"""
        # 存储到 model，供 AlgorithmProxy 使用
        self.model.algo_config = config






    def set_enabled(self, enabled):
        self.grp_easing.setEnabled(enabled)
        self.grp_vfx.setEnabled(enabled)
        if not enabled:
            self.lbl_title.setText("未选中")

    def update_panel(self):
        """当选中项改变时刷新面板"""
        selected_kfs = [k for k in self.model.keyframes if k.selected]

        if not selected_kfs:
            self.set_enabled(False)
            return

        self.set_enabled(True)
        count = len(selected_kfs)

        if count == 1:
            kf = selected_kfs[0]
            self.lbl_title.setText(f"关键帧 @ {kf.output_time:.2f}s")

            # 刷新 UI 状态 (block signals 防止死循环)
            self.combo_ease.blockSignals(True)
            self.combo_ease.setCurrentIndex(self.combo_ease.findData(kf.ease_type))
            self.combo_ease.blockSignals(False)

            self.combo_vfx.blockSignals(True)
            self.combo_vfx.setCurrentIndex(self.combo_vfx.findData(kf.effect_type))
            self.combo_vfx.blockSignals(False)

            # 显示参数面板
            if kf.effect_type != EffectType.NONE:
                self.grp_params.setVisible(True)

                # 读取参数
                self.spin_duration.blockSignals(True)
                self.spin_scale.blockSignals(True)

                self.spin_duration.setValue(kf.effect_params.get('duration', 0.2))
                self.spin_scale.setValue(kf.effect_params.get('scale', 1.2))

                self.spin_duration.blockSignals(False)
                self.spin_scale.blockSignals(False)
            else:
                self.grp_params.setVisible(False)

        else:
            self.lbl_title.setText(f"已选中 {count} 项")
            # 多选状态下，如果属性不一致，可以显示空白或保持不变
            # 这里简单处理：显示第一个选中项的属性
            # 但修改时会应用到所有
            pass

    def on_ease_changed(self, index):
        """批量修改变速函数"""
        new_type = self.combo_ease.currentData()
        self.model.batch_update_ease(new_type)


    def on_vfx_changed(self, index):
        """批量修改动效"""
        new_type = self.combo_vfx.currentData()
        self.model.batch_update_vfx(new_type)
        self.update_panel()

    def on_param_changed(self):
        """当参数滑动条改变时，更新 Model"""
        duration = self.spin_duration.value()
        scale = self.spin_scale.value()

        # 批量更新选中项的参数
        # 调用 Model API
        self.model.batch_update_effect_params({
            'duration': duration,
            'scale': scale
        })

    def _create_color_icon(self, color_str: str, size=16):
        """创建一个纯色的方形图标"""
        pixmap = QPixmap(size, size)
        pixmap.fill(QColor(color_str))
        return QIcon(pixmap)

    def _on_audio_offset_changed(self, value):
        """用户修改音频偏移时同步到 model"""
        self.model.set_audio_offset(value)

    def _sync_audio_offset_ui(self):
        """外部修改了偏移（如拖拽音频条），同步到 SpinBox"""
        self.spin_audio_offset.blockSignals(True)
        self.spin_audio_offset.setValue(self.model.audio_offset)
        self.spin_audio_offset.blockSignals(False)