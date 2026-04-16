# 包含滚动条、刻度尺、轨道的容器
# gui/timeline/timeline_container.py

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QScrollArea,
                               QScrollBar, QFrame)
from PySide6.QtCore import Qt

from core.data_model import ProjectModel
from gui.timeline.time_scaler import TimeScaler
from gui.timeline.header import TimeRuler
from gui.timeline.tracks import VideoTrack, AudioTrack
from gui.timeline.playhead import Playhead
from gui.styles import DesignTokens


class TimelineContainer(QWidget):
    def __init__(self, model: ProjectModel, parent=None):
        super().__init__(parent)
        self.model = model

        # 1. 核心工具：Scaler
        self.scaler = TimeScaler()

        # 2. 初始化 UI 组件
        self.playhead = Playhead(self.model, self.scaler, parent=self)
        self.playhead.show()
        self.ruler = TimeRuler(self.model, self.scaler)
        self.video_track = VideoTrack(self.model, self.scaler)
        self.audio_track = AudioTrack(self.model, self.scaler)

        # 底部水平滚动条
        self.h_scrollbar = QScrollBar(Qt.Horizontal)
        self.h_scrollbar.setRange(0, 0)  # 初始没有范围

        # 3. 组装布局
        self.setup_ui()

        # 4. 信号连接
        self.setup_connections()

    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # A. 顶部刻度尺
        main_layout.addWidget(self.ruler)

        # B. 中间轨道区域 (放入 ScrollArea 以支持垂直滚动)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # 禁用自带的水平滚动条

        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.scroll_area.setStyleSheet(f"background-color: {DesignTokens.BG_SURFACE};")

        # 轨道容器 Widget
        self.tracks_container = QWidget()
        self.tracks_layout = QVBoxLayout(self.tracks_container)
        self.tracks_layout.setContentsMargins(0, 0, 0, 0)
        self.tracks_layout.setSpacing(1)  # 轨道间距
        self.tracks_layout.setAlignment(Qt.AlignTop)

        # 添加具体轨道
        self.tracks_layout.addWidget(self.video_track)
        self.tracks_layout.addWidget(self.audio_track)

        self.scroll_area.setWidget(self.tracks_container)
        main_layout.addWidget(self.scroll_area)

        # C. 底部水平滚动条
        main_layout.addWidget(self.h_scrollbar)

    def setup_connections(self):
        # 1. 滚动条 -> Scaler
        self.h_scrollbar.valueChanged.connect(
            lambda val: self.scaler.scroll_to(float(val))
        )

        # 2. Scaler -> 滚动条 (例如当外部代码改变了 offset)
        # 注意：这里需要防止循环信号 (blockSignals)
        self.scaler.view_changed.connect(self.sync_scrollbar_from_scaler)

        # 3. 模型数据变化 -> 更新滚动条范围 (例如视频变长了)
        self.model.data_loaded.connect(self.update_scroll_range)

    def sync_scrollbar_from_scaler(self):
        """当 Scaler 内部状态改变时，同步更新滚动条滑块位置"""
        val = int(self.scaler.view_offset_x)
        if self.h_scrollbar.value() != val:
            self.h_scrollbar.blockSignals(True)
            self.h_scrollbar.setValue(val)
            self.h_scrollbar.blockSignals(False)

    def update_scroll_range(self):
        """根据视频总长度，计算滚动条的最大范围"""
        if self.model.duration <= 0:
            return

        # 总像素长度
        total_pixels = self.scaler.duration_to_pixels(self.model.duration)
        # 额外增加一点空白区域 (padding)，让用户能拖到最后
        total_pixels += 200

        # 视口宽度
        viewport_w = self.width()

        # 最大滚动值 = 内容总宽 - 视口宽
        max_scroll = max(0, int(total_pixels - viewport_w))

        self.h_scrollbar.setRange(0, max_scroll)
        self.h_scrollbar.setPageStep(viewport_w)

    def resizeEvent(self, event):
        """窗口大小改变时，更新 Scaler 的视口宽度信息"""
        super().resizeEvent(event)
        self.scaler.viewport_width = self.width()
        self.update_scroll_range()

        self.playhead.resize(self.size())
        self.playhead.raise_()  # 确保在最上层

    def wheelEvent(self, event):
        """处理滚轮缩放 (Ctrl+滚轮)"""
        if event.modifiers() == Qt.ControlModifier:
            delta = event.angleDelta().y()
            current_zoom = self.scaler.pixels_per_second

            # 缩放系数
            factor = 1.1 if delta > 0 else 0.9

            # 以鼠标位置为中心
            mouse_x = event.position().x()

            self.scaler.set_zoom(current_zoom * factor, mouse_x)

            # 缩放后内容长度变了，更新滚动条范围
            self.update_scroll_range()

            event.accept()
        else:
            # 如果没按 Ctrl，交给父类处理（可能是垂直滚动）
            super().wheelEvent(event)
