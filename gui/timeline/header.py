# 时间刻度尺
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, Signal, QRectF
from PySide6.QtGui import QPainter, QPen, QColor, QFont, QFontMetrics

from core.data_model import ProjectModel
from gui.timeline.time_scaler import TimeScaler
from gui.styles import DesignTokens
from utils.helpers import format_time
import math


class TimeRuler(QWidget):
    """
    时间轴顶部的刻度尺。
    负责绘制时间刻度，并处理点击以移动播放头。
    """

    def __init__(self, model: ProjectModel, scaler: TimeScaler, parent=None):
        super().__init__(parent)
        self.model = model
        self.scaler = scaler
        self.setFixedHeight(30)

        DT = DesignTokens
        self.bg_color = QColor(DT.BG_BASE)
        self.tick_color = QColor(DT.TEXT_DISABLED)
        self.tick_major_color = QColor(DT.TEXT_SECONDARY)
        self.text_color = QColor(DT.TEXT_SECONDARY)
        self.border_color = QColor(DT.BORDER_SUBTLE)
        self.font = QFont("Segoe UI", 8)

        self.scaler.view_changed.connect(self.update)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._update_time_from_mouse(event.position().x())

    def mouseMoveEvent(self, event):
        # 允许按住拖动游标
        if event.buttons() & Qt.LeftButton:
            self._update_time_from_mouse(event.position().x())

    def _update_time_from_mouse(self, x_pixel):
        new_time = self.scaler.pixels_to_time(x_pixel)
        new_time = max(0.0, new_time)

        # 核心：更新 Model 的时间
        # Model 应该有一个 set_current_time 方法，或者直接修改属性
        # 修改属性后，需要发出 playhead_moved 信号通知其他组件
        self.model.set_current_time(new_time)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)
        painter.fillRect(self.rect(), self.bg_color)

        # 底部分割线
        painter.setPen(QPen(self.border_color, 1))
        painter.drawLine(0, self.height() - 1, self.width(), self.height() - 1)

        start_time, end_time = self.scaler.get_visible_range()
        step = self._calculate_step()

        painter.setFont(self.font)

        # 计算第一个刻度
        # 使用 epsilon 避免浮点数精度问题导致的错位
        epsilon = 1e-9
        first_tick_index = math.ceil(start_time / step - epsilon)
        current_t = first_tick_index * step

        while current_t <= end_time + step:
            x = self.scaler.time_to_pixels(current_t)

            # 判断是否显示文字 (Major Tick)
            # 逻辑：
            # 1. 如果 step >= 1秒，每隔 step 显示一次 (都是 Major)
            # 2. 如果 step < 1秒 (如0.1)，我们希望每 0.5s 或 1s 才显示文字，或者全显示？
            #    为了简单且美观，如果 step 很小，我们可以每 5 个 step 算一个大刻度，
            #    或者如果 step 本身就是 0.5，那它就是大刻度。

            is_major = self._is_major_tick(current_t, step)

            if is_major:
                # 大刻度：线长一些，画文字
                tick_height = 12
                painter.setPen(self.tick_major_color)
                painter.drawLine(x, self.height() - 1, x, self.height() - 1 - tick_height)

                # 画文字
                time_str = format_time(current_t)
                fm = QFontMetrics(self.font)
                text_w = fm.horizontalAdvance(time_str)

                painter.setPen(self.text_color)
                painter.drawText(QRectF(x + 3, 0, text_w + 10, 15),
                                 Qt.AlignLeft | Qt.AlignVCenter, time_str)
            else:
                # 小刻度：线短，颜色更淡
                tick_height = 5
                painter.setPen(self.tick_color)
                painter.drawLine(x, self.height() - 1, x, self.height() - 1 - tick_height)

            current_t += step

    def _calculate_step(self) -> float:
        """
        计算最合适的刻度间隔
        """
        pps = self.scaler.pixels_per_second
        min_pixel_spacing = 60.0  # 期望每个刻度文字之间至少隔 60px

        # 候选步长：0.01s, 0.05s, 0.1s, 0.5s, 1s, 2s, 5s, 10s, 30s, 1min...
        potential_steps = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]

        for step in potential_steps:
            if step * pps >= min_pixel_spacing:
                return step

        return 60.0

    def _is_major_tick(self, time_val: float, step: float) -> bool:
        """
        判断是否是大刻度。
        现在的策略简化为：只要 _calculate_step 选出来的 step，都当作 Major 显示文字。
        除非 step 非常小（比如 0.01s），那时才需要每隔5个显示一个。
        目前最小 step 是 0.1s，只要 spacing 设置得当（如60px），全显示文字也没问题。
        """
        # 如果你想做得更像专业软件（如0.1s时，0.1,0.2是小刻度，0.5是大刻度）：
        # 这里我们采用“全显示”策略，依靠 _calculate_step 的 min_pixel_spacing 保证不重叠
        return True
