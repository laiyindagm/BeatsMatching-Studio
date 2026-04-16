from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import QPainter, QColor, QPolygonF, QPen

from core.data_model import ProjectModel
from gui.timeline.time_scaler import TimeScaler
from gui.styles import DesignTokens


class Playhead(QWidget):
    def __init__(self, model: ProjectModel, scaler: TimeScaler, parent=None):
        super().__init__(parent)
        self.model = model
        self.scaler = scaler

        # 透明背景，只画线
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_NoSystemBackground)

        self._color = QColor(DesignTokens.TL_PLAYHEAD)
        self._glow_color = QColor(DesignTokens.ACCENT_BLUE_DIM)

    def paintEvent(self, event):
        x = self.scaler.time_to_pixels(self.model.current_time)

        if x < -10 or x > self.width() + 10:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 1. 底层光晕线 (柔和蓝色 glow, 更宽)
        painter.setPen(QPen(self._glow_color, 4))
        painter.drawLine(x, 0, x, self.height())

        # 2. 上层实线 (精细蓝色)
        painter.setPen(QPen(self._color, 1.5))
        painter.drawLine(x, 0, x, self.height())

        # 3. 头部 (圆角倒梯形)
        head_w = 12
        head_h = 22
        r = 2  # 圆角半径

        points = [
            QPointF(x - head_w / 2 + r, 0),
            QPointF(x + head_w / 2 - r, 0),
            QPointF(x + head_w / 2, r),
            QPointF(x + head_w / 2, head_h - 6),
            QPointF(x, head_h),
            QPointF(x - head_w / 2, head_h - 6),
            QPointF(x - head_w / 2, r),
        ]

        painter.setBrush(self._color)
        painter.setPen(QPen(self._color.lighter(140), 0.5))
        painter.drawPolygon(QPolygonF(points))