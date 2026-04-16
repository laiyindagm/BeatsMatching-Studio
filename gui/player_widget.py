# 视频预览窗口
from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QSizePolicy, QGraphicsDropShadowEffect
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap, QColor, QImage

from gui.styles import DesignTokens


class PlayerWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        DT = DesignTokens

        # 外层容器，提供呼吸间距
        self.setStyleSheet(f"background-color: {DT.BG_DEEPEST};")

        self.label = QLabel("🎬  加载视频以开始")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet(f"""
            background-color: {DT.BG_BASE};
            color: {DT.TEXT_DISABLED};
            border-radius: {DT.RADIUS_LG}px;
            font-size: 13px;
            font-weight: 500;
        """)
        self.label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.label.setScaledContents(False)

        # 柔和投影阴影
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(24)
        shadow.setOffset(0, 4)
        shadow.setColor(QColor(0, 0, 0, 100))
        self.label.setGraphicsEffect(shadow)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.addWidget(self.label)

        # 延迟精渲染定时器
        self._smooth_timer = QTimer(self)
        self._smooth_timer.setSingleShot(True)
        self._smooth_timer.setInterval(150)
        self._smooth_timer.timeout.connect(self._smooth_render)
        self._last_qimage = None

    def update_image(self, q_image: QImage):
        """接收 QImage 并显示，使用 FastTransformation 保证流畅"""
        if q_image.isNull():
            return
        self._last_qimage = q_image
        scaled_pixmap = QPixmap.fromImage(q_image).scaled(
            self.label.size(), Qt.KeepAspectRatio, Qt.FastTransformation
        )
        self.label.setPixmap(scaled_pixmap)
        # 重置精渲染定时器，空闲 150ms 后精渲染
        self._smooth_timer.start()

    def _smooth_render(self):
        """空闲时使用高质量缩放重新渲染"""
        if self._last_qimage is not None and not self._last_qimage.isNull():
            scaled = QPixmap.fromImage(self._last_qimage).scaled(
                self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.label.setPixmap(scaled)