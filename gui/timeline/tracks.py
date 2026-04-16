# gui/timeline/tracks.py

from PySide6.QtWidgets import QWidget, QRubberBand
from PySide6.QtCore import Qt, QRectF, QPointF, QLineF, QTimer, QTime, QPoint, QRect, QSize
from PySide6.QtGui import QPainter, QColor, QBrush, QPen, QPolygonF, QPainterPath, QLinearGradient

from core.data_model import ProjectModel
from core.structures import Keyframe, EaseType, KeyframeSource
from gui.timeline.time_scaler import TimeScaler
from gui.styles import DesignTokens


class TrackConfig:
    """轨道视觉配置"""
    TRACK_HEIGHT = 80  # 轨道控件总高度
    CLIP_HEIGHT = 50  # 内部色块/波形的高度
    # 自动居中计算: (80 - 50) / 2 = 15
    MARGIN_TOP = (TRACK_HEIGHT - CLIP_HEIGHT) / 2

    KF_SIZE = 16  # 关键帧大小

    @staticmethod
    def get_center_y():
        """获取轨道内容的中心 Y 坐标"""
        return TrackConfig.MARGIN_TOP + TrackConfig.CLIP_HEIGHT / 2


class BaseTrack(QWidget):
    """所有轨道的基类"""

    def __init__(self, model: ProjectModel, scaler: TimeScaler, parent=None):
        super().__init__(parent)
        self.model = model
        self.scaler = scaler
        # 使用统一的高度
        self.setFixedHeight(TrackConfig.TRACK_HEIGHT)
        self.setAutoFillBackground(True)

        self.bg_color = QColor(DesignTokens.BG_SURFACE)
        self.border_color = QColor(DesignTokens.BG_ELEVATED)

        self.scaler.view_changed.connect(self.update)
        # 最好在子类监听 model 信号，或者在这里通用监听

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.bg_color)
        # 底部分割线
        painter.setPen(QPen(QColor(DesignTokens.BORDER_SUBTLE), 1))
        painter.drawLine(0, self.height() - 1, self.width(), self.height() - 1)
        # 顶部微妙高光线
        painter.setPen(QPen(QColor(255, 255, 255, 8), 1))
        painter.drawLine(0, 0, self.width(), 0)


class VideoTrack(BaseTrack):
    KF_COLOR = QColor(DesignTokens.TL_KF_DEFAULT)
    KF_SELECT_COLOR = QColor(DesignTokens.TL_KF_SELECTED)
    KF_HOVER_COLOR = QColor(DesignTokens.TL_KF_HOVER)

    def __init__(self, model, scaler):
        super().__init__(model, scaler)  # 高度由 BaseTrack 统一设置
        self.clip_color = QColor(DesignTokens.TL_CLIP_COLOR)

        self.hover_kf_index = -1
        self.dragging_kf_index = -1
        self.drag_start_x = 0
        self.drag_start_time = 0
        self.hover_segment_index = -1

        self.setMouseTracking(True)
        self.model.keyframes_changed.connect(self.update)

        # 选中态动画定时器
        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self.update)  # 触发重绘
        self.anim_timer.start(100)  # 10 FPS 足够产生流动感

        self.is_rubber_banding = False
        self.rubber_band_origin = QPoint()
        self.rubber_band_rect = QRect()

        # 使用 Qt 自带的 QRubberBand 组件，比自己画方便，而且样式统一
        self.rubber_band = QRubberBand(QRubberBand.Rectangle, self)

    def _get_keyframe_rect(self, time_sec: float) -> QRectF:
        """统一计算关键帧矩形"""
        x = self.scaler.time_to_pixels(time_sec)
        y = TrackConfig.get_center_y()
        s = TrackConfig.KF_SIZE
        return QRectF(x - s / 2, y - s / 2, s, s)

    def _draw_diamond(self, painter, x, y, color):
        """绘制峰值关键帧（菱形 ◆）"""
        s = TrackConfig.KF_SIZE / 2
        points = [
            QPointF(x, y - s),
            QPointF(x + s, y),
            QPointF(x, y + s),
            QPointF(x - s, y)
        ]
        poly = QPolygonF(points)
        painter.setBrush(QBrush(color))
        painter.setPen(QPen(color.darker(160), 1))
        painter.drawPolygon(poly)

    def _draw_valley_marker(self, painter, x, y, color):
        """绘制谷值关键帧（倒三角 ▽）"""
        s = TrackConfig.KF_SIZE / 2
        points = [
            QPointF(x - s, y - s * 0.6),
            QPointF(x + s, y - s * 0.6),
            QPointF(x, y + s),
        ]
        poly = QPolygonF(points)
        painter.setBrush(QBrush(color))
        painter.setPen(QPen(color.darker(160), 1))
        painter.drawPolygon(poly)

    def _draw_keyframe_shape(self, painter, cx, cy, color, source):
        """根据来源类型绘制不同形状的关键帧"""
        if source == KeyframeSource.VALLEY:
            self._draw_valley_marker(painter, cx, cy, color)
        else:
            self._draw_diamond(painter, cx, cy, color)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        if self.model.duration <= 0 or not self.model.keyframes:
            painter.setPen(QColor(DesignTokens.TEXT_DISABLED))
            painter.drawText(self.rect(), Qt.AlignCenter, "等待帧数据...")
            return


        painter.setRenderHint(QPainter.Antialiasing)

        # 获取当前时间用于计算虚线偏移
        current_msec = QTime.currentTime().msecsSinceStartOfDay()
        dash_offset = (current_msec / 50) % 10  # 移动速度

        # 1. 计算视频条位置 (基于首尾关键帧)
        start_time = self.model.keyframes[0].output_time
        end_time = self.model.keyframes[-1].output_time

        x_start = self.scaler.time_to_pixels(start_time)
        width_px = self.scaler.time_to_pixels(end_time) - x_start

        # 使用统一配置
        y = TrackConfig.MARGIN_TOP
        h = TrackConfig.CLIP_HEIGHT
        center_y = TrackConfig.get_center_y()  # 关键

        clip_rect = QRectF(x_start, y, width_px, h)

        if clip_rect.right() > 0 and clip_rect.left() < self.width():
            # 渐变填充 + 顶部高光
            clip_grad = QLinearGradient(clip_rect.left(), y, clip_rect.left(), y + h)
            clip_grad.setColorAt(0.0, self.clip_color.lighter(120))
            clip_grad.setColorAt(0.15, self.clip_color)
            clip_grad.setColorAt(1.0, self.clip_color.darker(130))
            painter.setBrush(QBrush(clip_grad))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(clip_rect, 4, 4)
            # 顶部高光线
            painter.setPen(QPen(QColor(255, 255, 255, 20), 1))
            painter.drawLine(clip_rect.left() + 4, y, clip_rect.right() - 4, y)

        # 2. 绘制连线
        if len(self.model.keyframes) > 1:
            for i in range(len(self.model.keyframes) - 1):
                kf_start = self.model.keyframes[i]
                kf_end = self.model.keyframes[i + 1]

                x1 = self.scaler.time_to_pixels(kf_start.output_time)
                x2 = self.scaler.time_to_pixels(kf_end.output_time)
                if x2 < 0 or x1 > self.width(): continue

                # 获取颜色
                base_color = QColor(ProjectModel.get_ease_color(kf_start.ease_type))

                # 状态处理
                is_selected = kf_start.selected
                is_hover = (i == self.hover_segment_index)

                # 构建路径
                path = QPainterPath()
                path.moveTo(x1, center_y)
                if kf_start.ease_type == EaseType.LINEAR:
                    path.lineTo(x2, center_y)
                elif kf_start.ease_type == EaseType.EASE_IN_QUAD:
                    ctrl1 = QPointF(x1 + (x2 - x1) * 0.7, center_y)
                    path.quadTo(ctrl1, QPointF(x2, center_y))
                elif kf_start.ease_type == EaseType.EASE_OUT_QUAD:
                    ctrl1 = QPointF(x1 + (x2 - x1) * 0.3, center_y)
                    path.quadTo(ctrl1, QPointF(x2, center_y))
                else:  # EASE_IN_OUT
                    path.cubicTo(QPointF(x1 + (x2 - x1) * 0.5, center_y),
                                 QPointF(x1 + (x2 - x1) * 0.5, center_y),
                                 QPointF(x2, center_y))

                # 绘制选中态 (底层光晕/虚线)
                if is_selected:
                    glow_pen = QPen(QColor("#ffffff"), 8)
                    glow_pen.setStyle(Qt.CustomDashLine)
                    glow_pen.setDashPattern([4, 4])  # 虚线模式
                    glow_pen.setDashOffset(dash_offset)  # 流动效果
                    painter.setPen(glow_pen)
                    painter.drawPath(path)

                # 绘制本体
                width = 7 if is_hover else 5
                if is_hover: base_color = base_color.lighter(130)

                painter.setPen(QPen(base_color, width))
                painter.drawPath(path)

        # 3. 绘制关键帧点
        for i, kf in enumerate(self.model.keyframes):
            # 使用统一方法获取 rect
            rect = self._get_keyframe_rect(kf.output_time)

            # 视口优化
            if rect.right() < 0 or rect.left() > self.width():
                continue

            # 根据来源类型选择颜色
            kf_source = getattr(kf, 'source', KeyframeSource.PEAK)
            if kf_source == KeyframeSource.VALLEY:
                base_color = QColor(DesignTokens.ACCENT_BLUE)  # 蓝色 = 谷(停顿)
            elif kf_source == KeyframeSource.BOUNDARY:
                base_color = QColor(DesignTokens.TEXT_DISABLED)  # 灰色 = 边界
            else:
                base_color = QColor(DesignTokens.ACCENT_ORANGE)  # 橙色 = 峰(激烈)
            # 动效色覆盖
            effect_color = ProjectModel.get_effect_color(kf.effect_type)
            if effect_color != DesignTokens.TL_KF_DEFAULT:
                base_color = QColor(effect_color)

            # 状态处理
            is_hover = (i == self.hover_kf_index)
            if is_hover: base_color = base_color.lighter(150)

            cx, cy = rect.center().x(), rect.center().y()

            # 绘制选中态 (流动虚线框 + 外发光)
            if kf.selected:
                # 外发光 (柔和蓝色光环)
                glow_s = TrackConfig.KF_SIZE / 2 + 6
                glow_pen = QPen(QColor(DesignTokens.ACCENT_BLUE_DIM), 4)
                painter.setPen(glow_pen)
                painter.setBrush(Qt.NoBrush)
                glow_pts = [
                    QPointF(cx, cy - glow_s), QPointF(cx + glow_s, cy),
                    QPointF(cx, cy + glow_s), QPointF(cx - glow_s, cy)
                ]
                painter.drawPolygon(QPolygonF(glow_pts))

                # 流动虚线框
                s_sel = TrackConfig.KF_SIZE / 2 + 3
                sel_pen = QPen(QColor("#ffffff"), 1.5)
                sel_pen.setStyle(Qt.CustomDashLine)
                sel_pen.setDashPattern([3, 3])
                sel_pen.setDashOffset(-dash_offset)

                painter.setPen(sel_pen)
                painter.setBrush(Qt.NoBrush)
                points_sel = [
                    QPointF(cx, cy - s_sel), QPointF(cx + s_sel, cy),
                    QPointF(cx, cy + s_sel), QPointF(cx - s_sel, cy)
                ]
                painter.drawPolygon(QPolygonF(points_sel))

            # 绘制本体 (根据来源选择形状)
            self._draw_keyframe_shape(painter, cx, cy, base_color, kf_source)

    def mouseMoveEvent(self, event):
        pos = event.position()
        center_y = TrackConfig.get_center_y()  # 使用统一配置

        if self.dragging_kf_index != -1:
            if self.dragging_kf_index == 0: return  # 防御性编程

            delta_x = pos.x() - self.drag_start_x
            delta_time = self.scaler.pixels_to_duration(delta_x)
            new_time = self.drag_start_time + delta_time

            new_time = max(0.0, new_time)
            if self.dragging_kf_index > 0:
                prev_time = self.model.keyframes[self.dragging_kf_index - 1].output_time
                new_time = max(new_time, prev_time + 0.01)
            if self.dragging_kf_index < len(self.model.keyframes) - 1:
                next_time = self.model.keyframes[self.dragging_kf_index + 1].output_time
                new_time = min(new_time, next_time - 0.01)

            self.model.update_keyframe_time_preview(self.dragging_kf_index, new_time)
            return

        if self.is_rubber_banding:
            self.rubber_band.setGeometry(QRect(self.rubber_band_origin, event.position().toPoint()).normalized())
            return  # 框选时不处理悬停检测
        # 悬停检测
        found_kf = -1
        # 1. 检测点
        for i, kf in enumerate(self.model.keyframes):
            rect = self._get_keyframe_rect(kf.output_time)
            if rect.adjusted(-3, -3, 3, 3).contains(pos):
                found_kf = i
                break

        found_seg = -1
        # 2. 检测线 (如果Y轴在范围内)
        if found_kf == -1 and abs(pos.y() - center_y) < 10:
            x = pos.x()
            t = self.scaler.pixels_to_time(x)
            for i in range(len(self.model.keyframes) - 1):
                t1 = self.model.keyframes[i].output_time
                t2 = self.model.keyframes[i + 1].output_time
                if t1 <= t <= t2:
                    found_seg = i
                    break

        need_update = False
        if self.hover_kf_index != found_kf:
            self.hover_kf_index = found_kf
            need_update = True
        if self.hover_segment_index != found_seg:
            self.hover_segment_index = found_seg
            need_update = True
        if need_update:
            self.update()

        if found_kf != -1:
            self.setCursor(Qt.SizeHorCursor)
        elif found_seg != -1:
            self.setCursor(Qt.PointingHandCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            is_ctrl = (event.modifiers() == Qt.ControlModifier)
            if self.hover_kf_index != -1:
                # 拦截首帧
                if self.hover_kf_index == 0:
                    # 可以选择，但不能拖拽
                    self.model.select_keyframe(0, is_ctrl)
                    self.dragging_kf_index = -1
                    return

                self.model.select_keyframe(self.hover_kf_index, is_ctrl)
                self.dragging_kf_index = self.hover_kf_index
                self.drag_start_x = event.position().x()
                self.drag_start_time = self.model.keyframes[self.dragging_kf_index].output_time
            elif self.hover_segment_index != -1:
                self.model.select_keyframe(self.hover_segment_index, is_ctrl)
            else:
                self.model.deselect_all_keyframes()
                self.dragging_kf_index = -1

                # 开始框选
                self.is_rubber_banding = True
                self.rubber_band_origin = event.position().toPoint()
                self.rubber_band.setGeometry(QRect(self.rubber_band_origin, QSize()))
                self.rubber_band.show()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.dragging_kf_index != -1:
                # 拖拽结束，提交最终 Command
                current_time = self.model.keyframes[self.dragging_kf_index].output_time

                # 只有当真的移动了才 push
                if abs(current_time - self.drag_start_time) > 1e-6:
                    self.model.update_keyframe_time(
                        self.dragging_kf_index,
                        current_time,
                        self.drag_start_time
                    )

            if self.is_rubber_banding:
                self.is_rubber_banding = False
                self.rubber_band.hide()

                # 计算选中项
                rect = self.rubber_band.geometry()
                self._select_items_in_rect(rect, event.modifiers() == Qt.ControlModifier)

            self.dragging_kf_index = -1

    def _select_items_in_rect(self, rect: QRect, multi_select: bool):
        """选中矩形内的元素"""
        center_y = TrackConfig.get_center_y()

        found_indices = []

        # 1. 检测关键帧点
        for i, kf in enumerate(self.model.keyframes):
            # 获取关键帧的屏幕坐标
            x = self.scaler.time_to_pixels(kf.output_time)
            # 简单的点检测，或者用菱形 rect 检测
            # 这里只要中心点在框内就算
            if rect.contains(int(x), int(center_y)):
                found_indices.append(i)

        # 2. 检测线段

        # 应用选中
        if not multi_select:
            # 如果不是追加模式，没按Ctrl就重置
            self.model.deselect_all_keyframes()

        for idx in found_indices:
            # 这里的 select 应该是“置为 True”，而不是 toggle
            # model.select_keyframe 默认逻辑是 toggle (如果是 multi_select=True)
            # 我们需要一个 explicit set selected 的方法
            self.model.keyframes[idx].selected = True

        if found_indices:
            self.model.keyframes_changed.emit()


class AudioTrack(BaseTrack):
    def __init__(self, model, scaler):
        super().__init__(model, scaler)  # 自动使用 TRACK_HEIGHT
        self.wave_color_top = QColor(DesignTokens.TL_WAVEFORM_TOP)
        self.wave_color_bottom = QColor(DesignTokens.TL_WAVEFORM_BOTTOM)
        self.beat_color = QColor(DesignTokens.TL_BEAT)
        self.beat_glow_color = QColor(DesignTokens.ACCENT_RED_DIM)

        self.dragging = False
        self.drag_start_x = 0
        self.original_offset = 0.0

        # 监听音频加载完成，触发重绘
        self.model.audio_data_loaded.connect(self.update)

    def paintEvent(self, event):
        super().paintEvent(event)
        # 1. 检查数据
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 1. 如果没有数据，画提示或空占位
        if self.model.audio_waveform is None or len(self.model.audio_waveform) == 0:
            if self.model.audio_path:
                painter.setPen(QColor(DesignTokens.TEXT_DISABLED))
                painter.drawText(self.rect(), Qt.AlignCenter, "等待波形数据...")
            return

        # 2. 计算几何参数
        # 假设 waveform 的采样率是每秒 100 点
        points_per_sec = 100

        # 视口范围 (时间)
        view_start, view_end = self.scaler.get_visible_range()

        # 转换为 waveform 数组的索引范围
        idx_start = int((view_start - self.model.audio_offset) * points_per_sec)
        idx_end = int((view_end - self.model.audio_offset) * points_per_sec)

        # 边界限制
        idx_start = max(0, idx_start)
        idx_end = min(len(self.model.audio_waveform), idx_end)

        if idx_start >= idx_end:
            return

        # 3. 绘制波形 (渐变填充)
        center_y = TrackConfig.get_center_y()
        max_h = TrackConfig.CLIP_HEIGHT / 2 - 2

        # 构建波形渐变
        wave_grad = QLinearGradient(0, center_y - max_h, 0, center_y + max_h)
        wave_grad.setColorAt(0.0, self.wave_color_top)
        wave_grad.setColorAt(0.5, self.wave_color_top.darker(110))
        wave_grad.setColorAt(1.0, self.wave_color_bottom)

        painter.setPen(QPen(QBrush(wave_grad), 1))

        # 优化步长
        step = 1
        if self.scaler.pixels_per_second < 20:
            step = 5
        elif self.scaler.pixels_per_second < 50:
            step = 2

        visible_data = self.model.audio_waveform[idx_start:idx_end:step]

        # 构建 Path
        lines = []
        for i, amp in enumerate(visible_data):
            real_idx = idx_start + i * step
            t = real_idx / points_per_sec
            x = self.scaler.time_to_pixels(t + self.model.audio_offset)

            h = amp * max_h
            if h < 1: h = 1  # 至少画一个像素

            p1 = QPointF(x, center_y - h)
            p2 = QPointF(x, center_y + h)
            lines.append(QLineF(p1, p2))

        painter.setPen(QPen(QBrush(wave_grad), 1))
        painter.drawLines(lines)

        # 4. 绘制节拍点 (Beats) — 双层: 底层 glow + 上层细线
        for t in self.model.beats:
            # 考虑 offset
            real_t = t + self.model.audio_offset
            if view_start <= real_t <= view_end:
                x = self.scaler.time_to_pixels(real_t)
                y1 = center_y - max_h - 5
                y2 = center_y + max_h + 5
                # 底层: 半透明光晕
                painter.setPen(QPen(self.beat_glow_color, 5))
                painter.drawLine(QPointF(x, y1), QPointF(x, y2))
                # 上层: 细实线
                painter.setPen(QPen(self.beat_color, 1.5))
                painter.drawLine(QPointF(x, y1), QPointF(x, y2))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # 简单的碰撞检测：点中音频条范围了吗？
            # 音频条范围: [offset, offset + duration]
            x_mouse = event.position().x()
            t_mouse = self.scaler.pixels_to_time(x_mouse)

            start_t = self.model.audio_offset
            end_t = start_t + self.model.audio_duration

            if start_t <= t_mouse <= end_t:
                self.dragging = True
                self.drag_start_x = x_mouse
                self.original_offset = self.model.audio_offset
                self.setCursor(Qt.SizeHorCursor)
            else:
                # 点击空白处，移动游标 (和 VideoTrack 一样)
                self.model.set_current_time(t_mouse)

    def mouseMoveEvent(self, event):
        if self.dragging:
            delta_x = event.position().x() - self.drag_start_x
            delta_t = self.scaler.pixels_to_duration(delta_x)

            # 更新 offset
            new_offset = self.original_offset + delta_t

            # 发出信号通知重绘
            # 注意：ProjectModel 需要一个 set_audio_offset 方法
            self.model.set_audio_offset(new_offset)
        else:
            # 悬停改变光标
            x_mouse = event.position().x()
            t_mouse = self.scaler.pixels_to_time(x_mouse)
            start_t = self.model.audio_offset
            end_t = start_t + self.model.audio_duration

            if start_t <= t_mouse <= end_t:
                self.setCursor(Qt.SizeHorCursor)
            else:
                self.setCursor(Qt.ArrowCursor)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = False
