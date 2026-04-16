"""
gui/pose_visualization_panel.py
关节位姿可视化面板
- 上半区域：ViTPose 2D 骨架可视化
- 下半区域：SMPL 3D 人体模型渲染
- 支持按帧同步（跟随时间轴游标）
- 支持切换原始/平滑数据
- 支持用户配置相机角度
"""
from __future__ import annotations

import numpy as np
from collections import deque
from functools import lru_cache
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSplitter,
    QComboBox, QDoubleSpinBox, QGroupBox, QSizePolicy,
)
from PySide6.QtCore import Qt, Signal, QThread, QMutex, QMutexLocker, QTimer
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont

from gui.styles import DesignTokens


# ── COCO-17 骨架定义 ──
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),       # 头部
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # 上肢
    (5, 11), (6, 12), (11, 12),            # 躯干
    (11, 13), (13, 15), (12, 14), (14, 16),  # 下肢
]

COCO_JOINT_COLORS = [
    (255, 200, 87), (255, 200, 87), (255, 200, 87),  # 0-2 鼻眼
    (255, 200, 87), (255, 200, 87),                    # 3-4 耳
    (87, 199, 255), (87, 199, 255),                    # 5-6 肩
    (87, 199, 255), (87, 199, 255),                    # 7-8 肘
    (255, 135, 87), (255, 135, 87),                    # 9-10 腕
    (150, 255, 150), (150, 255, 150),                  # 11-12 髋
    (150, 150, 255), (150, 150, 255),                  # 13-14 膝
    (255, 87, 199), (255, 87, 199),                    # 15-16 踝
]


def render_vitpose_frame_qimage(
    kp: np.ndarray,
    canvas_w: int = 300,
    canvas_h: int = 380,
    all_coords: np.ndarray | None = None,
) -> QImage:
    """
    绘制单帧 ViTPose 骨架图为 QImage

    Args:
        kp: (17, 2) 当前帧关节坐标
        canvas_w, canvas_h: 画布尺寸
        all_coords: (T, 17, 2) 全部帧坐标，用于计算全局缩放范围
    """
    img = QImage(canvas_w, canvas_h, QImage.Format.Format_RGB32)
    img.fill(QColor(22, 27, 38))  # BG_SURFACE

    painter = QPainter(img)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    # 计算缩放
    if all_coords is not None:
        x_min, x_max = np.nanmin(all_coords[:, :, 0]), np.nanmax(all_coords[:, :, 0])
        y_min, y_max = np.nanmin(all_coords[:, :, 1]), np.nanmax(all_coords[:, :, 1])
    else:
        x_min, x_max = np.nanmin(kp[:, 0]), np.nanmax(kp[:, 0])
        y_min, y_max = np.nanmin(kp[:, 1]), np.nanmax(kp[:, 1])

    pad = 20
    x_range = max(x_max - x_min, 1)
    y_range = max(y_max - y_min, 1)
    scale = min((canvas_w - 2 * pad) / x_range, (canvas_h - 2 * pad) / y_range)
    ox = (canvas_w - x_range * scale) / 2 - x_min * scale
    oy = (canvas_h - y_range * scale) / 2 - y_min * scale

    def to_canvas(pt):
        return int(pt[0] * scale + ox), int(pt[1] * scale + oy)

    # 绘制骨骼连线
    pen = QPen(QColor(100, 160, 255), 3)
    painter.setPen(pen)
    for a, b in COCO_SKELETON:
        pt1 = to_canvas(kp[a])
        pt2 = to_canvas(kp[b])
        painter.drawLine(pt1[0], pt1[1], pt2[0], pt2[1])

    # 绘制关节点
    painter.setPen(Qt.PenStyle.NoPen)
    for j in range(17):
        cx, cy = to_canvas(kp[j])
        r = 6 if j in (9, 10, 15, 16) else 5
        color = COCO_JOINT_COLORS[j]
        painter.setBrush(QColor(*color))
        painter.drawEllipse(cx - r, cy - r, 2 * r, 2 * r)

    painter.end()
    return img


class _SMPLRenderThread(QThread):
    """后台 SMPL 渲染线程（优先队列 + 预渲染 + LRU 图像缓存）"""
    rendered = Signal(object, int)  # (QImage, frame_idx)

    def __init__(self):
        super().__init__()
        self._mutex = QMutex()
        self._queue = deque()       # 渲染队列：(pose, size, az, el, flip, stab, frame_idx)
        self._priority = None       # 高优先级请求（用户主动跳转）
        self._rendered_keys = set() # 已渲染过的 frame_idx（避免重复预渲染）
        self._renderer = None
        self._running = True
        
        # LRU 图像缓存：frame_idx → QImage（已渲染帧直接返回，避免重复渲染）
        from collections import OrderedDict
        self._image_cache: OrderedDict = OrderedDict()
        self._cache_size = 128  # 最多缓存 128 帧
        self._cache_az = None   # 缓存对应的视角参数
        self._cache_el = None
        
        # 渲染统计
        self._render_count = 0
        self._cache_hits = 0

    def request_render(self, pose, frame_idx, size=(300, 380),
                       azimuth=270.0, elevation=0.0,
                       flip_y=False, stabilize_root=True, priority=True):
        """提交渲染请求。priority=True 时插入队列头部（用户请求）"""
        req = (pose, size, azimuth, elevation, flip_y, stabilize_root, frame_idx)
        with QMutexLocker(self._mutex):
            if priority:
                self._priority = req  # 高优先级覆盖
            else:
                self._queue.append(req)
        if not self.isRunning():
            self.start()

    def enqueue_prefetch(self, requests: list):
        """批量提交预渲染请求（低优先级）"""
        with QMutexLocker(self._mutex):
            for req in requests:
                self._queue.append(req)
        if not self.isRunning():
            self.start()

    def clear_queue(self):
        """清空预渲染队列"""
        with QMutexLocker(self._mutex):
            self._queue.clear()
            self._rendered_keys.clear()

    def invalidate_cache(self):
        """视角/参数变化时清除图像缓存"""
        with QMutexLocker(self._mutex):
            self._image_cache.clear()
            self._cache_az = None
            self._cache_el = None

    def stop(self):
        self._running = False
        self.wait(2000)

    def run(self):
        while self._running:
            req = None
            with QMutexLocker(self._mutex):
                # 高优先级优先
                if self._priority is not None:
                    req = self._priority
                    self._priority = None
                    # 拿到高优先级请求时清空旧的预渲染队列
                    self._queue.clear()
                elif self._queue:
                    req = self._queue.popleft()

            if req is None:
                self.msleep(15)
                continue

            pose, size, azimuth, elevation, flip_y, stabilize, frame_idx = req

            try:
                # ── 检查 LRU 图像缓存 ──
                cache_valid = (self._cache_az == azimuth and self._cache_el == elevation)
                if cache_valid and frame_idx in self._image_cache:
                    self._cache_hits += 1
                    qimg = self._image_cache[frame_idx]
                    # LRU 移到末尾
                    self._image_cache.move_to_end(frame_idx)
                    self.rendered.emit(qimg, frame_idx)
                    continue
                
                # 视角变化时清除缓存
                if not cache_valid:
                    self._image_cache.clear()
                    self._cache_az = azimuth
                    self._cache_el = elevation

                if self._renderer is None:
                    self._renderer = self._load_renderer()
                if self._renderer is None:
                    continue

                img_np = self._renderer.render_frame(
                    pose, size=size,
                    azimuth=azimuth, elevation=elevation,
                    flip_y=flip_y, stabilize_root=stabilize,
                )
                h, w, ch = img_np.shape
                qimg = QImage(img_np.data, w, h, w * ch, QImage.Format.Format_RGB888).copy()
                
                # 存入 LRU 缓存
                self._image_cache[frame_idx] = qimg
                if len(self._image_cache) > self._cache_size:
                    self._image_cache.popitem(last=False)
                
                self._render_count += 1
                with QMutexLocker(self._mutex):
                    self._rendered_keys.add(frame_idx)
                self.rendered.emit(qimg, frame_idx)
                
                # 定期输出统计
                if self._render_count % 50 == 0:
                    total = self._render_count + self._cache_hits
                    rate = self._cache_hits / total * 100 if total > 0 else 0
                    print(f"[SMPLRender] 已渲染 {self._render_count} 帧, "
                          f"缓存命中 {self._cache_hits}/{total} ({rate:.0f}%)")
            except Exception as e:
                print(f"[SMPLRender] Error: {e}")

    def _load_renderer(self):
        try:
            import os, sys
            from engines.smpl_renderer import SMPLRenderer
            from pose_extraction.weights_config import SMPL_NEUTRAL
            if os.path.exists(SMPL_NEUTRAL):
                return SMPLRenderer(SMPL_NEUTRAL)
            else:
                print(f"[SMPLRender] SMPL weights not found: {SMPL_NEUTRAL}")
                return None
        except Exception as e:
            print(f"[SMPLRender] Failed to load renderer: {e}")
            return None


class PoseVisualizationPanel(QWidget):
    """
    关节位姿可视化面板

    上半：ViTPose 2D 骨架
    下半：SMPL 3D 模型渲染
    """

    def __init__(self, model, parent=None):
        super().__init__(parent)
        self.model = model
        self._last_frame_idx = -1
        self._smpl_cache = {}  # cache_key → QPixmap
        self._smpl_cache_max = 500
        self._smpl_render_thread = _SMPLRenderThread()
        self._smpl_render_thread.rendered.connect(self._on_smpl_rendered)
        self._prefetch_count = 30
        self._is_playing = False  # 播放状态标志
        
        # 拖动节流 (不超过 20fps 更新位姿面板)
        self._drag_timer = QTimer(self)
        self._drag_timer.setSingleShot(True)
        self._drag_timer.setInterval(50)  # 50ms = 20fps
        self._drag_timer.timeout.connect(self._do_deferred_update)
        self._deferred_time = None

        self._build_ui()
        self._connect_signals()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # ── 控制栏 ──
        ctrl = QHBoxLayout()
        ctrl.setSpacing(8)

        # 数据模式
        lbl_mode = QLabel("模式:")
        lbl_mode.setStyleSheet(f"color: {DesignTokens.TEXT_SECONDARY}; font-size: 11px;")
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["平滑后", "原始检测"])
        self.combo_mode.setFixedWidth(90)
        self.combo_mode.setStyleSheet(f"""
            QComboBox {{
                background: {DesignTokens.BG_ELEVATED};
                color: {DesignTokens.TEXT_PRIMARY};
                border: 1px solid {DesignTokens.BORDER_DEFAULT};
                border-radius: 4px;
                padding: 2px 6px;
                font-size: 11px;
            }}
        """)
        ctrl.addWidget(lbl_mode)
        ctrl.addWidget(self.combo_mode)

        # 相机角度
        lbl_az = QLabel("方位:")
        lbl_az.setStyleSheet(f"color: {DesignTokens.TEXT_SECONDARY}; font-size: 11px;")
        self.spin_azimuth = QDoubleSpinBox()
        self.spin_azimuth.setRange(0, 360)
        self.spin_azimuth.setValue(270.0)
        self.spin_azimuth.setSingleStep(15.0)
        self.spin_azimuth.setFixedWidth(65)
        self.spin_azimuth.setStyleSheet(self._spin_style())

        lbl_el = QLabel("仰角:")
        lbl_el.setStyleSheet(f"color: {DesignTokens.TEXT_SECONDARY}; font-size: 11px;")
        self.spin_elevation = QDoubleSpinBox()
        self.spin_elevation.setRange(-90, 90)
        self.spin_elevation.setValue(0.0)
        self.spin_elevation.setSingleStep(5.0)
        self.spin_elevation.setFixedWidth(65)
        self.spin_elevation.setStyleSheet(self._spin_style())

        ctrl.addWidget(lbl_az)
        ctrl.addWidget(self.spin_azimuth)
        ctrl.addWidget(lbl_el)
        ctrl.addWidget(self.spin_elevation)
        ctrl.addStretch()

        layout.addLayout(ctrl)

        # ── 上下分割：骨架 + SMPL ──
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setHandleWidth(3)

        # 上：ViTPose
        self.vitpose_group = QGroupBox("ViTPose 骨架")
        self.vitpose_group.setStyleSheet(self._group_style())
        vit_layout = QVBoxLayout(self.vitpose_group)
        vit_layout.setContentsMargins(4, 16, 4, 4)
        self.vitpose_label = QLabel("等待姿态数据...")
        self.vitpose_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.vitpose_label.setStyleSheet(f"""
            color: {DesignTokens.TEXT_DISABLED};
            font-size: 12px;
            background: {DesignTokens.BG_SURFACE};
            border-radius: 4px;
        """)
        self.vitpose_label.setMinimumHeight(120)
        vit_layout.addWidget(self.vitpose_label)

        # 下：SMPL
        self.smpl_group = QGroupBox("SMPL 模型")
        self.smpl_group.setStyleSheet(self._group_style())
        smpl_layout = QVBoxLayout(self.smpl_group)
        smpl_layout.setContentsMargins(4, 16, 4, 4)
        self.smpl_label = QLabel("等待姿态数据...")
        self.smpl_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.smpl_label.setStyleSheet(f"""
            color: {DesignTokens.TEXT_DISABLED};
            font-size: 12px;
            background: {DesignTokens.BG_SURFACE};
            border-radius: 4px;
        """)
        self.smpl_label.setMinimumHeight(120)
        smpl_layout.addWidget(self.smpl_label)

        splitter.addWidget(self.vitpose_group)
        splitter.addWidget(self.smpl_group)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setStyleSheet(f"""
            QSplitter::handle {{ background: {DesignTokens.BORDER_SUBTLE}; }}
            QSplitter::handle:hover {{ background: {DesignTokens.PRIMARY}; }}
        """)

        layout.addWidget(splitter, 1)

        self.setStyleSheet(f"""
            PoseVisualizationPanel {{
                background: {DesignTokens.BG_BASE};
            }}
        """)

    def _connect_signals(self):
        self.model.playhead_moved.connect(self._on_playhead_moved)
        self.model.pose_data_loaded.connect(self._on_pose_data_loaded)
        self.combo_mode.currentIndexChanged.connect(self._on_mode_changed)
        self.spin_azimuth.valueChanged.connect(self._on_camera_changed)
        self.spin_elevation.valueChanged.connect(self._on_camera_changed)

    def _on_pose_data_loaded(self):
        """姿态数据就绪 → 清缓存 + 预渲染"""
        self._smpl_cache.clear()
        self._smpl_render_thread.clear_queue()
        self._smpl_render_thread.invalidate_cache()  # 新数据，清除渲染缓存
        self._last_frame_idx = -1
        self._update_frame(self.model.current_time)
        self._start_prefetch(max(0, int(round(self.model.current_time * (self.model.fps or 30.0)))))

    def _on_playhead_moved(self, t: float):
        """拖动游标时节流更新（50ms 间隔 = 20fps）"""
        if not self.isVisible() or self._is_playing:
            return
        self._deferred_time = t
        if not self._drag_timer.isActive():
            self._drag_timer.start()

    def _do_deferred_update(self):
        """节流定时器到期，执行延迟的帧更新"""
        if self._deferred_time is not None:
            self._update_frame(self._deferred_time)
            self._deferred_time = None

    def update_from_playback(self, t: float):
        """播放引擎信号驱动 — 播放中不更新位姿面板
        
        位姿渲染（ViTPose QPainter + SMPL pyrender）耗时较长，
        在主线程中执行会导致播放卡顿，且无法通过异步方案彻底消除。
        因此播放时仅标记状态，停止后再全质量渲染当前帧。
        详见 docs/pose_panel_playback_lag.txt
        """
        self._is_playing = True

    def notify_playback_stopped(self):
        """播放停止时调用，恢复拖动模式，立即全质量渲染当前帧"""
        self._is_playing = False
        self._last_frame_idx = -1  # 强制刷新
        self._update_frame(self.model.current_time)

    def _on_mode_changed(self):
        self._smpl_cache.clear()
        self._last_frame_idx = -1
        self._update_frame(self.model.current_time)

    def _on_camera_changed(self):
        self._smpl_cache.clear()
        self._last_frame_idx = -1
        self._smpl_render_thread.invalidate_cache()  # 视角变化，清除渲染缓存
        self._update_frame(self.model.current_time)

    def _update_frame(self, t: float):
        """根据当前时间更新两个渲染区域（拖动/停止时调用）"""
        fps = self.model.fps or 30.0
        frame_idx = max(0, int(round(t * fps)))

        if frame_idx == self._last_frame_idx:
            return
        self._last_frame_idx = frame_idx

        use_raw = (self.combo_mode.currentIndex() == 1)

        # ── ViTPose 骨架 ──
        coords = self.model.pose_coords_raw if use_raw else self.model.pose_coords
        if coords is not None and frame_idx < len(coords):
            all_c = self.model.pose_coords_raw if self.model.pose_coords_raw is not None else coords
            w = self.vitpose_label.width() or 300
            h = self.vitpose_label.height() or 380
            qimg = render_vitpose_frame_qimage(coords[frame_idx], w, h, all_c)
            self.vitpose_label.setPixmap(QPixmap.fromImage(qimg))
        elif coords is None:
            self.vitpose_label.setText("无 ViTPose 数据")
        else:
            self.vitpose_label.setText(f"帧 {frame_idx} 超出范围")

        # ── SMPL 模型 ──
        rotations = self.model.pose_rotations_raw if use_raw else self.model.pose_rotations
        if rotations is not None and frame_idx < len(rotations):
            cache_key = (frame_idx, self.combo_mode.currentIndex(),
                         self.spin_azimuth.value(), self.spin_elevation.value())
            if cache_key in self._smpl_cache:
                self.smpl_label.setPixmap(self._smpl_cache[cache_key])
            else:
                w = self.smpl_label.width() or 300
                h = self.smpl_label.height() or 380
                self._smpl_render_thread.request_render(
                    rotations[frame_idx], frame_idx,
                    size=(w, h),
                    azimuth=self.spin_azimuth.value(),
                    elevation=self.spin_elevation.value(),
                    priority=True,
                )
                nearest_pm = self._find_nearest_cached(frame_idx)
                if nearest_pm is not None:
                    self.smpl_label.setPixmap(nearest_pm)
                else:
                    self.smpl_label.setText(f"渲染中... 帧{frame_idx}")
                self._start_prefetch(frame_idx)
        elif rotations is None:
            self.smpl_label.setText("无 SMPL 数据")
        else:
            self.smpl_label.setText(f"帧 {frame_idx} 超出范围")

    def _on_smpl_rendered(self, qimg: QImage, frame_idx: int):
        """SMPL 渲染完成回调"""
        pm = QPixmap.fromImage(qimg)
        cache_key = (frame_idx, self.combo_mode.currentIndex(),
                     self.spin_azimuth.value(), self.spin_elevation.value())

        # LRU 淘汰
        if len(self._smpl_cache) >= self._smpl_cache_max:
            oldest = next(iter(self._smpl_cache))
            del self._smpl_cache[oldest]
        self._smpl_cache[cache_key] = pm

        # 仅当仍在该帧时更新显示
        if frame_idx == self._last_frame_idx:
            self.smpl_label.setPixmap(pm)

    def closeEvent(self, event):
        self._smpl_render_thread.stop()
        super().closeEvent(event)

    def resizeEvent(self, event):
        """面板大小变化 — 播放中跳过，空闲时仅重渲染当前帧"""
        super().resizeEvent(event)
        if not self.isVisible() or self._last_frame_idx < 0:
            return
        if self._is_playing:
            return  # 播放中不要触发同步重渲染
        # 尺寸变化导致缓存的 SMPL 尺寸不匹配，清除并重渲染当前帧
        self._smpl_cache.clear()
        self._smpl_render_thread.clear_queue()
        self._last_frame_idx = -1
        self._update_frame(self.model.current_time)

    def _find_nearest_cached(self, frame_idx: int):
        """查找最近已缓存的帧 QPixmap（向前搜索优先）"""
        mode_idx = self.combo_mode.currentIndex()
        az = self.spin_azimuth.value()
        el = self.spin_elevation.value()
        # 向前搜索最多 30 帧
        for delta in range(1, 31):
            key = (frame_idx - delta, mode_idx, az, el)
            if key in self._smpl_cache:
                return self._smpl_cache[key]
        return None

    def _start_prefetch(self, start_frame: int):
        """从 start_frame 开始向后预渲染 N 帧"""
        use_raw = (self.combo_mode.currentIndex() == 1)
        rotations = self.model.pose_rotations_raw if use_raw else self.model.pose_rotations
        if rotations is None:
            return
        
        mode_idx = self.combo_mode.currentIndex()
        az = self.spin_azimuth.value()
        el = self.spin_elevation.value()
        w = self.smpl_label.width() or 300
        h = self.smpl_label.height() or 380
        
        requests = []
        for i in range(1, self._prefetch_count + 1):
            fid = start_frame + i
            if fid >= len(rotations):
                break
            cache_key = (fid, mode_idx, az, el)
            if cache_key in self._smpl_cache:
                continue  # 已缓存
            requests.append((rotations[fid], (w, h), az, el, False, True, fid))
        
        if requests:
            self._smpl_render_thread.enqueue_prefetch(requests)

    # ── 样式辅助 ──
    @staticmethod
    def _spin_style():
        return f"""
            QDoubleSpinBox {{
                background: {DesignTokens.BG_ELEVATED};
                color: {DesignTokens.TEXT_PRIMARY};
                border: 1px solid {DesignTokens.BORDER_DEFAULT};
                border-radius: 4px;
                padding: 2px 4px;
                font-size: 11px;
            }}
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
                width: 14px;
                border: none;
                background: {DesignTokens.BG_OVERLAY};
            }}
        """

    @staticmethod
    def _group_style():
        return f"""
            QGroupBox {{
                font-size: 11px;
                font-weight: bold;
                color: {DesignTokens.ACCENT_BLUE};
                border: 1px solid {DesignTokens.BORDER_SUBTLE};
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 4px;
                background: {DesignTokens.BG_BASE};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }}
        """
