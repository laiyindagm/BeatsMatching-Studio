# 坐标转换器 (Time <-> Pixels)
from PySide6.QtCore import QObject, Signal


class TimeScaler(QObject):
    """
    负责将时间(秒)转换为屏幕坐标(像素)的工具类。
    所有时间轴组件(TimeRuler, VideoTrack, AudioTrack)共享同一个 TimeScaler 实例。
    """

    # 当缩放级别或滚动位置发生改变时发出信号，通知 UI 重绘
    view_changed = Signal()

    def __init__(self):
        super().__init__()

        # 核心属性
        self._pixels_per_second: float = 100.0  # 缩放级别 (1秒 = 100像素)
        self._view_offset_x: float = 0.0  # 水平滚动条偏移量 (像素)

        # 限制
        self.min_zoom = 10.0  # 最小缩放 (1秒=10像素，概览模式)
        self.max_zoom = 2000.0  # 最大缩放 (1秒=2000像素，精细剪辑)

        # 视口宽度 (用于计算可见区域，由外部容器更新)
        self.viewport_width: int = 1000

    @property
    def pixels_per_second(self) -> float:
        return self._pixels_per_second

    @property
    def view_offset_x(self) -> float:
        return self._view_offset_x

    def time_to_pixels(self, time_sec: float) -> float:
        """
        核心转换：时间 -> 控件内的局部 x 坐标
        用于：绘制关键帧、刻度线、波形
        """
        return (time_sec * self._pixels_per_second) - self._view_offset_x

    def pixels_to_time(self, x_px: float) -> float:
        """
        核心转换：控件内的局部 x 坐标 -> 时间
        用于：处理鼠标点击、拖拽
        """
        return (x_px + self._view_offset_x) / self._pixels_per_second

    def duration_to_pixels(self, duration_sec: float) -> float:
        """长度转换：将一段时间长度转换为像素长度"""
        return duration_sec * self._pixels_per_second

    def pixels_to_duration(self, length_px: float) -> float:
        """长度转换：将像素长度转换为时间长度"""
        return length_px / self._pixels_per_second

    def scroll_to(self, x: float):
        """设置滚动条位置 (直接设置像素偏移)"""
        if self._view_offset_x != x:
            self._view_offset_x = max(0.0, x)  # 不允许滚动到负时间
            self.view_changed.emit()

    def scroll_by_pixels(self, dx: float):
        """相对滚动"""
        self.scroll_to(self._view_offset_x + dx)

    def set_zoom(self, new_pps: float, mouse_x: float = 0):
        """
        设置缩放级别。

        :param new_pps: 新的 pixels_per_second
        :param mouse_x: 缩放中心点的 x 坐标 (通常是鼠标位置)
                        这保证了用户鼠标指向的时间点在缩放后位置不变。
        """
        # 1. 限制范围
        target_pps = max(self.min_zoom, min(new_pps, self.max_zoom))

        if target_pps == self._pixels_per_second:
            return

        # 2. 计算鼠标当前指向的时间点 (这个时间点应保持不变)
        hover_time = self.pixels_to_time(mouse_x)

        # 3. 更新缩放比例
        self._pixels_per_second = target_pps

        # 4. 反向计算新的 offset，使得 hover_time 仍然在 mouse_x 处
        # mouse_x = hover_time * new_pps - new_offset
        # => new_offset = hover_time * new_pps - mouse_x
        new_offset = hover_time * self._pixels_per_second - mouse_x

        self._view_offset_x = max(0.0, new_offset)

        self.view_changed.emit()

    def get_visible_range(self) -> tuple[float, float]:
        """返回当前视口可见的时间范围 (start_time, end_time)"""
        start = self.pixels_to_time(0)
        end = self.pixels_to_time(self.viewport_width)
        return start, end
