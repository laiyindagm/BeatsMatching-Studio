# 多线程加载视频帧

import cv2
import threading
from PySide6.QtCore import QObject, Signal, QThread, QMutex
import numpy as np


class FrameLoadingWorker(QThread):
    """后台加载线程"""
    progress = Signal(int)  # 进度信号 (0-100)
    finished = Signal()     # 完成信号（不传数据，通过属性获取）
    error = Signal(str)

    def __init__(self, path):
        super().__init__()
        self.path = path
        self._is_cancelled = False
        self.frames = []  # 加载结果存储在属性中，避免通过信号传递大对象

    def run(self):
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            self.error.emit(f"Failed to open video: {self.path}")
            return

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frames = []

        print(f"Start loading {total} frames from {self.path}...")

        for i in range(total):
            if self._is_cancelled:
                break

            ret, frame = cap.read()
            if not ret:
                break

            # 格式转换 BGR -> RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.frames.append(frame_rgb)

            # 每加载 1% 或每 10 帧发送一次进度，避免信号拥塞
            if i % 10 == 0 or i == total - 1:
                percent = int((i + 1) / total * 100)
                self.progress.emit(percent)

        cap.release()

        if not self._is_cancelled:
            self.finished.emit()
            print(f"Loading finished. {len(self.frames)} frames cached.")

    def cancel(self):
        self._is_cancelled = True


class VideoLoader(QObject):
    """
    视频加载管理器 (运行在主线程)
    """
    # 信号定义
    meta_loaded = Signal(float, int, int)  # duration, w, h
    loading_progress = Signal(int)  # 0-100
    loading_finished = Signal()  # 全部加载完成
    loading_error = Signal(str)  # 加载错误

    def __init__(self):
        super().__init__()
        self.fps = 30.0
        self.total_frames = 0
        self.frames_cache = []  # 全量帧缓存 [Frame 0, Frame 1, ...]
        self._worker = None
        self.video_path = None
        self.width = -1
        self.height = -1
        self._is_loading = False

    @property
    def is_loading(self):
        return self._is_loading

    def load_video(self, path: str):
        """启动异步加载"""
        # 安全停止旧的加载
        self._cancel_worker()

        # 1. 先同步读取元数据 (这一步很快)
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            self.loading_error.emit(f"无法打开视频: {path}")
            return

        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = self.total_frames / self.fps if self.fps > 0 else 0
        cap.release()

        # 通知 UI 元数据已好
        self.meta_loaded.emit(duration, w, h)
        self.width = w
        self.height = h

        # 清空旧缓存
        self.frames_cache = []
        self._is_loading = True

        # 2. 启动后台线程加载全量帧
        self.video_path = path
        self._worker = FrameLoadingWorker(path)
        self._worker.progress.connect(self.loading_progress)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.error.connect(self._on_worker_error)
        # 确保线程结束后自动清理
        self._worker.finished.connect(self._cleanup_worker)
        self._worker.error.connect(self._cleanup_worker)
        self._worker.start()

    def _on_worker_finished(self):
        if self._worker is not None:
            self.frames_cache = self._worker.frames
            self.total_frames = len(self.frames_cache)
        self._is_loading = False
        self.loading_finished.emit()

    def _on_worker_error(self, err_msg):
        self._is_loading = False
        self.loading_error.emit(err_msg)

    def _cleanup_worker(self):
        """安全清理 worker（在信号处理完毕后调用）"""
        if self._worker is not None:
            try:
                self._worker.disconnect(self)
            except (RuntimeError, TypeError):
                pass
            self._worker = None

    def _cancel_worker(self):
        """安全取消并等待 worker"""
        if self._worker is not None:
            if self._worker.isRunning():
                self._worker._is_cancelled = True
                # 不使用 wait()，因为它会阻塞主线程
                # 使用 quit + 等待事件循环处理
                self._worker.quit()
            self._worker = None
        self._is_loading = False

    def get_frame(self, frame_idx: float) -> np.ndarray:
        """
        获取帧：直接从内存拿
        """
        if not self.frames_cache:
            return None

        idx = int(frame_idx)
        idx = max(0, min(idx, len(self.frames_cache) - 1))
        return self.frames_cache[idx]
