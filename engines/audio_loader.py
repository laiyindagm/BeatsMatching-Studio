# 多线程加载音频、提取波形
import librosa
import numpy as np
from PySide6.QtCore import QThread, Signal, QObject


class AudioWorker(QThread):
    finished = Signal(str, float, np.ndarray, list, np.ndarray)  # path, duration, waveform, beats
    progress = Signal(int)  # 新增进度信号
    error = Signal(str)

    def __init__(self, path):
        super().__init__()
        self.path = path

    def run(self):
        try:
            self.progress.emit(70)
            print(f"Loading audio: {self.path}")
            # 1. 加载音频
            y, sr = librosa.load(self.path, sr=44100)
            if y.ndim > 1:
                y_playback = y.T.astype(np.float32)
                # 分析用的数据混音成单声道
                y_mono = librosa.to_mono(y)
            else:
                y_playback = y.astype(np.float32)
                # 单声道转立体声播放需要复制，或者告诉 sounddevice 是单声道
                # 这里为了简单，如果播放时发现是一维，就在播放引擎里处理
                y_mono = y
            duration = librosa.get_duration(y=y_mono, sr=sr)


            # 2. 提取波形 (降采样用于绘图)
            # 目标：每秒 100 个点 (10ms 精度)
            pixels_per_sec = 100
            total_points = int(duration * pixels_per_sec)

            # 简单的峰值采样 (Resample 可能丢失峰值，这里用简单的切片取最大值)
            # 将数组切分成 chunk，取每个 chunk 的最大绝对值
            if total_points > 0:
                # 简单重采样：取绝对值后切片
                y_abs = np.abs(y)
                # 计算步长
                step = len(y_abs) // total_points
                if step < 1: step = 1

                # 关键修复：确保 waveform 长度与 total_points 一致
                waveform = y_abs[::step]

                # 归一化 (防爆音)
                m = waveform.max()
                if m > 0:
                    waveform = waveform / m

                # 截断或填充
                if len(waveform) > total_points:
                    waveform = waveform[:total_points]
            else:
                waveform = np.array([])



            # 3. 提取节拍 (Beats)
            tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
            beat_times = librosa.frames_to_time(beat_frames, sr=sr)

            print("Audio analysis finished.")
            self.finished.emit(self.path, duration, waveform, beat_times.tolist(), y_playback)

        except Exception as e:
            self.error.emit(str(e))


class AudioLoader(QObject):
    # 转发 Worker 的信号
    audio_loaded = Signal(float, np.ndarray, list, np.ndarray)  # duration, waveform, beats
    loading_progress = Signal(int) # 新增

    def __init__(self):
        super().__init__()
        self.worker = None
        self.path = None

    def load_audio(self, path):
        self.path = path
        if self.worker and self.worker.isRunning():
            self.worker.terminate()  # 强制停止旧任务

        self.worker = AudioWorker(path)
        self.worker.progress.connect(self.loading_progress)  # 连接进度
        self.worker.finished.connect(self._on_finished)
        self.worker.start()

    def _on_finished(self, path, duration, waveform, beats, full_audio):
        self.audio_loaded.emit(duration, waveform, beats, full_audio)
