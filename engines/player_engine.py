# 播放器逻辑（时钟驱动）
import bisect
import os

import numpy as np
from PySide6.QtCore import QObject, Signal, QTimer, QTime, Qt
from PySide6.QtGui import QImage, QPixmap
import time
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtCore import QUrl
import librosa
import sounddevice as sd
from core.data_model import ProjectModel
from core.structures import Keyframe, EffectType
from core.time_mapper import TimeMapper
from .vfx_engine import VFXEngine
from .video_loader import VideoLoader
import threading



class PlayerEngine(QObject):
    frame_ready = Signal(QImage)  # 输出图像给 UI 显示
    time_updated = Signal(float)  # 输出当前时间给时间轴
    playback_stopped = Signal()   # 播放停止

    def __init__(self, model: ProjectModel):
        super().__init__()
        self.model = model
        self.loader = VideoLoader()

        # 播放控制
        self.timer = QTimer()
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)  # PreciseTimer (尽可能精确)
        self.timer.timeout.connect(self._on_tick)

        # === 音频核心状态  ===
        self.audio_data = None  # Numpy array (frames, channels)
        self.samplerate = 44100
        self.current_frame = 0  # 播放指针
        self.is_audio_playing = False  # 控制回调是否填充数据
        self.stream = None
        self.audio_lock = threading.Lock()  # 线程锁

        self.is_playing = False
        self.last_real_time = 0.0

        # 连接 Loader meta_loaded信号
        self.loader.meta_loaded.connect(self._on_video_loaded)
        self.model.playhead_moved.connect(self.seek)

    def load_video(self, path):
        self.loader.load_video(path)
        # 注意：不在 load_video 中调用 seek(0)
        # 帧数据是异步加载的，seek 会在 on_loading_finished 中触发

    def set_audio_data(self, full_audio_data):
        """接收完整音频数据 (由 AudioLoader 在线程中加载好传过来)"""
        # 确保数据格式正确 (frames, channels)
        data = full_audio_data.astype(np.float32)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        elif data.ndim > 1 and data.shape[0] < data.shape[1]:
            # 如果是 (2, N) 这种形状，转置为 (N, 2)
            data = data.T

        with self.audio_lock:
            self.audio_data = data
            # 注意：AudioLoader 用的 librosa 默认可能重采样了，这里假设 SR=44100
            # 严谨做法是 AudioLoader 也传回 SR
            self.samplerate = 44100
            self.current_frame = 0

        # 数据就绪后，初始化 Stream (但不一定马上开始播)
        self._init_stream()

    def _init_stream(self):
        """初始化并启动 Stream，保持常驻"""
        if self.stream:
            self.stream.close()

        if self.audio_data is None:
            return

        try:
            self.stream = sd.OutputStream(
                samplerate=self.samplerate,
                channels=self.audio_data.shape[1],
                callback=self._audio_callback,
                latency='low'  # 关键参数
            )
            self.stream.start()  # 启动流，但通过 is_audio_playing 控制静音
            print("SoundDevice Stream Started (Standby)")
        except Exception as e:
            print(f"Stream Init Failed: {e}")

    def _audio_callback(self, outdata, frames, time_info, status):
        """音频回调：运行在独立线程"""
        if status:
            print(status)

        # 如果全局暂停，填充静音
        if not self.is_audio_playing or self.audio_data is None:
            outdata.fill(0)
            return

        # 即使有 Lock，在回调里也要尽量快
        # 获取当前指针
        current = self.current_frame
        data_len = len(self.audio_data)
        remaining = data_len - current

        chunk_size = min(remaining, frames)

        # 填充数据
        if chunk_size > 0:
            outdata[:chunk_size] = self.audio_data[current: current + chunk_size]

        # 更新指针 (无需锁，因为只有回调在写，seek在写，atomic assignment safe enough in Python for int)
        self.current_frame += chunk_size

        # 处理数据不足 (播放结束)
        if chunk_size < frames:
            # 策略：不循环填充，直接补零，并让主循环处理逻辑循环
            outdata[chunk_size:].fill(0)
            # 或者：如果需要无缝音频循环，在这里填充开头数据
            # frames_needed = frames - chunk_size
            # outdata[chunk_size:] = self.audio_data[:frames_needed]
            # self.current_frame = frames_needed

    def _on_video_loaded(self, duration, w, h):
        self.model.duration = duration
        self.model.fps = self.loader.fps


        # 初始化默认关键帧 (首尾 BOUNDARY)
        # 清空旧的，添加 0->0 和 duration->end_frame
        from core.structures import KeyframeSource
        self.model.keyframes.clear()
        self.model.add_keyframe(0, 0.0, source=KeyframeSource.BOUNDARY)
        self.model.add_keyframe(self.loader.total_frames - 1, duration,
                                source=KeyframeSource.BOUNDARY)

        self.model.data_loaded.emit()  # 通知 UI 刷新

    def play(self):
        if not self.is_playing:
            self.is_playing = True
            self.last_real_time = time.time()
            # 设定定时器间隔：例如 30FPS -> 33ms
            interval = int(1000 / self.loader.fps) if self.loader.fps > 0 else 33
            self.timer.start(interval)

            # 同步音频状态
            if self.audio_data is not None:
                self._sync_audio_cursor_to_time()

    def pause(self):
        self.is_playing = False
        self.timer.stop()
        self.is_audio_playing = False
        self.playback_stopped.emit()

    def toggle_play(self):
        if self.is_playing:
            self.pause()
        else:
            self.play()

    def seek(self, t_sec: float):
        """跳转到指定时间 (预览用)"""
        self.model.current_time = max(0.0, min(t_sec, self.model.duration))
        self._update_frame()
        # 调整音频指针
        if self.audio_data is not None:
            self._sync_audio_cursor_to_time()
        self.time_updated.emit(self.model.current_time)

    def _on_tick(self):
        """定时器回调"""
        if not self.is_playing:
            return

        # --- 音频状态检查 ---
        # 检查是否需要自动开启/关闭音频 (针对 Offset > 0 的情况)
        if self.audio_data is not None:
            audio_rel_sec = self.model.current_time - self.model.audio_offset

            # 如果进入了音频播放范围，且之前没开声音，则开启
            if 0 <= audio_rel_sec * self.samplerate < len(self.audio_data):
                if not self.is_audio_playing:
                    self.is_audio_playing = True
            else:
                # 超出范围，静音
                if self.is_audio_playing:
                    self.is_audio_playing = False



        # --- 音频同步 (Audio Master) (仅当音频正在输出时才以它为准) ---
        if self.is_audio_playing and self.stream and self.stream.active:
            # 直接读取音频指针算时间
            # 这是最准的，因为 callback 刚好消耗到这里
            audio_pos_sec = self.current_frame / self.samplerate
            correct_time = audio_pos_sec + self.model.audio_offset
            self.model.current_time = correct_time
        else:
            # 无音频回退模式
            now = time.time()
            dt = now - self.last_real_time
            self.model.current_time += dt

        self.last_real_time = time.time()

        # 循环播放检查

        effective_end_time = self._get_effective_duration()
        if self.model.keyframes:
            # 以最后一个关键帧的时间为准
            effective_end_time = self.model.keyframes[-1].output_time

        if self.model.current_time >= effective_end_time:
            self.model.current_time = 0.0
            # 循环播放: 重置音频状态
            if self.audio_data is not None:
                self._sync_audio_cursor_to_time()

        self._update_frame()
        self.time_updated.emit(self.model.current_time)

    def _update_frame(self):
        """核心渲染逻辑"""
        # 1. 算法计算：时间 -> 帧索引
        target_frame_idx = TimeMapper.map_time_to_frame(
            self.model.current_time, self.model.keyframes,
            fps=self.model.fps or 30.0,
            total_frames=len(self.loader.frames_cache) if self.loader.frames_cache else 0,
        )

        # 2. 读取帧数据
        frame_img = self.loader.get_frame(target_frame_idx)

        if frame_img is not None:

            dummy = Keyframe(output_time=self.model.current_time, frame_idx=0)
            kf_idx = bisect.bisect_right(self.model.keyframes, dummy) - 1

            if kf_idx >= 0:
                kf = self.model.keyframes[kf_idx]
                if kf.effect_type != EffectType.NONE:
                    # 检查时间范围
                    dt = self.model.current_time - kf.output_time
                    duration = kf.get_effect_duration()

                    if 0 <= dt <= duration:
                        # 计算进度
                        progress = dt / duration
                        # 应用动效
                        frame_img = VFXEngine.apply_effect(frame_img, kf.effect_type, progress, kf.effect_params)



            # 3. 转换为 QImage
            # 视频帧来自 frames_cache（稳定引用），无需拷贝
            # 仅当 VFX 修改过帧数据时才需要 copy
            h, w, ch = frame_img.shape
            bytes_per_line = ch * w
            if not frame_img.flags['C_CONTIGUOUS']:
                frame_img = np.ascontiguousarray(frame_img)
            q_img = QImage(frame_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
            q_img._numpy_ref = frame_img  # 绑定引用防止 GC

            # 4. 发送信号
            self.frame_ready.emit(q_img)

    def _sync_audio_cursor_to_time(self):
        """
        根据 model.current_time 和 audio_offset 计算音频指针位置，
        并决定是否应该输出音频数据。
        """
        if self.audio_data is None:
            return

        # 计算相对于音频文件开头的秒数
        audio_rel_sec = self.model.current_time - self.model.audio_offset

        # 计算目标帧索引
        target_frame = int(audio_rel_sec * self.samplerate)

        with self.audio_lock:
            if target_frame < 0:
                # 情况 A: 还没到音频开始的时间 (Offset > CurrentTime)
                # 指针设为 0，且暂时静音
                self.current_frame = 0
                self.is_audio_playing = False
            elif target_frame >= len(self.audio_data):
                # 情况 B: 音频已经播完了
                self.current_frame = len(self.audio_data)  # 指向末尾
                self.is_audio_playing = False
            else:
                # 情况 C: 正常播放范围内
                self.current_frame = target_frame
                # 只有当全局处于播放状态时，才开启音频输出
                if self.is_playing:
                    self.is_audio_playing = True

    def _get_effective_duration(self):
        """计算时间轴的总有效长度 = Max(视频结束点, 音频结束点)"""
        video_end = self.model.duration
        if self.model.keyframes:
            video_end = self.model.keyframes[-1].output_time

        audio_end = 0.0
        if self.model.audio_path:
            # 音频结束点 = offset + duration
            audio_end = self.model.audio_offset + self.model.audio_duration

        return max(video_end, audio_end)
