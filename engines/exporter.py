# engines/exporter.py

import cv2
import numpy as np
import os
from PySide6.QtCore import QThread, Signal

from core.data_model import ProjectModel
from core.time_mapper import TimeMapper
from engines.video_loader import VideoLoader
from engines.vfx_engine import VFXEngine

# 引入 moviepy 用于音频合成
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip


class ExportWorker(QThread):
    progress = Signal(int)
    finished = Signal(str)  # success message
    error = Signal(str)

    def __init__(self, model: ProjectModel, output_path: str, loader: VideoLoader):
        super().__init__()
        self.model = model
        self.output_path = output_path
        self.loader = loader
        self.is_cancelled = False

    def run(self):
        try:
            # 1. 准备参数
            temp_video_path = self.output_path.replace(".mp4", "_temp_video.mp4")

            import copy
            src_fps = self.model.fps or 30.0
            model_duration = self.model.duration
            audio_path = self.model.audio_path
            audio_offset = self.model.audio_offset
            frames_cache = list(self.loader.frames_cache) if self.loader.frames_cache else []
            src_total = len(frames_cache)
            width = self.loader.width
            height = self.loader.height

            def get_cached_frame(frame_idx):
                if not frames_cache:
                    return None
                idx = int(frame_idx)
                idx = max(0, min(idx, len(frames_cache) - 1))
                return frames_cache[idx]

            # ── 构建导出关键帧（加入首尾虚拟锚点，使首尾区间原速） ──
            export_keyframes = copy.deepcopy(self.model.keyframes)

            if export_keyframes:
                first_kf = export_keyframes[0]
                last_kf = export_keyframes[-1]

                # 首帧虚拟锚点：frame 0 在 output_time = first_kf.output_time - first_kf.frame_idx/src_fps
                head_out = first_kf.output_time - first_kf.frame_idx / src_fps
                # 尾帧虚拟锚点：last_frame 在 output_time = last_kf.output_time + (last_frame - last_kf.frame_idx)/src_fps
                last_frame_idx = src_total - 1 if src_total > 0 else last_kf.frame_idx
                tail_out = last_kf.output_time + (last_frame_idx - last_kf.frame_idx) / src_fps

                # 插入首帧虚拟锚点（如果首 KF 不是 frame 0）
                if first_kf.frame_idx > 0:
                    from core.structures import Keyframe, KeyframeSource
                    kf0 = Keyframe(output_time=head_out, frame_idx=0)
                    kf0.source = KeyframeSource.BOUNDARY
                    export_keyframes.insert(0, kf0)

                # 插入尾帧虚拟锚点（如果末 KF 不是最后一帧）
                if last_kf.frame_idx < last_frame_idx:
                    from core.structures import Keyframe, KeyframeSource
                    kf_end = Keyframe(output_time=tail_out, frame_idx=last_frame_idx)
                    kf_end.source = KeyframeSource.BOUNDARY
                    export_keyframes.append(kf_end)
            else:
                head_out = 0.0
                tail_out = model_duration

            # ── 平移使视频从 0s 开始 ──
            video_start = export_keyframes[0].output_time if export_keyframes else 0.0
            if video_start < 0:
                # 需要右移所有 output_time
                shift = -video_start
                for kf in export_keyframes:
                    kf.output_time += shift
                print(f"[Export] 首帧时间 < 0, 整体右移 {shift:.3f}s")
                video_start = 0.0

            if video_start > 1e-6:
                for kf in export_keyframes:
                    kf.output_time -= video_start
                print(f"[Export] 首帧偏移修正: 所有 output_time 左移 {video_start:.3f}s")

            # 音频偏移相应调整
            audio_offset_export = audio_offset - video_start

            # 计算有效时长。导出视频的时长应由时间映射后的视频末尾决定；
            # 音频只作为配乐叠加，超过视频范围的部分必须裁掉。
            video_end = export_keyframes[-1].output_time if export_keyframes else model_duration
            total_duration = max(video_end, 0)
            if total_duration <= 0:
                raise Exception("Invalid export duration")

            fps = 30.0  # 输出帧率
            total_frames = int(total_duration * fps)

            # 获取源视频尺寸 (或者自定义输出尺寸)
            # 假设输出 1080p 或者源尺寸
            if width == 0: width = 1920
            if height == 0: height = 1080

            print(f"Start Exporting: {total_frames} frames, {width}x{height}")

            # 2. 初始化 VideoWriter
            # mp4v or avc1 (h264)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

            if not writer.isOpened():
                raise Exception("Failed to create video writer")

            # 3. 逐帧渲染循环
            for i in range(total_frames):
                if self.is_cancelled: break

                t = i / fps

                # A. 映射时间
                src_frame_idx = TimeMapper.map_time_to_frame(
                    t, export_keyframes, fps=src_fps, total_frames=src_total)

                # B. 获取源帧
                frame = get_cached_frame(src_frame_idx)

                if frame is None:
                    # 如果取不到帧 (比如结束了)，生成黑帧
                    frame = np.zeros((height, width, 3), dtype=np.uint8)
                else:
                    # 确保尺寸匹配 (如果源视频尺寸不一，这里需要 resize)
                    if frame.shape[1] != width or frame.shape[0] != height:
                        frame = cv2.resize(frame, (width, height))

                # C. 应用动效 (VFX)
                # 查找是否受动效影响
                # 简单遍历优化：只看 t 附近的关键帧
                # 这里为了简单直接复用逻辑，实际上可以优化查找
                import bisect
                from core.structures import Keyframe, EffectType
                dummy = Keyframe(output_time=t, frame_idx=0)
                kf_idx = bisect.bisect_right(export_keyframes, dummy) - 1

                if kf_idx >= 0:
                    kf = export_keyframes[kf_idx]
                    if kf.effect_type != EffectType.NONE:
                        dt = t - kf.output_time
                        dur = kf.get_effect_duration()
                        if 0 <= dt <= dur:
                            # 必须 copy 一份，因为 frame 是缓存里的引用，不能改原图
                            frame = frame.copy()
                            frame = VFXEngine.apply_effect(frame, kf.effect_type, dt / dur, kf.effect_params)

                # D. 颜色空间转换 RGB -> BGR (OpenCV Writer 需要 BGR)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                writer.write(frame_bgr)

                # 进度汇报
                if i % 10 == 0:
                    self.progress.emit(int(i / total_frames * 90))

            writer.release()

            if self.is_cancelled:
                if os.path.exists(temp_video_path): os.remove(temp_video_path)
                return

            self.progress.emit(95)

            # 4. 音频合成 (MoviePy)
            print("Muxing audio...")
            final_clip = VideoFileClip(temp_video_path)

            if audio_path and os.path.exists(audio_path):
                # 读取音频
                audio_source = AudioFileClip(audio_path)

                clips_to_composite = []

                video_duration = final_clip.duration

                # 处理 offset（使用导出修正后的偏移），并显式裁剪到视频时长内。
                if audio_offset_export > 0:
                    # 音频晚开始：只保留能落在视频范围内的前段音频。
                    visible_duration = min(audio_source.duration, max(video_duration - audio_offset_export, 0))
                    if visible_duration > 0:
                        audio_clip = audio_source.subclip(0, visible_duration).set_start(audio_offset_export)
                        clips_to_composite.append(audio_clip)
                else:
                    # 音频早开始：截取掉前面 (-offset) 秒，再限制到视频末尾。
                    start_cut = -audio_offset_export
                    if start_cut < audio_source.duration:
                        end_cut = min(audio_source.duration, start_cut + video_duration)
                        if end_cut > start_cut:
                            audio_clip = audio_source.subclip(start_cut, end_cut)
                            clips_to_composite.append(audio_clip)

                # 关键修复：
                # 无论如何，都使用 CompositeAudioClip。
                # CompositeAudioClip 允许设置一个比内容更长的 duration，
                # 空白区域会自动视为静音，而不会像 AudioFileClip 那样报错。
                if clips_to_composite:
                    final_audio = CompositeAudioClip(clips_to_composite)
                    final_audio = final_audio.set_duration(video_duration)
                    final_clip = final_clip.set_duration(video_duration).set_audio(final_audio)
                else:
                    # 如果音频被裁没了，就不设置音频
                    pass

            # 5. 写出最终文件
            # remove_temp=True (删除临时音频文件), codec='libx264' (兼容性好)
            # 注意：write_videofile 会打印进度到控制台
            final_clip.write_videofile(self.output_path, codec="libx264", audio_codec="aac", logger=None)

            # 清理临时视频
            final_clip.close()
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)

            self.progress.emit(100)
            self.finished.emit(f"Exported to {self.output_path}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))

    def cancel(self):
        self.is_cancelled = True
