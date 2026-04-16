# 核心算法：负责计算 时间<->帧 的变速映射
import bisect
import math
from .structures import Keyframe, EaseType


class TimeMapper:
    @staticmethod
    def map_time_to_frame(current_time: float, keyframes: list[Keyframe],
                          fps: float = 30.0, total_frames: int = 0) -> float:
        """
        核心映射算法
        :param current_time: 时间轴上的当前时间 (秒)
        :param keyframes: 已排序的关键帧列表
        :param fps: 视频帧率（用于首尾区间原速映射）
        :param total_frames: 视频总帧数（用于尾部区间上限，0=不限制）
        :return: 对应的源视频帧索引 (float, 允许小数以便后续做动效或精确插值)
        """
        if not keyframes:
            return current_time * fps if fps > 0 else 0.0

        first_kf = keyframes[0]
        last_kf = keyframes[-1]

        # ── 首关键帧之前：原速播放 ──
        # frame = first_kf.frame_idx - (first_kf.output_time - t) * fps
        if current_time < first_kf.output_time:
            frame = first_kf.frame_idx - (first_kf.output_time - current_time) * fps
            return max(frame, 0.0)

        # ── 尾关键帧之后：原速播放 ──
        # frame = last_kf.frame_idx + (t - last_kf.output_time) * fps
        if current_time > last_kf.output_time:
            frame = last_kf.frame_idx + (current_time - last_kf.output_time) * fps
            if total_frames > 0:
                frame = min(frame, float(total_frames - 1))
            return frame

        # ── 恰好在首/尾关键帧上 ──
        if current_time <= first_kf.output_time:
            return float(first_kf.frame_idx)
        if current_time >= last_kf.output_time:
            return float(last_kf.frame_idx)

        # 2. 查找当前时间所在的区间 [prev_kf, next_kf]
        idx = bisect.bisect_right(keyframes, Keyframe(output_time=current_time, frame_idx=0))
        prev_kf = keyframes[idx - 1]
        next_kf = keyframes[idx]

        # 3. 计算线性进度 (Linear Progress) [0, 1]
        t_duration = next_kf.output_time - prev_kf.output_time
        if t_duration <= 0:
            return float(prev_kf.frame_idx)

        linear_p = (current_time - prev_kf.output_time) / t_duration

        # 4. 应用变速函数 (Easing)
        eased_p = TimeMapper.apply_easing(linear_p, prev_kf.ease_type)

        # 5. 映射到源帧
        frame_diff = next_kf.frame_idx - prev_kf.frame_idx
        target_frame = prev_kf.frame_idx + frame_diff * eased_p

        return target_frame

    @staticmethod
    def apply_easing(x: float, ease_type: EaseType) -> float:
        """
        变速函数实现: [0, 1] -> [0, 1]
        """
        if x <= 0: return 0.0
        if x >= 1: return 1.0

        if ease_type == EaseType.LINEAR:
            return x
        elif ease_type == EaseType.EASE_IN_QUAD:
            return x * x
        elif ease_type == EaseType.EASE_OUT_QUAD:
            return x * (2 - x)
        elif ease_type == EaseType.EASE_IN_OUT:
            # 简单的 sigmoid 变体或 cubic
            return x * x * (3 - 2 * x)  # SmoothStep

        return x