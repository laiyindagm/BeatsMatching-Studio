# engines/algorithms.py

import cv2
import numpy as np
import math

import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from core.algorithm_config import AlgorithmConfig


class AlgorithmEngine:
    """
    核心算法封装，不涉及 UI 逻辑
    """

    @staticmethod
    def extract_motion_keyframes(frames: list[np.ndarray], threshold: float = 0.5) -> list[int]:
        """
        基于内存中已缓存的帧列表提取动作关键帧。

        :param frames: 视频帧列表 (RGB numpy arrays)
        :param threshold: 敏感度阈值
        :return: 关键帧索引列表
        """
        if not frames:
            return [0]

        frame_indices = []
        prev_gray = None

        # 遍历所有帧
        # 为了性能，可以跳帧检测，比如每隔 2 帧检测一次
        # 但为了准确性，帧差法最好逐帧比较

        for i, frame in enumerate(frames):
            # 1. 转换为灰度图
            # 注意：frames 里存的是 RGB，cv2 需要 BGR 转 GRAY?
            # 其实 RGB 转 GRAY 和 BGR 转 GRAY 系数不同，但做帧差无所谓，只要统一即可。
            # 直接用 cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            if frame is None: continue

            # 缩小图像以加速计算 (64x64 足够检测宏观动作了)
            small_frame = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small_frame, cv2.COLOR_RGB2GRAY)

            if prev_gray is not None:
                # 2. 计算帧差 (绝对差)
                diff = cv2.absdiff(prev_gray, gray)

                # 3. 计算变化量 (平均像素差异)
                score = np.mean(diff)

                # 4. 阈值判断
                # 这里的阈值 threshold (0.5) 是一个相对值，我们需要把它映射到像素差异(0-255)
                # 假设 threshold 0.5 对应 平均差异 5.0 (经验值)
                abs_threshold = threshold * 10.0

                # 并且设置最小关键帧间隔 (例如 10 帧)，防止太密
                min_interval = 10

                if score > abs_threshold and (not frame_indices or i - frame_indices[-1] > min_interval):
                    frame_indices.append(i)

            prev_gray = gray

        # 如果没检测到任何关键帧，至少返回首帧
        if not frame_indices:
            frame_indices = [0]

        return frame_indices

    @staticmethod
    def match_beats(
        video_keyframes: list[int],
        audio_beats: list[float],
        video_fps: float,
        config: AlgorithmConfig | None = None,
    ) -> list[float]:
        """
        将视频关键帧匹配到音频节拍。

        支持三种算法（通过 config.match_algorithm 选择）:
          - 'greedy':     贪心最近邻匹配（简单快速）
          - 'dp_full':    全匹配 DP（每个关键帧都严格对齐到 beat）
          - 'dp_subset':  子集 DP + 等比例插值（锚点严格对齐，非锚点比例插值）

        注意：调用方（AlgorithmProxy）已将 BOUNDARY 首尾帧过滤掉，
        此处 video_keyframes 仅包含提取出的运动关键帧。
        首KF 的 output_time 需满足 ≥ head_time（首关键帧的原始时间），
        否则头部 BOUNDARY 区间的 output_time 会变为负数。

        :return: 输出时间点列表（与 video_keyframes 一一对应）
        """
        if not video_keyframes:
            return []
        if config is None:
            config = AlgorithmConfig()

        kf_times = [kf / video_fps for kf in video_keyframes]
        n_kf = len(kf_times)
        n_bt = len(audio_beats)

        # 首KF的原始时间 = 首帧到首KF的原速时长
        head_time = kf_times[0]

        # ── 边界情况 ──
        if n_bt == 0:
            return list(kf_times)

        beat_arr = np.asarray(audio_beats, dtype=np.float64)

        if n_kf == 1:
            # 单KF：选最近且满足 head_time 约束的 beat
            valid = beat_arr[beat_arr >= head_time - 1e-9]
            if len(valid) == 0:
                valid = beat_arr  # 放宽约束
            return [float(valid[np.argmin(np.abs(valid - kf_times[0]))])]
        if n_bt == 1:
            offset = audio_beats[0] - kf_times[0]
            return [t + offset for t in kf_times]

        # 兼容旧配置: "dp" 视为 "dp_subset"
        algo = getattr(config, 'match_algorithm', 'dp_subset')
        if algo == 'dp':
            algo = 'dp_subset'

        if algo == 'greedy':
            return AlgorithmEngine._match_greedy(kf_times, beat_arr, head_time)
        elif algo == 'dp_full':
            return AlgorithmEngine._match_dp_full(
                kf_times, audio_beats, beat_arr, config, head_time)
        else:  # dp_subset
            return AlgorithmEngine._match_dp_subset(
                kf_times, audio_beats, beat_arr, config, head_time)

    # ─────────────────────────────────────────────────────────
    # 算法 1: 贪心最近邻
    # ─────────────────────────────────────────────────────────
    @staticmethod
    def _match_greedy(
        kf_times: list[float],
        beat_arr: np.ndarray,
        head_time: float = 0.0,
    ) -> list[float]:
        """
        贪心最近邻匹配。

        每个关键帧依次匹配到时间最近的、尚未使用的 beat（保持时间单调）。
        首KF只匹配 >= head_time 的 beat（保证首帧区间不被挤入负时间）。
        """
        n_kf = len(kf_times)
        # 首KF约束：只能匹配 >= head_time 的 beat
        first_valid = int(np.searchsorted(beat_arr, head_time - 1e-9, side='left'))
        used_beat = first_valid
        result = []

        for i in range(n_kf):
            if used_beat >= len(beat_arr):
                # beat 用完，按前段间隔外推
                if len(result) >= 2:
                    last_gap = result[-1] - result[-2]
                    result.append(result[-1] + last_gap)
                else:
                    result.append(kf_times[i])
                continue

            candidates = beat_arr[used_beat:]
            diffs = np.abs(candidates - kf_times[i])
            best_local = int(np.argmin(diffs))
            best_global = used_beat + best_local

            result.append(float(beat_arr[best_global]))
            used_beat = best_global + 1

        print(f"[Match] greedy: {n_kf} 关键帧全部贪心匹配")
        return result

    # ─────────────────────────────────────────────────────────
    # 算法 2: 全匹配 DP（所有关键帧 → beat）
    # ─────────────────────────────────────────────────────────
    @staticmethod
    def _match_dp_full(
        kf_times: list[float],
        audio_beats: list[float],
        beat_arr: np.ndarray,
        config: AlgorithmConfig,
        head_time: float = 0.0,
    ) -> list[float]:
        """
        全匹配动态规划：每个关键帧都对齐到一个 beat。

        DP 状态: dp[i][j] = kf[i] 对齐到 beat[j] 的最小累计代价
        转移:    dp[i][j] = min over k<j { dp[i-1][k] + cost(i, k→j) }
        约束:    ratio ∈ [speed_ratio_min, speed_ratio_max]
        代价:    速度偏离 + 速度平滑

        特点: 所有关键帧严格对齐 beat，无插值
        适用: 关键帧数 ≤ beat 数 的场景
        """
        n_kf = len(kf_times)
        n_bt = len(audio_beats)

        ratio_lo = config.speed_ratio_min
        ratio_hi = config.speed_ratio_max
        smooth_w = getattr(config, 'speed_smoothness', 4.0)
        dur_lo = getattr(config, 'duration_scale_min', 0.0)
        dur_hi = getattr(config, 'duration_scale_max', float('inf'))
        orig_dur = kf_times[-1] - kf_times[0]

        INF = float('inf')

        dp = [[INF] * n_bt for _ in range(n_kf)]
        back = [[-1] * n_bt for _ in range(n_kf)]
        seg_r = [[1.0] * n_bt for _ in range(n_kf)]

        # 首帧初始化（首KF只能匹配 >= head_time 的 beat）
        for j in range(n_bt):
            if audio_beats[j] < head_time - 1e-9:
                continue  # 跳过会使首帧落入负时间的 beat
            dp[0][j] = 0.3 * abs(kf_times[0] - audio_beats[j])

        # 填充 DP: kf[i] → beat[j], 前一帧 kf[i-1] → beat[k]
        for i in range(1, n_kf):
            dt_in = kf_times[i] - kf_times[i - 1]
            if dt_in < 1e-9:
                dt_in = 1e-9

            for j in range(i, n_bt):  # j >= i (每个 kf 占一个 beat)
                # 用 ratio 约束二分搜索 k 范围
                target_lo = audio_beats[j] - ratio_hi * dt_in
                target_hi = audio_beats[j] - ratio_lo * dt_in
                k_start = max(i - 1, int(np.searchsorted(beat_arr, target_lo, side='left')))
                k_end = min(j, int(np.searchsorted(beat_arr, target_hi, side='right')))

                for k in range(k_start, k_end):
                    if dp[i - 1][k] >= INF:
                        continue
                    dt_out = audio_beats[j] - audio_beats[k]
                    if dt_out <= 0:
                        continue
                    ratio = dt_out / dt_in

                    c_speed = (ratio - 1.0) ** 2
                    c_smooth = smooth_w * (ratio - seg_r[i - 1][k]) ** 2 if i >= 2 else 0.0

                    total = dp[i - 1][k] + c_speed + c_smooth
                    if total < dp[i][j]:
                        dp[i][j] = total
                        back[i][j] = k
                        seg_r[i][j] = ratio

        # 回溯（满足时长缩放约束）
        endings = sorted(
            ((dp[n_kf - 1][j], j) for j in range(n_bt) if dp[n_kf - 1][j] < INF),
            key=lambda x: x[0],
        )

        best_path = None
        for _, j_end in endings:
            path = [0] * n_kf
            path[n_kf - 1] = j_end
            for ii in range(n_kf - 2, -1, -1):
                path[ii] = back[ii + 1][path[ii + 1]]

            if orig_dur > 1e-9:
                out_dur = audio_beats[j_end] - audio_beats[path[0]]
                scale = out_dur / orig_dur
                if scale < dur_lo or scale > dur_hi:
                    continue
            best_path = path
            break

        if best_path is None:
            print("[Match] dp_full: DP 无解, 回退贪心")
            return AlgorithmEngine._match_greedy(kf_times, beat_arr, head_time)

        result = [audio_beats[best_path[i]] for i in range(n_kf)]

        # 日志
        ratios = []
        for i in range(n_kf - 1):
            dt_i = kf_times[i + 1] - kf_times[i]
            dt_o = result[i + 1] - result[i]
            ratios.append(dt_o / max(dt_i, 1e-9))
        chg = [abs(ratios[s + 1] - ratios[s]) for s in range(len(ratios) - 1)]
        print(f"[Match] dp_full: {n_kf}/{n_kf} 全匹配")
        if ratios:
            print(f"[Match] 段速度比: {[f'{r:.3f}' for r in ratios]}"
                  f"{', 最大跳变: ' + f'{max(chg):.4f}' if chg else ''}")

        return result

    # ─────────────────────────────────────────────────────────
    # 算法 3: 子集 DP + 等比例插值
    # ─────────────────────────────────────────────────────────
    @staticmethod
    def _match_dp_subset(
        kf_times: list[float],
        audio_beats: list[float],
        beat_arr: np.ndarray,
        config: AlgorithmConfig,
        head_time: float = 0.0,
    ) -> list[float]:
        """
        子集匹配 + 等比例插值。

        Phase 1 — DP 子集匹配
          · 选择视频关键帧的最优子集作为锚点，严格对齐到 beat
          · 首末帧始终为锚点
          · 代价 = 速度偏离 + 段间速度变化 + 跳过惩罚

        Phase 2 — 等比例插值
          · 相邻锚点间的未匹配关键帧按原始时间比例线性插值
          · 保证: 锚点帧严格对齐 beat; 同段内速度恒定; 段间平滑过渡

        例: kf=[a1,a2,a3,a4], beats=[b1,b2]
            DP 选 a1→b1, a4→b2 为锚点
            a2, a3 按原始比例插值:
              a2_new = b1 + (a2-a1)/(a4-a1) * (b2-b1)
              a3_new = b1 + (a3-a1)/(a4-a1) * (b2-b1)
        """
        n_kf = len(kf_times)
        n_bt = len(audio_beats)

        ratio_lo = config.speed_ratio_min
        ratio_hi = config.speed_ratio_max
        smooth_w = getattr(config, 'speed_smoothness', 4.0)
        dur_lo = getattr(config, 'duration_scale_min', 0.0)
        dur_hi = getattr(config, 'duration_scale_max', float('inf'))
        skip_w = getattr(config, 'skip_cost', 0.1)
        orig_dur = kf_times[-1] - kf_times[0]

        INF = float('inf')

        dp = [[INF] * n_bt for _ in range(n_kf)]
        back = [[(-1, -1)] * n_bt for _ in range(n_kf)]
        seg_r = [[1.0] * n_bt for _ in range(n_kf)]

        for j in range(n_bt):
            if audio_beats[j] < head_time - 1e-9:
                continue  # 跳过会使首帧落入负时间的 beat
            dp[0][j] = 0.3 * abs(kf_times[0] - audio_beats[j])

        MAX_KF_BACK = min(n_kf, 20)

        for i in range(1, n_kf):
            for j in range(1, n_bt):
                k_lo = max(0, i - MAX_KF_BACK)
                for k in range(k_lo, i):
                    dt_in = kf_times[i] - kf_times[k]
                    if dt_in < 1e-9:
                        continue

                    target_lo = audio_beats[j] - ratio_hi * dt_in
                    target_hi = audio_beats[j] - ratio_lo * dt_in
                    l_start = max(0, int(np.searchsorted(beat_arr, target_lo, side='left')))
                    l_end = min(int(np.searchsorted(beat_arr, target_hi, side='right')), j)

                    for l in range(l_start, l_end):
                        if dp[k][l] >= INF:
                            continue
                        dt_out = audio_beats[j] - audio_beats[l]
                        if dt_out <= 0:
                            continue
                        ratio = dt_out / dt_in

                        c_speed = (ratio - 1.0) ** 2
                        c_smooth = 0.0 if k == 0 else smooth_w * (ratio - seg_r[k][l]) ** 2
                        c_skip = skip_w * (i - k - 1)

                        total = dp[k][l] + c_speed + c_smooth + c_skip
                        if total < dp[i][j]:
                            dp[i][j] = total
                            back[i][j] = (k, l)
                            seg_r[i][j] = ratio

        endings = sorted(
            ((dp[n_kf - 1][j], j) for j in range(n_bt) if dp[n_kf - 1][j] < INF),
            key=lambda x: x[0],
        )

        matched = None
        for _, j_end in endings:
            pairs = []
            ci, cj = n_kf - 1, j_end
            while ci >= 0:
                pairs.append((ci, cj))
                ci, cj = back[ci][cj]
            pairs.reverse()

            if len(pairs) >= 2 and orig_dur > 1e-9:
                out_dur = audio_beats[pairs[-1][1]] - audio_beats[pairs[0][1]]
                scale = out_dur / orig_dur
                if scale < dur_lo or scale > dur_hi:
                    continue
            matched = pairs
            break

        if matched is None:
            print("[Match] dp_subset: DP 无解, 回退首末锚定")
            matched = AlgorithmEngine._fallback_anchors(kf_times, beat_arr)

        # 日志
        n_anchored = len(matched)
        anchors_str = ", ".join(
            f"kf{p[0]}({kf_times[p[0]]:.2f}s)→beat{p[1]}({audio_beats[p[1]]:.2f}s)"
            for p in matched
        )
        ratios = []
        for idx in range(len(matched) - 1):
            i0, j0 = matched[idx]
            i1, j1 = matched[idx + 1]
            dt_i = kf_times[i1] - kf_times[i0]
            dt_o = audio_beats[j1] - audio_beats[j0]
            ratios.append(dt_o / max(dt_i, 1e-9))
        print(f"[Match] dp_subset: {n_anchored}/{n_kf} 锚定 ({n_kf - n_anchored} 插值)")
        print(f"[Match] 锚点: {anchors_str}")
        if ratios:
            chg = [abs(ratios[s + 1] - ratios[s]) for s in range(len(ratios) - 1)]
            print(f"[Match] 段速度比: {[f'{r:.3f}' for r in ratios]}"
                  f"{', 最大跳变: ' + f'{max(chg):.4f}' if chg else ''}")

        return AlgorithmEngine._build_output(kf_times, audio_beats, matched)

    # ─────────────────────────────────────────────────────────
    # 辅助方法
    # ─────────────────────────────────────────────────────────
    @staticmethod
    def _build_output(
        kf_times: list[float],
        audio_beats: list[float],
        matched_pairs: list[tuple[int, int]],
    ) -> list[float]:
        """
        从锚点对构建完整输出时间序列。

        锚点帧 → 严格对齐到 beat（精确值）
        非锚点帧 → 相邻锚点间按原始时间比例线性插值
        """
        n_kf = len(kf_times)
        output = [0.0] * n_kf

        for kf_i, bt_j in matched_pairs:
            output[kf_i] = audio_beats[bt_j]

        for seg in range(len(matched_pairs) - 1):
            i_start, j_start = matched_pairs[seg]
            i_end, j_end = matched_pairs[seg + 1]

            t_start = audio_beats[j_start]
            t_end = audio_beats[j_end]
            orig_start = kf_times[i_start]
            orig_end = kf_times[i_end]
            orig_span = orig_end - orig_start

            for m in range(i_start + 1, i_end):
                if orig_span > 1e-9:
                    frac = (kf_times[m] - orig_start) / orig_span
                else:
                    frac = (m - i_start) / (i_end - i_start)
                output[m] = t_start + frac * (t_end - t_start)

        return output

    @staticmethod
    def _fallback_anchors(
        kf_times: list[float],
        beat_arr: np.ndarray,
    ) -> list[tuple[int, int]]:
        """DP 无解时回退：仅首末关键帧锚定到最近 beat"""
        n = len(kf_times)
        j_first = int(np.argmin(np.abs(beat_arr - kf_times[0])))
        j_last = int(np.argmin(np.abs(beat_arr - kf_times[-1])))
        if j_last <= j_first:
            j_last = min(j_first + 1, len(beat_arr) - 1)
        if j_last == j_first and j_first > 0:
            j_first = j_last - 1
        return [(0, j_first), (n - 1, j_last)]
