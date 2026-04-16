"""
论文实验数据采集脚本
====================
生成第5章表5-4、表5-5、表5-6的实验数据。
使用合成关键帧+节拍序列，精确测量三种匹配算法的性能指标。

用法:
    conda activate pose_unified
    python run_thesis_experiments.py
"""

import os, sys, time, json
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from engines.algorithms import AlgorithmEngine
from core.algorithm_config import AlgorithmConfig

OUTPUT_DIR = os.path.join(_HERE, "test_output", "thesis_experiments")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_test_cases(n_cases=10, seed=42, ensure_n_le_m=False):
    """
    生成多组模拟关键帧和节拍序列，模拟真实场景。
    关键帧和节拍的时间跨度相近（模拟同一段音乐的场景）。

    ensure_n_le_m: 若True，保证关键帧数 <= 节拍数（适合dp_full测试）
    """
    rng = np.random.RandomState(seed)
    cases = []
    for i in range(n_cases):
        # 音乐参数
        bpm = rng.uniform(100, 150)
        beat_interval = 60.0 / bpm
        # 音乐时长 8~20秒
        music_dur = rng.uniform(8.0, 20.0)
        beat_start = rng.uniform(0.2, 1.0)
        n_beats = int((music_dur - beat_start) / beat_interval)
        n_beats = max(n_beats, 10)
        audio_beats = [beat_start + j * beat_interval + rng.normal(0, 0.015)
                       for j in range(n_beats)]
        audio_beats = sorted([max(0, b) for b in audio_beats])

        # 关键帧：在相似时间范围内生成
        kf_start = rng.uniform(0.3, 1.5)
        if ensure_n_le_m:
            n_kf = rng.randint(max(5, n_beats // 3), n_beats + 1)  # N <= M
        else:
            n_kf = rng.randint(max(5, n_beats // 3), max(n_beats + 5, 15))
        # 关键帧间距: 平均间距使得总时长接近音乐时长
        avg_gap = (music_dur - kf_start) / max(n_kf, 1)
        avg_gap = max(avg_gap, 0.3)
        kf_gaps = rng.uniform(avg_gap * 0.5, avg_gap * 1.5, size=n_kf)
        kf_times_sec = kf_start + np.cumsum(kf_gaps)
        kf_indices = (kf_times_sec * 30).astype(int).tolist()

        cases.append({
            "id": i,
            "n_kf": len(kf_indices),
            "n_beats": len(audio_beats),
            "bpm": bpm,
            "kf_indices": kf_indices,
            "audio_beats": audio_beats,
            "fps": 30.0,
        })
    return cases


def compute_metrics(kf_indices, audio_beats, output_times, fps, algorithm_name):
    """计算匹配质量指标"""
    kf_times = [kf / fps for kf in kf_indices]
    beat_arr = np.array(audio_beats)
    n_kf = len(kf_times)

    # 1. 平均对齐偏差 (ms): 每个输出时间到最近节拍的距离
    deviations = []
    anchored = 0
    for t in output_times:
        min_dist = float(np.min(np.abs(beat_arr - t)))
        deviations.append(min_dist * 1000)  # 转ms
        if min_dist < 1e-6:
            anchored += 1

    avg_deviation = np.mean(deviations)
    anchor_rate = anchored / n_kf

    # 2. 速度比序列和标准差
    ratios = []
    for i in range(n_kf - 1):
        dt_in = kf_times[i + 1] - kf_times[i]
        dt_out = output_times[i + 1] - output_times[i]
        if dt_in > 1e-9:
            ratios.append(dt_out / dt_in)

    ratio_std = float(np.std(ratios)) if ratios else 0.0
    ratio_mean = float(np.mean(ratios)) if ratios else 1.0

    return {
        "avg_deviation_ms": round(avg_deviation, 2),
        "ratio_std": round(ratio_std, 4),
        "ratio_mean": round(ratio_mean, 3),
        "anchor_rate": round(anchor_rate * 100, 1),
        "n_kf": n_kf,
        "n_beats": len(audio_beats),
    }


def experiment_table_5_4():
    """表5-4: 三种匹配算法性能对比"""
    print("=" * 60)
    print("实验: 表5-4 三种匹配算法性能对比")
    print("=" * 60)

    cases = generate_test_cases(n_cases=20, seed=42, ensure_n_le_m=True)
    algorithms = {
        "greedy": AlgorithmConfig(match_algorithm="greedy"),
        "dp_full": AlgorithmConfig(match_algorithm="dp_full",
                                    speed_ratio_min=0.7, speed_ratio_max=1.4,
                                    speed_smoothness=4.0,
                                    duration_scale_min=0.5, duration_scale_max=2.0),
        "dp_subset": AlgorithmConfig(match_algorithm="dp_subset",
                                      speed_ratio_min=0.7, speed_ratio_max=1.4,
                                      speed_smoothness=4.0, skip_cost=0.1,
                                      duration_scale_min=0.5, duration_scale_max=2.0),
    }

    results = {name: {"deviations": [], "ratio_stds": [], "anchor_rates": [],
                       "times_ms": [], "ratio_means": []}
               for name in algorithms}

    for case in cases:
        for algo_name, config in algorithms.items():
            kf = case["kf_indices"]
            beats = case["audio_beats"]
            fps = case["fps"]

            t0 = time.perf_counter()
            try:
                output = AlgorithmEngine.match_beats(kf, beats, fps, config)
            except Exception as e:
                print(f"  [WARN] case {case['id']} {algo_name}: {e}")
                continue
            elapsed_ms = (time.perf_counter() - t0) * 1000

            if len(output) != len(kf):
                print(f"  [WARN] case {case['id']} {algo_name}: output length mismatch")
                continue

            metrics = compute_metrics(kf, beats, output, fps, algo_name)
            results[algo_name]["deviations"].append(metrics["avg_deviation_ms"])
            results[algo_name]["ratio_stds"].append(metrics["ratio_std"])
            results[algo_name]["anchor_rates"].append(metrics["anchor_rate"])
            results[algo_name]["times_ms"].append(elapsed_ms)
            results[algo_name]["ratio_means"].append(metrics["ratio_mean"])

    # 汇总
    print("\n表5-4 三种匹配算法性能对比")
    print(f"{'指标':<25} {'贪心':>12} {'全匹配DP':>12} {'子集DP':>12}")
    print("-" * 65)

    for algo_name in ["greedy", "dp_full", "dp_subset"]:
        r = results[algo_name]
        if not r["deviations"]:
            continue

    row_dev = "平均对齐偏差 (ms)"
    row_std = "速度比标准差 σ_r"
    row_anc = "锚定率 (%)"
    row_time = "运行时间 (ms)"
    row_rmean = "平均速度比"

    table = {}
    for algo_name in ["greedy", "dp_full", "dp_subset"]:
        r = results[algo_name]
        if not r["deviations"]:
            table[algo_name] = {"dev": "N/A", "std": "N/A", "anc": "N/A", "time": "N/A", "rmean": "N/A"}
            continue
        table[algo_name] = {
            "dev": f"{np.mean(r['deviations']):.1f}",
            "std": f"{np.mean(r['ratio_stds']):.4f}",
            "anc": f"{np.mean(r['anchor_rates']):.1f}",
            "time": f"{np.mean(r['times_ms']):.1f}",
            "rmean": f"{np.mean(r['ratio_means']):.3f}",
        }

    print(f"{'平均对齐偏差 (ms)':<25} {table['greedy']['dev']:>12} {table['dp_full']['dev']:>12} {table['dp_subset']['dev']:>12}")
    print(f"{'速度比标准差 σ_r':<25} {table['greedy']['std']:>12} {table['dp_full']['std']:>12} {table['dp_subset']['std']:>12}")
    print(f"{'锚定率 (%)':<25} {table['greedy']['anc']:>12} {table['dp_full']['anc']:>12} {table['dp_subset']['anc']:>12}")
    print(f"{'运行时间 (ms)':<25} {table['greedy']['time']:>12} {table['dp_full']['time']:>12} {table['dp_subset']['time']:>12}")
    print(f"{'平均速度比':<25} {table['greedy']['rmean']:>12} {table['dp_full']['rmean']:>12} {table['dp_subset']['rmean']:>12}")

    return results


def experiment_table_5_5():
    """表5-5: 速度平滑权重 w_smooth 对匹配结果的影响 (dp_subset)"""
    print("\n" + "=" * 60)
    print("实验: 表5-5 速度平滑权重 w_smooth 的影响")
    print("=" * 60)

    cases = generate_test_cases(n_cases=20, seed=123)
    w_smooth_values = [0.0, 2.0, 4.0, 8.0, 16.0]

    print(f"\n{'w_smooth':<10} {'σ_r':>10} {'锚定率(%)':>12} {'平均速度比':>14}")
    print("-" * 50)

    table_data = {}
    for ws in w_smooth_values:
        config = AlgorithmConfig(
            match_algorithm="dp_subset",
            speed_ratio_min=0.7, speed_ratio_max=1.4,
            speed_smoothness=ws, skip_cost=0.1,
            duration_scale_min=0.5, duration_scale_max=2.0,
        )
        ratio_stds = []
        anchor_rates = []
        ratio_means = []

        for case in cases:
            try:
                output = AlgorithmEngine.match_beats(
                    case["kf_indices"], case["audio_beats"], case["fps"], config)
            except Exception:
                continue
            if len(output) != len(case["kf_indices"]):
                continue
            m = compute_metrics(case["kf_indices"], case["audio_beats"], output, case["fps"], "dp_subset")
            ratio_stds.append(m["ratio_std"])
            anchor_rates.append(m["anchor_rate"])
            ratio_means.append(m["ratio_mean"])

        avg_std = np.mean(ratio_stds) if ratio_stds else 0
        avg_anc = np.mean(anchor_rates) if anchor_rates else 0
        avg_rmean = np.mean(ratio_means) if ratio_means else 1
        table_data[ws] = {"ratio_std": avg_std, "anchor_rate": avg_anc, "ratio_mean": avg_rmean}
        print(f"{ws:<10.1f} {avg_std:>10.4f} {avg_anc:>12.1f} {avg_rmean:>14.3f}")

    return table_data


def experiment_table_5_6():
    """表5-6: 跳过代价 w_skip 对匹配结果的影响 (dp_subset)"""
    print("\n" + "=" * 60)
    print("实验: 表5-6 跳过代价 w_skip 的影响")
    print("=" * 60)

    cases = generate_test_cases(n_cases=20, seed=456)
    w_skip_values = [0.01, 0.05, 0.1, 0.5, 1.0]

    print(f"\n{'w_skip':<10} {'锚定率(%)':>12} {'σ_r':>10} {'最大段跨度(帧)':>16}")
    print("-" * 52)

    table_data = {}
    for wsk in w_skip_values:
        config = AlgorithmConfig(
            match_algorithm="dp_subset",
            speed_ratio_min=0.7, speed_ratio_max=1.4,
            speed_smoothness=4.0, skip_cost=wsk,
            duration_scale_min=0.5, duration_scale_max=2.0,
        )
        anchor_rates = []
        ratio_stds = []
        max_spans = []

        for case in cases:
            try:
                output = AlgorithmEngine.match_beats(
                    case["kf_indices"], case["audio_beats"], case["fps"], config)
            except Exception:
                continue
            if len(output) != len(case["kf_indices"]):
                continue
            m = compute_metrics(case["kf_indices"], case["audio_beats"], output, case["fps"], "dp_subset")
            anchor_rates.append(m["anchor_rate"])
            ratio_stds.append(m["ratio_std"])

            # 计算最大段跨度（相邻锚点之间的最大关键帧数）
            kf_times = [kf / case["fps"] for kf in case["kf_indices"]]
            beat_arr = np.array(case["audio_beats"])
            anchored_indices = []
            for idx, t in enumerate(output):
                min_dist = float(np.min(np.abs(beat_arr - t)))
                if min_dist < 1e-6:
                    anchored_indices.append(idx)
            if len(anchored_indices) >= 2:
                spans = [anchored_indices[i+1] - anchored_indices[i]
                         for i in range(len(anchored_indices)-1)]
                max_spans.append(max(spans))
            else:
                max_spans.append(len(kf_times))

        avg_anc = np.mean(anchor_rates) if anchor_rates else 0
        avg_std = np.mean(ratio_stds) if ratio_stds else 0
        avg_span = np.mean(max_spans) if max_spans else 0
        table_data[wsk] = {"anchor_rate": avg_anc, "ratio_std": avg_std, "max_span": avg_span}
        print(f"{wsk:<10.2f} {avg_anc:>12.1f} {avg_std:>10.4f} {avg_span:>16.0f}")

    return table_data


if __name__ == "__main__":
    print("论文实验数据采集")
    print("=" * 60)

    all_results = {}

    # 表5-4
    t0 = time.time()
    r54 = experiment_table_5_4()
    all_results["table_5_4"] = {k: {mk: round(np.mean(mv), 4) for mk, mv in v.items()}
                                  for k, v in r54.items()}
    print(f"\n表5-4 耗时: {time.time()-t0:.1f}s")

    # 表5-5
    t0 = time.time()
    r55 = experiment_table_5_5()
    all_results["table_5_5"] = {str(k): v for k, v in r55.items()}
    print(f"\n表5-5 耗时: {time.time()-t0:.1f}s")

    # 表5-6
    t0 = time.time()
    r56 = experiment_table_5_6()
    all_results["table_5_6"] = {str(k): v for k, v in r56.items()}
    print(f"\n表5-6 耗时: {time.time()-t0:.1f}s")

    # 保存原始数据
    out_path = os.path.join(OUTPUT_DIR, "experiment_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n原始数据已保存: {out_path}")
    print("完成！")
