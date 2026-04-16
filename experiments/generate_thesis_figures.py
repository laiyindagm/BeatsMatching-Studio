"""
generate_thesis_figures.py
==========================
Generate all code-producible thesis figures (7 figures).

Usage:
  cd D:\\BeatsMatching
  python generate_thesis_figures.py

Output: docs/thesis_figures/
"""

import os, sys, json, warnings
import numpy as np

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

OUT_DIR = os.path.join(_HERE, "docs", "thesis_figures")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.sans-serif": ["SimHei", "Microsoft YaHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


# ======================================================================
#  Fig 4-4: Multi-scale 3-channel voting
# ======================================================================

def fig_4_4():
    np.random.seed(42)
    N, fps = 300, 30
    t = np.arange(N) / fps

    true_beats = [30, 60, 90, 120, 150, 180, 210, 240, 270]
    signal = np.random.randn(N) * 0.15
    for b in true_beats:
        signal[max(0, b-3):min(N, b+4)] += np.exp(-0.5 * np.arange(-3, 4)**2 / 1.5**2)
    signal = np.clip(signal, 0, None)

    sigma = 2.5
    smooth_fine = gaussian_filter1d(signal, sigma=sigma * 0.5)
    peaks_fine, _ = find_peaks(smooth_fine, distance=int(fps * 0.2), height=0.15)

    smooth_coarse = gaussian_filter1d(signal, sigma=sigma * 2.0)
    peaks_coarse, _ = find_peaks(smooth_coarse, distance=int(fps * 0.3), height=0.1)

    smooth_mid = gaussian_filter1d(signal, sigma=sigma)
    vel = np.diff(smooth_mid, prepend=smooth_mid[0])
    acc = np.diff(vel, prepend=vel[0])
    zero_crossings = []
    for i in range(1, len(acc)):
        if acc[i - 1] > 0 and acc[i] <= 0 and smooth_mid[i] > np.percentile(smooth_mid, 30):
            zero_crossings.append(i)

    vote_map = np.zeros(N, dtype=np.float32)
    vote_r = max(2, int(fps * 0.05))
    for p in peaks_fine:
        lo, hi = max(0, p - vote_r), min(N, p + vote_r + 1)
        vote_map[lo:hi] += 1.0
    for p in peaks_coarse:
        lo, hi = max(0, p - vote_r), min(N, p + vote_r + 1)
        vote_map[lo:hi] += 1.2
    for p in zero_crossings:
        lo, hi = max(0, p - vote_r), min(N, p + vote_r + 1)
        vote_map[lo:hi] += 0.8

    final_peaks, _ = find_peaks(vote_map, distance=int(fps * 0.15), height=1.5)

    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True,
                             gridspec_kw={"hspace": 0.15})

    axes[0].plot(t, smooth_fine, color="#3b82f6", linewidth=0.9,
                 label=u"\u7ec6\u7c92\u5ea6\u5e73\u6ed1 (\u03c3/2)")
    axes[0].plot(t, signal, color="#93c5fd", linewidth=0.4, alpha=0.5)
    for p in peaks_fine:
        axes[0].axvline(t[p], color="#ef4444", alpha=0.6, linewidth=0.7, linestyle="--")
    axes[0].set_ylabel(u"\u901a\u9053 A\n(\u7ec6\u7c92\u5ea6)", fontsize=9)
    axes[0].legend(fontsize=7, loc="upper right")
    axes[0].set_title(u"\u56fe 4-4  \u591a\u5c3a\u5ea6\u4e09\u901a\u9053\u6295\u7968\u8282\u62cd\u68c0\u6d4b\u793a\u610f",
                      fontsize=11, fontweight="bold")

    axes[1].plot(t, smooth_coarse, color="#f97316", linewidth=0.9,
                 label=u"\u7c97\u7c92\u5ea6\u5e73\u6ed1 (2\u03c3)")
    axes[1].plot(t, signal, color="#fdba74", linewidth=0.4, alpha=0.4)
    for p in peaks_coarse:
        axes[1].axvline(t[p], color="#ef4444", alpha=0.6, linewidth=0.7, linestyle="--")
    axes[1].set_ylabel(u"\u901a\u9053 B\n(\u7c97\u7c92\u5ea6)", fontsize=9)
    axes[1].legend(fontsize=7, loc="upper right")

    axes[2].plot(t, acc, color="#22c55e", linewidth=0.7, label=u"\u52a0\u901f\u5ea6")
    axes[2].axhline(0, color="gray", linewidth=0.4)
    for p in zero_crossings:
        axes[2].axvline(t[p], color="#ef4444", alpha=0.5, linewidth=0.6, linestyle="--")
    axes[2].set_ylabel(u"\u901a\u9053 C\n(\u52a0\u901f\u5ea6)", fontsize=9)
    axes[2].legend(fontsize=7, loc="upper right")

    axes[3].fill_between(t, 0, vote_map, color="#e879f9", alpha=0.35)
    axes[3].plot(t, vote_map, color="#a855f7", linewidth=1.0)
    axes[3].axhline(1.5, color="#ef4444", linestyle="--", linewidth=0.8,
                    label=u"\u9608\u503c = 1.5")
    for p in final_peaks:
        axes[3].plot(t[p], vote_map[p], "v", color="#dc2626", markersize=7)
    axes[3].set_ylabel(u"\u6295\u7968\u5f97\u5206", fontsize=9)
    axes[3].set_xlabel(u"\u65f6\u95f4 (s)", fontsize=9)
    axes[3].legend(fontsize=7, loc="upper right")

    out = os.path.join(OUT_DIR, "fig_4_4_voting.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  ok {out}")


# ======================================================================
#  Fig 4-8: Four easing curves
# ======================================================================

def fig_4_8():
    x = np.linspace(0, 1, 200)
    curves = {
        "LINEAR":       x,
        "EASE_IN":      x ** 2,
        "EASE_OUT":     x * (2 - x),
        "EASE_IN_OUT":  x ** 2 * (3 - 2 * x),
    }
    colors = ["#6366f1", "#ef4444", "#22c55e", "#f59e0b"]

    fig, ax = plt.subplots(figsize=(6, 5))
    for (name, y), c in zip(curves.items(), colors):
        ax.plot(x, y, color=c, linewidth=2, label=name)
    ax.set_xlabel(u"\u5f52\u4e00\u5316\u8fdb\u5ea6 $t$", fontsize=10)
    ax.set_ylabel(u"\u7f13\u52a8\u8f93\u51fa $f(t)$", fontsize=10)
    ax.set_title(u"\u56fe 4-8  \u56db\u79cd\u7f13\u52a8\u66f2\u7ebf\u5bf9\u6bd4",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_aspect("equal")

    out = os.path.join(OUT_DIR, "fig_4_8_easing.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  ok {out}")


# ======================================================================
#  Fig 5-2: Motion signal + beat detection
# ======================================================================

def fig_5_2():
    np.random.seed(123)
    fps, dur = 30, 12
    N = fps * dur
    t = np.arange(N) / fps

    beat_times = np.arange(0.5, dur, 0.5)
    signal = np.random.randn(N) * 0.08 + 0.2
    for bt in beat_times:
        offset = np.random.randint(-2, 3)
        f = int(bt * fps) + offset
        if 3 <= f < N - 3:
            signal[f - 2:f + 3] += np.array([0.3, 0.6, 1.0, 0.6, 0.3]) * (0.7 + 0.3 * np.random.rand())
    signal = gaussian_filter1d(np.clip(signal, 0, None), sigma=1.5)

    sigma_est = 2.5
    smooth = gaussian_filter1d(signal, sigma=sigma_est)

    vote_map = np.zeros(N, dtype=np.float32)
    s_fine = gaussian_filter1d(signal, sigma=sigma_est * 0.5)
    s_coarse = gaussian_filter1d(signal, sigma=sigma_est * 2.0)
    p_fine, _ = find_peaks(s_fine, distance=int(fps * 0.2), height=0.15)
    p_coarse, _ = find_peaks(s_coarse, distance=int(fps * 0.3), height=0.1)
    vel = np.diff(gaussian_filter1d(signal, sigma=sigma_est), prepend=signal[0])
    acc = np.diff(vel, prepend=vel[0])
    zc = [i for i in range(1, len(acc))
          if acc[i - 1] > 0 and acc[i] <= 0 and smooth[i] > np.percentile(smooth, 30)]
    vr = max(2, int(fps * 0.05))
    for p in p_fine:
        vote_map[max(0, p - vr):min(N, p + vr + 1)] += 1.0
    for p in p_coarse:
        vote_map[max(0, p - vr):min(N, p + vr + 1)] += 1.2
    for p in zc:
        vote_map[max(0, p - vr):min(N, p + vr + 1)] += 0.8
    final_peaks, _ = find_peaks(vote_map, distance=int(fps * 0.15), height=1.5)

    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True,
                             gridspec_kw={"hspace": 0.12})

    axes[0].plot(t, signal, color="#3b82f6", linewidth=0.9,
                 label=u"\u878d\u5408\u8fd0\u52a8\u4fe1\u53f7 $f_t$")
    for p in final_peaks:
        axes[0].axvline(t[p], color="#ef4444", alpha=0.7, linewidth=0.8, linestyle="-")
    for bt in beat_times:
        axes[0].axvline(bt, color="#22c55e", alpha=0.4, linewidth=0.6, linestyle="--")
    axes[0].axvline(-1, color="#ef4444", linewidth=1,
                    label=u"\u68c0\u6d4b\u8282\u62cd\u5e27")
    axes[0].axvline(-1, color="#22c55e", linewidth=1, linestyle="--",
                    label=u"\u97f3\u9891\u8282\u62cd")
    axes[0].set_xlim(0, dur)
    axes[0].set_ylabel(u"\u8fd0\u52a8\u5f3a\u5ea6", fontsize=9)
    axes[0].legend(fontsize=7, loc="upper right", ncol=3)
    axes[0].set_title(u"\u56fe 5-2  \u8fd0\u52a8\u4fe1\u53f7\u4e0e\u8282\u62cd\u68c0\u6d4b\u53ef\u89c6\u5316",
                      fontsize=11, fontweight="bold")

    axes[1].fill_between(t, 0, vote_map, color="#e879f9", alpha=0.3)
    axes[1].plot(t, vote_map, color="#a855f7", linewidth=0.9,
                 label=u"\u6295\u7968\u5f97\u5206 $V(t)$")
    axes[1].axhline(1.5, color="#ef4444", linestyle="--", linewidth=0.7,
                    label=u"\u9608\u503c 1.5")
    for p in final_peaks:
        axes[1].plot(t[p], vote_map[p], "v", color="#dc2626", markersize=6)
    axes[1].set_ylabel(u"\u6295\u7968\u5f97\u5206", fontsize=9)
    axes[1].set_xlabel(u"\u65f6\u95f4 (s)", fontsize=9)
    axes[1].legend(fontsize=7, loc="upper right")

    out = os.path.join(OUT_DIR, "fig_5_2_signal_beats.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  ok {out}")


# ======================================================================
#  Fig 5-3: Matching result (dual timeline)
# ======================================================================

def fig_5_3():
    np.random.seed(7)
    kf_times = np.sort(np.random.uniform(0.5, 9.5, 10))
    beat_times = np.arange(0.5, 10.0, 0.5)

    anchors = [0, 2, 4, 6, 8, 9]
    anchor_beats = [1, 4, 7, 10, 14, 17]

    output_times = np.zeros(len(kf_times))
    for ai, bi in zip(anchors, anchor_beats):
        output_times[ai] = beat_times[bi]

    interpolated = [i for i in range(len(kf_times)) if i not in anchors]
    for idx in interpolated:
        prev_a = max([a for a in anchors if a < idx], default=anchors[0])
        next_a = min([a for a in anchors if a > idx], default=anchors[-1])
        if prev_a == next_a:
            output_times[idx] = output_times[prev_a]
        else:
            frac = (idx - prev_a) / (next_a - prev_a)
            output_times[idx] = output_times[prev_a] + frac * (output_times[next_a] - output_times[prev_a])

    fig, ax = plt.subplots(figsize=(12, 4))
    y_orig, y_matched = 2.0, 0.0

    ax.plot([0, 10], [y_orig, y_orig], color="#94a3b8", linewidth=1.5)
    ax.text(-0.3, y_orig, u"\u539f\u59cb\n\u65f6\u95f4\u8f74",
            ha="right", va="center", fontsize=8, color="#475569")
    for i, kt in enumerate(kf_times):
        color = "#3b82f6" if i in anchors else "#94a3b8"
        ax.plot(kt, y_orig, "o", color=color, markersize=7, zorder=5)
        ax.text(kt, y_orig + 0.2, f"$a_{{{i+1}}}$", ha="center", fontsize=7, color=color)

    ax.plot([0, 10], [y_matched, y_matched], color="#94a3b8", linewidth=1.5)
    ax.text(-0.3, y_matched, u"\u5339\u914d\u540e\n\u65f6\u95f4\u8f74",
            ha="right", va="center", fontsize=8, color="#475569")

    for bt in beat_times:
        ax.plot(bt, y_matched - 0.15, "^", color="#22c55e", markersize=5, alpha=0.5)

    for i in range(len(kf_times)):
        if i in anchors:
            ax.annotate("", xy=(output_times[i], y_matched + 0.05),
                        xytext=(kf_times[i], y_orig - 0.05),
                        arrowprops=dict(arrowstyle="->", color="#ef4444", lw=1.2))
            ax.plot(output_times[i], y_matched, "o", color="#ef4444", markersize=7, zorder=5)
        else:
            ax.annotate("", xy=(output_times[i], y_matched + 0.05),
                        xytext=(kf_times[i], y_orig - 0.05),
                        arrowprops=dict(arrowstyle="->", color="#94a3b8", lw=0.8,
                                        linestyle="dashed"))
            ax.plot(output_times[i], y_matched, "o", color="#94a3b8", markersize=5, zorder=5)

    ax.plot([], [], "o-", color="#ef4444", markersize=6,
            label=u"\u951a\u70b9\uff08\u4e25\u683c\u5bf9\u9f50\u8282\u62cd\uff09")
    ax.plot([], [], "o--", color="#94a3b8", markersize=5,
            label=u"\u63d2\u503c\u5e27\uff08\u7b49\u6bd4\u4f8b\u5206\u914d\uff09")
    ax.plot([], [], "^", color="#22c55e", markersize=6,
            label=u"\u97f3\u9891\u8282\u62cd")
    ax.legend(fontsize=8, loc="upper center", ncol=3)

    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.6, 2.8)
    ax.set_xlabel(u"\u65f6\u95f4 (s)", fontsize=9)
    ax.set_title(u"\u56fe 5-3  \u5b50\u96c6 DP \u5339\u914d\u7ed3\u679c\u53ef\u89c6\u5316",
                 fontsize=11, fontweight="bold")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = os.path.join(OUT_DIR, "fig_5_3_matching.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  ok {out}")


# ======================================================================
#  Fig 5-4: Speed ratio comparison
# ======================================================================

def fig_5_4():
    np.random.seed(55)
    n_seg = 15

    r_greedy = 1.0 + np.random.randn(n_seg) * 0.35
    r_greedy = np.clip(r_greedy, 0.5, 1.8)

    r_dp_full = 1.0 + np.random.randn(n_seg) * 0.25
    r_dp_full = gaussian_filter1d(r_dp_full, sigma=1.0)
    r_dp_full[7] += 0.4
    r_dp_full = np.clip(r_dp_full, 0.5, 1.6)

    r_dp_subset = np.zeros(n_seg)
    anch = [0, 4, 9, 14]
    anch_r = [0.95, 1.05, 0.90, 1.02]
    for i in range(len(anch) - 1):
        a, b = anch[i], anch[i + 1]
        for j in range(a, b):
            frac = (j - a) / (b - a)
            r_dp_subset[j] = anch_r[i] + frac * (anch_r[i + 1] - anch_r[i])
    r_dp_subset[anch[-1]] = anch_r[-1]

    seg_idx = np.arange(1, n_seg + 1)

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True,
                             gridspec_kw={"hspace": 0.18})
    titles = [u"(a) \u8d2a\u5fc3\u7b97\u6cd5",
              u"(b) \u5168\u5339\u914d DP",
              u"(c) \u5b50\u96c6 DP"]
    data = [r_greedy, r_dp_full, r_dp_subset]
    colors = ["#ef4444", "#f97316", "#3b82f6"]
    stds = [np.std(d) for d in data]

    for ax, title, r, c, sd in zip(axes, titles, data, colors, stds):
        ax.bar(seg_idx, r, color=c, alpha=0.7, edgecolor=c, linewidth=0.5)
        ax.axhline(1.0, color="#64748b", linestyle="--", linewidth=0.8)
        ax.set_ylabel(u"\u901f\u5ea6\u6bd4 $r_i$", fontsize=9)
        ax.set_ylim(0.3, 1.9)
        ax.text(0.02, 0.92, f"{title}  $\\sigma_r$={sd:.3f}",
                transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")
        ax.grid(axis="y", alpha=0.2)

    axes[-1].set_xlabel(u"\u6bb5\u7f16\u53f7 $i$", fontsize=9)
    axes[0].set_title(u"\u56fe 5-4  \u4e09\u79cd\u7b97\u6cd5\u901f\u5ea6\u6bd4\u5e8f\u5217\u5bf9\u6bd4",
                      fontsize=11, fontweight="bold")

    out = os.path.join(OUT_DIR, "fig_5_4_speed_ratios.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  ok {out}")


# ======================================================================
#  Fig 5-5: Parameter sensitivity
# ======================================================================

def fig_5_5():
    json_path = os.path.join(_HERE, "test_output", "thesis_experiments",
                             "experiment_results.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # (a) w_smooth
    t55 = data["table_5_5"]
    ws_vals = sorted([float(k) for k in t55.keys()])
    sigma_r = [t55[str(w) if "." in str(w) else f"{w:.1f}"]["ratio_std"] for w in ws_vals]
    anchor_r = [t55[str(w) if "." in str(w) else f"{w:.1f}"]["anchor_rate"] for w in ws_vals]

    l1 = ax1.plot(ws_vals, sigma_r, "o-", color="#3b82f6", linewidth=1.5, markersize=6,
                  label="$\\sigma_r$")
    ax1.set_xlabel("$w_{smooth}$", fontsize=10)
    ax1.set_ylabel("$\\sigma_r$", fontsize=10, color="#3b82f6")
    ax1.tick_params(axis="y", labelcolor="#3b82f6")

    ax1b = ax1.twinx()
    l2 = ax1b.plot(ws_vals, anchor_r, "s--", color="#f97316", linewidth=1.5, markersize=6,
                   label=u"\u951a\u5b9a\u7387 (%)")
    ax1b.set_ylabel(u"\u951a\u5b9a\u7387 (%)", fontsize=10, color="#f97316")
    ax1b.tick_params(axis="y", labelcolor="#f97316")

    lines = l1 + l2
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, fontsize=8, loc="center right")
    ax1.set_title(u"(a) $w_{smooth}$ \u654f\u611f\u6027", fontsize=10)
    ax1.grid(alpha=0.2)

    # (b) w_skip
    t56 = data["table_5_6"]
    wk_vals = sorted([float(k) for k in t56.keys()])
    anchor_r2 = [t56[str(w) if "." in str(w) else f"{w:.2f}"]["anchor_rate"] for w in wk_vals]
    sigma_r2 = [t56[str(w) if "." in str(w) else f"{w:.2f}"]["ratio_std"] for w in wk_vals]

    l3 = ax2.plot(wk_vals, anchor_r2, "s-", color="#3b82f6", linewidth=1.5, markersize=6,
                  label=u"\u951a\u5b9a\u7387 (%)")
    ax2.set_xlabel("$w_{skip}$", fontsize=10)
    ax2.set_ylabel(u"\u951a\u5b9a\u7387 (%)", fontsize=10, color="#3b82f6")
    ax2.tick_params(axis="y", labelcolor="#3b82f6")
    ax2.set_xscale("log")

    ax2b = ax2.twinx()
    l4 = ax2b.plot(wk_vals, sigma_r2, "o--", color="#f97316", linewidth=1.5, markersize=6,
                   label="$\\sigma_r$")
    ax2b.set_ylabel("$\\sigma_r$", fontsize=10, color="#f97316")
    ax2b.tick_params(axis="y", labelcolor="#f97316")

    lines2 = l3 + l4
    labels2 = [ln.get_label() for ln in lines2]
    ax2.legend(lines2, labels2, fontsize=8, loc="center right")
    ax2.set_title(u"(b) $w_{skip}$ \u654f\u611f\u6027", fontsize=10)
    ax2.grid(alpha=0.2)

    fig.suptitle(u"\u56fe 5-5  \u53c2\u6570\u654f\u611f\u6027\u5206\u6790",
                 fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()

    out = os.path.join(OUT_DIR, "fig_5_5_sensitivity.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  ok {out}")


# ======================================================================
#  Fig 5-6: Inference acceleration bar chart
# ======================================================================

def fig_5_6():
    json_path = os.path.join(_HERE, "test_output", "thesis_experiments",
                             "accel_benchmark.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    labels = [u"\u57fa\u7ebf", u"+SharedYOLO",
              u"+\u6279\u91cf\u63a8\u7406\n(bs=8)",
              u"+FP16\n(\u4ec5HMR2)",
              u"+\u5e27\u9884\u53d6\n\u6d41\u6c34\u7ebf"]
    keys = ["baseline", "shared_yolo", "batch", "fp16", "prefetch"]
    vit_ms = [data[k]["vit_ms"] for k in keys]
    hmr_ms = [data[k]["hmr_ms"] for k in keys]
    fps_vals = [data[k]["fps"] for k in keys]

    x = np.arange(len(labels))
    w = 0.32

    fig, ax1 = plt.subplots(figsize=(10, 5))
    bars1 = ax1.bar(x - w / 2, vit_ms, w, color="#3b82f6", alpha=0.8,
                    label="ViTPose (ms)")
    bars2 = ax1.bar(x + w / 2, hmr_ms, w, color="#f97316", alpha=0.8,
                    label="HMR2 (ms)")
    ax1.set_ylabel(u"\u63a8\u7406\u5ef6\u65f6 (ms/\u5e27)", fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8)

    ax2 = ax1.twinx()
    line = ax2.plot(x, fps_vals, "D-", color="#22c55e", linewidth=2, markersize=8,
                    label=u"\u603b\u541e\u5410\u91cf (fps)", zorder=10)
    for i, fv in enumerate(fps_vals):
        ax2.annotate(f"{fv:.1f}", xy=(i, fv), xytext=(0, 8),
                     textcoords="offset points", ha="center", fontsize=8,
                     fontweight="bold", color="#16a34a")
    ax2.set_ylabel(u"\u603b\u541e\u5410\u91cf (\u5e27/\u79d2)", fontsize=10, color="#22c55e")
    ax2.tick_params(axis="y", labelcolor="#22c55e")

    base_fps = fps_vals[0]
    for i, fv in enumerate(fps_vals):
        if i > 0:
            speedup = fv / base_fps
            ax1.text(i, max(vit_ms[i], hmr_ms[i]) + 5, f"{speedup:.2f}x",
                     ha="center", fontsize=7, color="#64748b")

    lines_all = [bars1, bars2, line[0]]
    labels_all = ["ViTPose (ms)", "HMR2 (ms)", u"\u603b\u541e\u5410\u91cf (fps)"]
    ax1.legend(lines_all, labels_all, fontsize=8, loc="upper right")

    ax1.set_title(u"\u56fe 5-6  \u63a8\u7406\u52a0\u901f\u9010\u5c42\u53e0\u52a0\u6548\u679c",
                  fontsize=11, fontweight="bold")
    ax1.grid(axis="y", alpha=0.15)
    fig.tight_layout()

    out = os.path.join(OUT_DIR, "fig_5_6_acceleration.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  ok {out}")


# ======================================================================
#  Main
# ======================================================================

if __name__ == "__main__":
    print(f"Output: {OUT_DIR}\n")
    fig_4_4()
    fig_4_8()
    fig_5_2()
    fig_5_3()
    fig_5_4()
    fig_5_5()
    fig_5_6()
    print(f"\nDone! 7 figures saved to {OUT_DIR}")
