"""
分析 ViTPose vs HMR2 的相关性、周期性，以及不同平滑程度对峰值提取的影响。
直接复用上次 test_algorithm_pipeline 留下的 npz 缓存或重新推理。
"""
import sys, os, time, glob
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "test_output", "algorithm_test")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def log(msg):
    print(msg)

# ═══════════ 加载 / 推理数据 ═══════════
# 尝试从缓存 npz 加载（之前 test_algorithm_pipeline 跑过的）
cache = os.path.join(OUTPUT_DIR, "cached_signals.npz")
if os.path.exists(cache):
    d = np.load(cache, allow_pickle=True)
    vel_smooth = d["vel_smooth"]
    rot_smooth = d["rot_smooth"]
    vel_raw = d.get("vel_raw", None)
    rot_raw = d.get("rot_raw", None)
    fps = float(d["fps"])
    log(f"从缓存加载: vel={vel_smooth.shape}, rot={rot_smooth.shape}, fps={fps}")
else:
    # 需要跑完整推理 — 仅在没缓存时
    log("未找到缓存，运行完整推理...")
    # 避免直接 import PySide6 依赖的模块
    import cv2
    
    # 找视频
    vids = glob.glob("*.mp4") + glob.glob("*.avi") + glob.glob("resources/*.mp4")
    video_path = vids[0] if vids else "test1.mp4"
    log(f"视频: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    log(f"读取 {len(frames)} 帧, fps={fps}")
    
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pose_extraction'))
    from vitpose.detector import ViTPoseDetector
    det = ViTPoseDetector()
    coords_list = []
    for f in frames:
        kp, sc = det.detect_single(f)
        coords_list.append(kp)
    coords = np.array(coords_list)
    
    from hmr2.video_processor import HMR2VideoProcessor
    h = HMR2VideoProcessor()
    rot_list = h.process_frames(frames)
    rotations = np.array(rot_list)
    
    # 平滑
    from pose_extraction.core.smoothing import SmoothingFilter
    smoother = SmoothingFilter()
    coords_clean = smoother.smooth_coordinates(coords.copy())
    rotations_clean = smoother.smooth_rotations(rotations.copy())
    
    # 特征计算 — 直接用 numpy
    # weighted velocity
    JOINT_WEIGHTS = np.array([0.5,0.3,0.3,0.3,0.3, 1.0,1.0,0.8,0.8,1.2,1.2, 1.0,1.0,0.8,0.8,1.2,1.2])
    def weighted_vel(c):
        diff = np.diff(c, axis=0)
        speed = np.sqrt(np.sum(diff**2, axis=-1))
        return np.concatenate([[0], np.average(speed, axis=1, weights=JOINT_WEIGHTS)])
    def rot_metric(r):
        diff = np.diff(r[:, 1:, :], axis=0)
        return np.concatenate([[0], np.sqrt(np.sum(diff**2, axis=(1,2)))])
    
    vel_raw = weighted_vel(coords)
    vel_smooth = weighted_vel(coords_clean)
    rot_raw = rot_metric(rotations)
    rot_smooth = rot_metric(rotations_clean)
    
    np.savez(cache, vel_smooth=vel_smooth, rot_smooth=rot_smooth,
             vel_raw=vel_raw, rot_raw=rot_raw, fps=fps)
    log("已缓存信号数据")

# ═══════════ 2. 相关性分析 ═══════════
log("\n" + "=" * 70)
log("2. ViTPose vs HMR2 相关性")
log("=" * 70)

min_len = min(len(vel_smooth), len(rot_smooth))
v = vel_smooth[:min_len]
r = rot_smooth[:min_len]
# Pearson 相关系数
from scipy.stats import pearsonr, spearmanr
pear_r, pear_p = pearsonr(v, r)
spear_r, spear_p = spearmanr(v, r)
log(f"  Pearson  r = {pear_r:.4f}  (p = {pear_p:.2e})")
log(f"  Spearman ρ = {spear_r:.4f}  (p = {spear_p:.2e})")

# 互相关 (cross-correlation) 找最佳滞后
from scipy.signal import correlate
v_z = (v - v.mean()) / (v.std() + 1e-8)
r_z = (r - r.mean()) / (r.std() + 1e-8)
xcorr = correlate(v_z, r_z, mode='full') / len(v_z)
lags = np.arange(-len(v_z)+1, len(v_z))
best_lag = lags[np.argmax(xcorr)]
best_xcorr = xcorr.max()
log(f"  互相关最大值 = {best_xcorr:.4f}  @ lag = {best_lag} 帧 ({best_lag/fps:.3f}s)")

# ═══════════ 3. 周期性分析 (FFT) ═══════════
log("\n" + "=" * 70)
log("3. 周期性分析 (FFT)")
log("=" * 70)

from scipy.fft import rfft, rfftfreq

for name, sig in [("vel_smooth", v), ("rot_smooth", r)]:
    N = len(sig)
    sig_centered = sig - sig.mean()
    # Hanning 窗减少频谱泄漏
    window = np.hanning(N)
    spectrum = np.abs(rfft(sig_centered * window))
    freqs = rfftfreq(N, d=1.0/fps)
    
    # 跳过 DC (index 0)，找前5个主导频率
    spectrum[0] = 0
    top5_idx = np.argsort(spectrum)[-5:][::-1]
    log(f"  [{name}] 主导频率 (Hz) / 周期 (s):")
    for idx in top5_idx:
        f_hz = freqs[idx]
        period = 1.0 / f_hz if f_hz > 0 else float('inf')
        log(f"    {f_hz:.2f} Hz  →  周期 {period:.2f}s  (幅值 {spectrum[idx]:.2f})")

# ═══════════ 4. 不同 sigma 对峰值数量的影响 ═══════════
log("\n" + "=" * 70)
log("4. 不同 gaussian_sigma 对峰值检测的影响")
log("=" * 70)

from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

# 先对融合信号进行分析
v_norm = np.clip((v - np.percentile(v,2)) / (np.percentile(v,98) - np.percentile(v,2) + 1e-8), 0, 1)
r_norm = np.clip((r - np.percentile(r,2)) / (np.percentile(r,98) - np.percentile(r,2) + 1e-8), 0, 1)
alpha = 0.6
fused_raw = alpha * v_norm + (1 - alpha) * r_norm

sigmas = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0]
log(f"  融合信号长度: {len(fused_raw)}, fps={fps}")
log(f"  {'sigma':>6} | {'峰数':>5} | {'平均间距(帧)':>14} | {'平均间距(s)':>12} | {'信号std':>10}")
log(f"  {'-'*6}-+-{'-'*5}-+-{'-'*14}-+-{'-'*12}-+-{'-'*10}")

for sigma in sigmas:
    smoothed = gaussian_filter1d(fused_raw, sigma=sigma)
    # 自适应阈值
    h_threshold = np.mean(smoothed) + 0.3 * np.std(smoothed)
    min_dist = max(3, int(fps * 0.15))
    peaks, _ = find_peaks(smoothed, height=h_threshold, distance=min_dist)
    
    avg_gap = np.mean(np.diff(peaks)) if len(peaks) > 1 else 0
    avg_gap_s = avg_gap / fps if avg_gap > 0 else 0
    sig_std = np.std(smoothed)
    log(f"  {sigma:>6.1f} | {len(peaks):>5} | {avg_gap:>14.1f} | {avg_gap_s:>12.2f} | {sig_std:>10.4f}")

# ═══════════ 5. 峰谷对比：峰 vs 谷作为关键帧 ═══════════
log("\n" + "=" * 70)
log("5. 峰检测 vs 谷检测 vs 峰+谷")
log("=" * 70)

for sigma in [3.0, 5.0, 8.0]:
    smoothed = gaussian_filter1d(fused_raw, sigma=sigma)
    h_threshold = np.mean(smoothed) + 0.3 * np.std(smoothed)
    l_threshold = np.mean(smoothed) - 0.3 * np.std(smoothed)
    min_dist = max(3, int(fps * 0.15))
    
    peaks, _ = find_peaks(smoothed, height=h_threshold, distance=min_dist)
    valleys, _ = find_peaks(-smoothed, height=-l_threshold, distance=min_dist)
    combined = sorted(set(list(peaks) + list(valleys)))
    
    log(f"  σ={sigma:.0f}: 峰={len(peaks)}, 谷={len(valleys)}, 峰+谷={len(combined)}")

# ═══════════ 6. 绘图：不同 sigma 平滑对比 ═══════════
log("\n绘制对比图...")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

zh = [f.name for f in fm.fontManager.ttflist if 'SimHei' in f.name or 'Microsoft YaHei' in f.name]
if zh:
    plt.rcParams['font.sans-serif'] = [zh[0]]
plt.rcParams['axes.unicode_minus'] = False

t_axis = np.arange(len(fused_raw)) / fps

fig, axes = plt.subplots(5, 1, figsize=(18, 20), sharex=True)
fig.patch.set_facecolor('#1e1e1e')

test_sigmas = [0.75, 3.0, 5.0, 8.0, 12.0]
labels = ['σ=0.75 (当前细粒度)', 'σ=3.0 (当前粗粒度)', 'σ=5.0', 'σ=8.0', 'σ=12.0']

for i, (sigma, label) in enumerate(zip(test_sigmas, labels)):
    ax = axes[i]
    ax.set_facecolor('#2a2a2a')
    ax.tick_params(colors='white', labelsize=9)
    for spine in ax.spines.values():
        spine.set_color('#555')
    
    smoothed = gaussian_filter1d(fused_raw, sigma=sigma)
    
    # 原始信号 (浅灰)
    ax.plot(t_axis, fused_raw, color='#444', linewidth=0.3, alpha=0.5)
    # 平滑后
    ax.plot(t_axis, smoothed, color='#22c55e', linewidth=1.2, alpha=0.9)
    
    # 峰值
    h_threshold = np.mean(smoothed) + 0.3 * np.std(smoothed)
    min_dist = max(3, int(fps * 0.15))
    peaks, _ = find_peaks(smoothed, height=h_threshold, distance=min_dist)
    ax.scatter(peaks / fps, smoothed[peaks], color='#ff6b6b', s=30, zorder=5)
    
    # 谷值
    valleys, _ = find_peaks(-smoothed, distance=min_dist)
    ax.scatter(valleys / fps, smoothed[valleys], color='#5b8cff', s=15, zorder=5, marker='v')
    
    ax.set_title(f'{label}  →  峰: {len(peaks)}, 谷: {len(valleys)}',
                 color='white', fontsize=11)
    ax.set_ylabel('Signal', color='white')

axes[-1].set_xlabel('Time (s)', color='white')
plt.tight_layout()
path = os.path.join(OUTPUT_DIR, "sigma_comparison.png")
plt.savefig(path, dpi=150, facecolor='#1e1e1e')
plt.close()
log(f"  sigma 对比图已保存: {path}")

# ═══════════ 7. 互相关图 ═══════════
fig, axes = plt.subplots(2, 1, figsize=(16, 8))
fig.patch.set_facecolor('#1e1e1e')
for ax in axes:
    ax.set_facecolor('#2a2a2a')
    ax.tick_params(colors='white')
    for sp in ax.spines.values():
        sp.set_color('#555')

# 7a: 归一化信号叠加
axes[0].plot(t_axis, v_norm, color='#22c55e', linewidth=0.8, alpha=0.7, label='v_norm (ViTPose)')
axes[0].plot(t_axis, r_norm, color='#5b8cff', linewidth=0.8, alpha=0.7, label='r_norm (HMR2)')
axes[0].set_title(f'ViTPose vs HMR2 归一化信号  (Pearson r={pear_r:.3f}, Spearman ρ={spear_r:.3f})',
                  color='white', fontsize=11)
axes[0].legend(facecolor='#333', edgecolor='#555', labelcolor='white')
axes[0].set_ylabel('Normalized', color='white')

# 7b: 互相关
lag_range = 30  # ±30帧
mask = (lags >= -lag_range) & (lags <= lag_range)
axes[1].plot(lags[mask] / fps, xcorr[mask], color='#e879f9', linewidth=1.2)
axes[1].axvline(best_lag / fps, color='#ff6b6b', linewidth=1, linestyle='--',
                label=f'最佳滞后={best_lag}帧 ({best_lag/fps:.3f}s)')
axes[1].set_title('互相关函数 (Cross-Correlation)', color='white', fontsize=11)
axes[1].set_xlabel('Lag (s)', color='white')
axes[1].set_ylabel('Correlation', color='white')
axes[1].legend(facecolor='#333', edgecolor='#555', labelcolor='white')

plt.tight_layout()
path2 = os.path.join(OUTPUT_DIR, "correlation_analysis.png")
plt.savefig(path2, dpi=150, facecolor='#1e1e1e')
plt.close()
log(f"  相关性分析图已保存: {path2}")

# ═══════════ 8. FFT 频谱图 ═══════════
fig, axes = plt.subplots(2, 1, figsize=(14, 8))
fig.patch.set_facecolor('#1e1e1e')
for ax in axes:
    ax.set_facecolor('#2a2a2a')
    ax.tick_params(colors='white')
    for sp in ax.spines.values():
        sp.set_color('#555')

for i, (name, sig, color) in enumerate([
    ("ViTPose 速度", v, '#22c55e'), ("HMR2 旋转梯度", r, '#5b8cff')
]):
    N = len(sig)
    window = np.hanning(N)
    spectrum = np.abs(rfft((sig - sig.mean()) * window))
    freqs = rfftfreq(N, d=1.0/fps)
    spectrum[0] = 0
    
    axes[i].plot(freqs, spectrum, color=color, linewidth=0.8)
    axes[i].fill_between(freqs, 0, spectrum, color=color, alpha=0.2)
    axes[i].set_title(f'{name} 频谱', color='white', fontsize=11)
    axes[i].set_ylabel('Amplitude', color='white')
    axes[i].set_xlim(0, fps/2)
    
    # 标注主导频率
    top3 = np.argsort(spectrum)[-3:][::-1]
    for idx in top3:
        if freqs[idx] > 0.1:
            axes[i].annotate(f'{freqs[idx]:.1f}Hz\n({1/freqs[idx]:.2f}s)',
                           xy=(freqs[idx], spectrum[idx]),
                           fontsize=8, color='#ffb84d', ha='center', va='bottom')

axes[-1].set_xlabel('Frequency (Hz)', color='white')
plt.tight_layout()
path3 = os.path.join(OUTPUT_DIR, "fft_spectrum.png")
plt.savefig(path3, dpi=150, facecolor='#1e1e1e')
plt.close()
log(f"  FFT 频谱图已保存: {path3}")

log("\n分析完成。")
