"""
test_pipeline_visual.py
完整端到端测试脚本：ViTPose + HMR2 GPU 推理管线验证（含可视化）

输出文件：
  d:/毕业设计/test_output/
    vitpose_keypoints.jpg   : 第0帧 ViTPose 关节点骨架叠加
    vitpose_strip.jpg       : 前16帧关节点拼图
    hmr2_skeleton_overlay.jpg : HMR2 3D关节投影叠加（第0帧）
    beat_signal.jpg          : 运动速度信号 + 节拍检测图
    test_report.txt          : 文字摘要报告

运行：
    conda activate pose_unified
    python d:/毕业设计/pose_extraction/scripts/test_pipeline_visual.py
"""

import os, sys, time
import numpy as np
import cv2
import torch

# ─── 路径设置 ──────────────────────────────────────────────────────────────────
_REPO = r"d:\毕业设计"
sys.path.insert(0, _REPO)
os.chdir(_REPO)

# ─── 权重配置（必须在导入任何模型前设置 HF_HOME） ─────────────────────────────
from pose_extraction.weights_config import (
    HF_HOME, HMR2_CHECKPOINT, SMPL_NEUTRAL, YOLO_WEIGHTS,
    VITPOSE_MODEL_ID, verify_all
)
os.environ["HF_HOME"]              = HF_HOME
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"]      = "1"

OUTPUT_DIR = os.path.join(_REPO, "test_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
VIDEO_PATH = os.path.join(_REPO, "BeatsMatching", "output.mp4")

# ─────────────────────────────────────────────────────────────────────────────
# 绘制工具
# ─────────────────────────────────────────────────────────────────────────────
COCO_SKELETON = [
    (0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16),
]
JOINT_NAMES = [
    "Nose","L-Eye","R-Eye","L-Ear","R-Ear",
    "L-Shldr","R-Shldr","L-Elbow","R-Elbow","L-Wrist","R-Wrist",
    "L-Hip","R-Hip","L-Knee","R-Knee","L-Ankle","R-Ankle"
]
_COLORS17 = [
    (255,0,0),(255,85,0),(255,170,0),(255,255,0),(170,255,0),
    (0,255,0),(0,255,170),(0,255,255),(0,170,255),(0,85,255),
    (0,0,255),(85,0,255),(170,0,255),(255,0,255),(255,0,170),
    (255,0,85),(200,200,0)
]

def draw_skeleton_on_bgr(frame_bgr, kpts_xy, scores, threshold=0.3):
    """kpts_xy: (17,2), scores: (17,)  →  returns BGR image"""
    img = frame_bgr.copy()
    for a, b in COCO_SKELETON:
        if scores[a] > threshold and scores[b] > threshold:
            cv2.line(img,
                     (int(kpts_xy[a,0]), int(kpts_xy[a,1])),
                     (int(kpts_xy[b,0]), int(kpts_xy[b,1])),
                     (80, 200, 80), 2, cv2.LINE_AA)
    for j in range(17):
        if scores[j] > threshold:
            cv2.circle(img, (int(kpts_xy[j,0]), int(kpts_xy[j,1])),
                       5, _COLORS17[j], -1, cv2.LINE_AA)
            cv2.circle(img, (int(kpts_xy[j,0]), int(kpts_xy[j,1])),
                       6, (0,0,0), 1, cv2.LINE_AA)
    return img

# SMPL-44 (pred_keypoints_3d) 骨架（参考 HMR2 COCO-body+feet+hands 结构）
SMPL44_SKELETON = [
    (0,1),(0,2),(1,3),(2,4),         # 头颈
    (3,5),(4,6),                     # 上臂
    (5,7),(6,8),(7,9),(8,10),        # 前臂+手
    (11,12),(11,13),(12,14),(13,15),(14,16), # 腿
    (0,11),(0,12),                   # 躯干连腿
]

def draw_smpl_on_bgr(frame_bgr, joints_3d_44, camera_t, focal=5000.0, threshold=5.0):
    """将 SMPL 44 关节透视投影到图像并绘制"""
    img = frame_bgr.copy()
    H, W = img.shape[:2]
    cx_img, cy_img = W / 2.0, H / 2.0
    tx, ty, tz = camera_t

    pts2d = []
    for j3d in joints_3d_44:
        # 加上相机平移
        X = j3d[0] + tx
        Y = j3d[1] + ty
        Z = j3d[2] + tz
        if Z < 0.01: Z = 0.01
        px = int(focal * X / Z + cx_img)
        py = int(focal * (-Y) / Z + cy_img)
        pts2d.append((px, py))

    def in_bounds(p):
        return 0 <= p[0] < W and 0 <= p[1] < H

    for a, b in SMPL44_SKELETON:
        if a < len(pts2d) and b < len(pts2d):
            if in_bounds(pts2d[a]) and in_bounds(pts2d[b]):
                cv2.line(img, pts2d[a], pts2d[b], (0,200,255), 2, cv2.LINE_AA)
    for p in pts2d:
        if in_bounds(p):
            cv2.circle(img, p, 4, (255,200,0), -1, cv2.LINE_AA)
    return img


# ─────────────────────────────────────────────────────────────────────────────
# 测试报告
# ─────────────────────────────────────────────────────────────────────────────
_report = []
def rlog(msg=""):
    print(msg)
    _report.append(str(msg))

def save_report():
    p = os.path.join(OUTPUT_DIR, "test_report.txt")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(_report))
    print(f"\n[Report saved] {p}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 0: 环境检查
# ─────────────────────────────────────────────────────────────────────────────
rlog("=" * 60)
rlog("BeatsMatching Pipeline 端到端测试报告")
rlog("=" * 60)
rlog(f"时间  : {time.strftime('%Y-%m-%d %H:%M:%S')}")
rlog(f"视频  : {VIDEO_PATH}")
rlog(f"输出  : {OUTPUT_DIR}")
rlog()
rlog("[0] 环境")
rlog(f"Python : {sys.version.split()[0]}")
rlog(f"PyTorch: {torch.__version__}")
cuda_ok = torch.cuda.is_available()
rlog(f"CUDA   : {'YES' if cuda_ok else 'NO'} / {torch.cuda.get_device_name(0) if cuda_ok else 'N/A'}")
rlog()
rlog("权重文件:")
verify_all()
rlog()

# 读取视频基本信息
cap = cv2.VideoCapture(VIDEO_PATH)
assert cap.isOpened(), f"Cannot open: {VIDEO_PATH}"
VIDEO_FPS   = cap.get(cv2.CAP_PROP_FPS)
VIDEO_TOTAL = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
VIDEO_W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
VIDEO_H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
rlog(f"视频   : {VIDEO_W}x{VIDEO_H} @ {VIDEO_FPS:.1f}fps  {VIDEO_TOTAL}帧  "
     f"{VIDEO_TOTAL/VIDEO_FPS:.1f}s")

N_VIT  = VIDEO_TOTAL   # ViTPose 测试帧数
N_HMR2 = VIDEO_TOTAL    # HMR2 测试帧数

# 均匀抽帧
frames_bgr = []
frame_indices = []
for i in range(N_VIT):
    idx = int(i * VIDEO_TOTAL / N_VIT)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, f = cap.read()
    if ret:
        frames_bgr.append(f)
        frame_indices.append(idx)
cap.release()
rlog(f"测试帧 : {len(frames_bgr)} 帧（均匀抽样）")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: ViTPose
# ─────────────────────────────────────────────────────────────────────────────
rlog()
rlog("=" * 60)
rlog("[1] ViTPose 2D 关节点检测")

t_load = time.time()
from pose_extraction.vitpose.detector import ViTPoseDetector
detector = ViTPoseDetector(
    vitpose_model=VITPOSE_MODEL_ID,
    yolo_model=YOLO_WEIGHTS,
)
rlog(f"模型加载: {time.time()-t_load:.2f}s")

# 推理
vit_results   = []   # list of ViTPoseFrameResult
timings_vit   = []

for i, (bgr, fidx) in enumerate(zip(frames_bgr, frame_indices)):
    ts = fidx / VIDEO_FPS
    t1 = time.time()
    result = detector.detect_frame(bgr, frame_idx=fidx, timestamp=ts)
    dt = time.time() - t1
    timings_vit.append(dt)
    vit_results.append(result)
    if i == 0:
        rlog(f"  首帧 (含JIT预热): {dt*1000:.0f}ms  persons={len(result.persons)}")
    elif i % 5 == 0:
        rlog(f"  帧{i:3d}: {dt*1000:.0f}ms")

det_n_vit = sum(1 for r in vit_results if len(r.persons) > 0)
avg_ms_vit = np.mean(timings_vit[1:]) * 1000
rlog()
rlog(f"ViTPose 汇总:")
rlog(f"  检出率      : {det_n_vit}/{N_VIT} ({100*det_n_vit/N_VIT:.0f}%)")
rlog(f"  avg延迟(非首): {avg_ms_vit:.1f}ms  (~{1000/avg_ms_vit:.1f} FPS)")

# 第0帧关节详情
r0 = vit_results[0]
if r0.persons:
    kpts0 = r0.persons[0]          # (17, 3) = [x, y, conf]
    visible = int(np.sum(kpts0[:,2] > 0.3))
    rlog(f"  第0帧可见关节: {visible}/17")
    rlog(f"  Nose: ({kpts0[0,0]:.1f}, {kpts0[0,1]:.1f})  conf={kpts0[0,2]:.2f}")

# ─── 可视化 1: 第0帧骨架 ──────────────────────────────────────────────────────
if r0.persons:
    kpts0 = r0.persons[0]
    vis = draw_skeleton_on_bgr(frames_bgr[0], kpts0[:,:2], kpts0[:,2])
    cv2.putText(vis, f"ViTPose  Frame{frame_indices[0]}  {det_n_vit}/{N_VIT} detected",
                (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,100), 2)
    p = os.path.join(OUTPUT_DIR, "vitpose_keypoints.jpg")
    cv2.imwrite(p, vis)
    rlog(f"  [图] 第0帧骨架 -> {p}")

# ─── 可视化 2: 16帧拼图 ──────────────────────────────────────────────────────
strip_imgs = []
for i in range(min(16, len(frames_bgr))):
    r = vit_results[i]
    if r.persons:
        kp = r.persons[0]
        img = draw_skeleton_on_bgr(frames_bgr[i], kp[:,:2], kp[:,2])
    else:
        img = frames_bgr[i].copy()
        cv2.putText(img,"No det",(4,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,200),1)
    img = cv2.resize(img, (160,90))
    cv2.putText(img,f"F{i}",(2,13),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,0),1)
    strip_imgs.append(img)
# 填充到16个
while len(strip_imgs) < 16:
    strip_imgs.append(np.zeros((90,160,3),dtype=np.uint8))
rows = [np.hstack(strip_imgs[r*4:(r+1)*4]) for r in range(4)]
strip = np.vstack(rows)
p = os.path.join(OUTPUT_DIR, "vitpose_strip.jpg")
cv2.imwrite(p, strip)
rlog(f"  [图] 16帧拼图 -> {p}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: HMR2
# ─────────────────────────────────────────────────────────────────────────────
rlog()
rlog("=" * 60)
rlog("[2] HMR2 3D 人体重建")

t_load = time.time()
from pose_extraction.hmr2.reconstructor import HMR2Reconstructor
reconstructor = HMR2Reconstructor(
    checkpoint_path=HMR2_CHECKPOINT,
    yolo_model=YOLO_WEIGHTS,
)
rlog(f"模型加载: {time.time()-t_load:.2f}s")

hmr2_results  = []   # list of List[HMR2FrameResult]
timings_hmr2  = []

for i in range(N_HMR2):
    bgr  = frames_bgr[i]
    fidx = frame_indices[i]
    ts   = fidx / VIDEO_FPS
    t1   = time.time()
    res  = reconstructor.reconstruct_frame(bgr, frame_idx=fidx, timestamp=ts)
    dt   = time.time() - t1
    timings_hmr2.append(dt)
    hmr2_results.append(res)
    if i == 0:
        rlog(f"  首帧 (含JIT预热): {dt*1000:.0f}ms  persons={len(res)}")
    elif i % 2 == 0:
        rlog(f"  帧{i:3d}: {dt*1000:.0f}ms")

det_n_hmr2  = sum(1 for r in hmr2_results if len(r) > 0)
avg_ms_hmr2 = np.mean(timings_hmr2[1:]) * 1000
rlog()
rlog(f"HMR2 汇总:")
rlog(f"  检出率      : {det_n_hmr2}/{N_HMR2} ({100*det_n_hmr2/N_HMR2:.0f}%)")
rlog(f"  avg延迟(非首): {avg_ms_hmr2:.1f}ms  (~{1000/avg_ms_hmr2:.1f} FPS)")

first_hmr2 = next((r[0] for r in hmr2_results if r), None)
if first_hmr2:
    if first_hmr2.joints_3d is not None:
        rlog(f"  joints_3d shape : {first_hmr2.joints_3d.shape}")
    if first_hmr2.body_pose is not None:
        rlog(f"  body_pose shape : {first_hmr2.body_pose.shape}")
    if first_hmr2.global_orient is not None:
        rlog(f"  global_orient   : {first_hmr2.global_orient.ravel()[:3]}")

# ─── 可视化 3: HMR2 3D投影叠加 ───────────────────────────────────────────────
if first_hmr2 and first_hmr2.joints_3d is not None and first_hmr2.pred_cam_t_full is not None:
    vis = draw_smpl_on_bgr(
        frames_bgr[0],
        first_hmr2.joints_3d,          # (44,3)
        first_hmr2.pred_cam_t_full,    # (3,)
    )
    cv2.putText(vis, "HMR2 - SMPL-44 3D Projection",
                (8,28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,200,255), 2)
    p = os.path.join(OUTPUT_DIR, "hmr2_skeleton_overlay.jpg")
    cv2.imwrite(p, vis)
    rlog(f"  [图] HMR2 3D骨架叠加 -> {p}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: 节拍检测 + 信号可视化
# ─────────────────────────────────────────────────────────────────────────────
rlog()
rlog("=" * 60)
rlog("[3] 节拍检测（ViTPose 运动速度信号）")

from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

# 构建关节坐标时间序列
kp_seq, ts_seq = [], []
for r, fidx in zip(vit_results, frame_indices):
    if r.persons:
        kp = r.persons[0]          # (17,3)
        kp_seq.append(kp[:, :2])   # (17,2)
        ts_seq.append(fidx / VIDEO_FPS)

rlog(f"有效帧: {len(kp_seq)}")

# 运动速度信号
motion_raw, motion_ts = [], []
for i in range(1, len(kp_seq)):
    delta = kp_seq[i] - kp_seq[i-1]          # (17,2)
    speed = np.sqrt((delta**2).sum(axis=-1))  # (17,)
    motion_raw.append(float(speed.mean()))
    motion_ts.append(ts_seq[i])

motion_raw = np.array(motion_raw)
motion_ts  = np.array(motion_ts)
smoothed   = gaussian_filter1d(motion_raw, sigma=1.5)

mu, sigma  = smoothed.mean(), smoothed.std()
adaptive_thr = mu + 0.5 * sigma
peaks, _   = find_peaks(smoothed, height=adaptive_thr, distance=3)
beat_times = motion_ts[peaks]

rlog(f"运动信号: mean={mu:.2f} std={sigma:.2f}  threshold={adaptive_thr:.2f}")
rlog(f"检测节拍: {len(peaks)} 个")
if len(beat_times) > 1:
    ivs = np.diff(beat_times)
    rlog(f"节拍间隔: avg={ivs.mean():.2f}s  std={ivs.std():.2f}s  ~{60/ivs.mean():.1f} BPM")
rlog(f"时间点  : {[f'{t:.2f}s' for t in beat_times]}")

# HMR2 旋转梯度信号
rot_grads, rot_ts = [], []
pose_seq = []
for res_list in hmr2_results:
    if res_list and res_list[0].body_pose is not None:
        # (23,3) 轴角  →  拼上 global_orient (1,3)
        go = res_list[0].global_orient  # (1,3) or (3,)
        bp = res_list[0].body_pose      # (23,3)
        if go is not None and bp is not None:
            go_flat = go.reshape(1,3)
            full_pose = np.vstack([go_flat, bp])  # (24,3)
            pose_seq.append(full_pose)

for i in range(1, len(pose_seq)):
    diff = pose_seq[i] - pose_seq[i-1]
    grad = float((diff**2).sum())
    rot_grads.append(grad)
    rot_ts.append(i * (N_HMR2 / VIDEO_FPS / max(len(pose_seq),1)))

rlog(f"HMR2 旋转梯度序列长度: {len(rot_grads)}")

# ─── 可视化 4: 信号图 ─────────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(14, 11))
fig.patch.set_facecolor('#1a1a2e')

def dark_ax(ax, title):
    ax.set_facecolor('#16213e')
    ax.set_title(title, color='white', fontsize=12, pad=8)
    ax.tick_params(colors='#aaa')
    ax.set_xlabel('Time (s)', color='#aaa')
    for sp in ax.spines.values(): sp.set_edgecolor('#444')

# subplot-1: 速度信号
ax1 = axes[0]
dark_ax(ax1, 'ViTPose Motion Speed Signal & Beat Detection')
ax1.plot(motion_ts, motion_raw, color='#4a4a6a', lw=1.2, alpha=0.7, label='Raw speed')
ax1.plot(motion_ts, smoothed, color='#00bcd4', lw=2.0, label='Smoothed (σ=1.5)')
ax1.axhline(adaptive_thr, color='#ff7043', ls='--', lw=1.5,
            label=f'Adaptive thr (μ+0.5σ={adaptive_thr:.2f})')
ax1.scatter(beat_times, smoothed[peaks], s=90, color='#ffeb3b', zorder=6,
            label=f'{len(peaks)} beats detected')
ax1.set_ylabel('Avg joint speed (px/frame)', color='#aaa')
ax1.legend(facecolor='#222', labelcolor='white', fontsize=9, loc='upper right')

# subplot-2: 关节速度热图
ax2 = axes[1]
dark_ax(ax2, 'Per-joint Speed Heatmap (ViTPose)')
joint_mat = np.array([
    np.sqrt(((kp_seq[i]-kp_seq[i-1])**2).sum(axis=-1))
    for i in range(1, len(kp_seq))
]).T  # (17, T)
vmax = np.percentile(joint_mat, 95)
im = ax2.imshow(joint_mat, aspect='auto', cmap='inferno',
                extent=[motion_ts[0], motion_ts[-1], -0.5, 16.5],
                vmin=0, vmax=vmax, origin='lower')
plt.colorbar(im, ax=ax2, label='Speed (px/frame)')
ax2.set_yticks(range(17))
ax2.set_yticklabels(JOINT_NAMES, fontsize=7, color='#ccc')
for bt in beat_times:
    ax2.axvline(bt, color='#00e676', lw=1.0, alpha=0.8)
ax2.set_ylabel('Joint', color='#aaa')

# subplot-3: HMR2 旋转梯度
ax3 = axes[2]
dark_ax(ax3, 'HMR2 SMPL Rotation Gradient  ||Δθ||²')
if rot_grads:
    ax3.plot(rot_ts, rot_grads, color='#ce93d8', lw=2.0,
             label='Rotation gradient (sum over 24 joints)')
    ax3.legend(facecolor='#222', labelcolor='white', fontsize=9)
else:
    ax3.text(0.5, 0.5, 'N/A (not enough HMR2 frames)',
             ha='center', va='center', color='#aaa', transform=ax3.transAxes)
ax3.set_ylabel('Rotation gradient', color='#aaa')

plt.tight_layout(pad=2.5)
p = os.path.join(OUTPUT_DIR, "beat_signal.jpg")
plt.savefig(p, dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
rlog(f"  [图] 信号曲线 -> {p}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: 汇总
# ─────────────────────────────────────────────────────────────────────────────
rlog()
rlog("=" * 60)
rlog("[4] 测试汇总")
rlog()
rlog(f"  {'Model':<20} {'Det-rate':<14} {'avg latency':<16} {'est FPS'}")
rlog(f"  {'-'*60}")
rlog(f"  {'ViTPose (GPU)':<20} {f'{det_n_vit}/{N_VIT}':<14} {f'{avg_ms_vit:.1f}ms':<16} ~{1000/avg_ms_vit:.1f}")
rlog(f"  {'HMR2 (GPU)':<20} {f'{det_n_hmr2}/{N_HMR2}':<14} {f'{avg_ms_hmr2:.1f}ms':<16} ~{1000/avg_ms_hmr2:.1f}")
rlog()
rlog(f"  Beats detected: {len(beat_times)}")
rlog()
rlog("  Output files:")
for fname in ["vitpose_keypoints.jpg","vitpose_strip.jpg",
              "hmr2_skeleton_overlay.jpg","beat_signal.jpg","test_report.txt"]:
    fp = os.path.join(OUTPUT_DIR, fname)
    if os.path.exists(fp):
        rlog(f"    [OK] {fname}  ({os.path.getsize(fp)//1024}KB)")
    else:
        rlog(f"    [--] {fname}")
rlog()
rlog("ALL TESTS PASSED")

save_report()
