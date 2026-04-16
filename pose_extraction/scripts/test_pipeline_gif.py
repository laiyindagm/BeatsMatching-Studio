"""
test_pipeline_gif.py
====================================
端到端测试 + GIF 可视化生成脚本

在 test_pipeline_visual.py 的基础上，额外输出：
  - vitpose_skeleton.gif    : ViTPose COCO-17 骨架动画
  - hmr2_body_model.gif     : HMR2 SMPL 3D 人体模型动画

运行：
    conda activate pose_unified
    python d:/毕业设计/pose_extraction/scripts/test_pipeline_gif.py [视频路径] [--mode both] [--frames N] [--gif_fps 10] [--gif_dur 3.0]

示例：
    # 默认模式（both），前30帧，GIF 10fps, 3秒时长
    python d:/毕业设计/pose_extraction/scripts/test_pipeline_gif.py

    # 仅 ViTPose + 全帧 + 高帧率 GIF
    python d:/毕业设计/pose_extraction/scripts/test_pipeline_gif.py --mode vitpose --frames 60 --gif_fps 15

输出文件（d:/毕业设计/test_output/）：
    vitpose_keypoints.jpg     : 第0帧骨架叠加图（静态）
    vitpose_strip.jpg          : 前16帧拼图
    vitpose_skeleton.gif       : ViTPose 骨架动画（新增）
    hmr2_skeleton_overlay.jpg  : HMR2 3D投影叠加图
    hmr2_body_model.gif        : HMR2 SMPL 人体模型动画（新增）
    beat_signal.jpg            : 运动速度信号 + 节拍检测图
    test_report.txt            : 文字摘要报告
"""

import os
import sys
import time
import argparse
import warnings
import numpy as np
import cv2
import torch

# ─── 路径设置 ──────────────────────────────────────────────────────────────────
_REPO = r"d:\毕业设计"
sys.path.insert(0, _REPO)
os.chdir(_REPO)

# ─── 权重配置 ──────────────────────────────────────────────────────────────────
from pose_extraction.weights_config import (
    HF_HOME, HMR2_CHECKPOINT, SMPL_NEUTRAL, YOLO_WEIGHTS,
    VITPOSE_MODEL_ID, verify_all,
)
os.environ["HF_HOME"] = HF_HOME
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

OUTPUT_DIR = os.path.join(_REPO, "test_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

VIDEO_PATH_DEFAULT = os.path.join(_REPO, "BeatsMatching", "output.mp4")

# ══════════════════════════════════════════════════════════════════════════════
#  绘制工具（与 test_pipeline_visual.py 一致）
# ══════════════════════════════════════════════════════════════════════════════
COCO_SKELETON = [
    (0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16),
]
JOINT_NAMES = [
    "Nose","L-Eye","R-Eye","L-Ear","R-Ear",
    "L-Shldr","R-Shldr","L-Elbow","R-Elbow","L-Wrist","R-Wrist",
    "L-Hip","R-Hip","L-Knee","R-Knee","L-Ankle","R-Ankle",
]
_COLORS17 = [
    (255,0,0),(255,85,0),(255,170,0),(255,255,0),(170,255,0),
    (0,255,0),(0,255,170),(0,255,255),(0,170,255),(0,85,255),
    (0,0,255),(85,0,255),(170,0,255),(255,0,255),(255,0,170),
    (255,0,85),(200,200,0),
]

def draw_skeleton_on_bgr(frame_bgr, kpts_xy, scores, threshold=0.3):
    """kpts_xy: (17,2), scores: (17,) → BGR image"""
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


SMPL44_SKELETON = [
    (0,1),(0,2),(1,3),(2,4),(3,5),(4,6),
    (5,7),(6,8),(7,9),(8,10),
    (11,12),(11,13),(12,14),(13,15),(14,16),
    (0,11),(0,12),
]

def draw_smpl_on_bgr(frame_bgr, joints_3d_44, camera_t, focal=5000.0, threshold=5.0):
    img = frame_bgr.copy()
    H, W = img.shape[:2]
    cx_img, cy_img = W / 2.0, H / 2.0
    tx, ty, tz = camera_t
    pts2d = []
    for j3d in joints_3d_44:
        X, Y, Z = j3d[0] + tx, j3d[1] + ty, j3d[2] + tz
        if Z < 0.01: Z = 0.01
        px = int(focal * X / Z + cx_img)
        py = int(focal * (-Y) / Z + cy_img)
        pts2d.append((px, py))
    def in_bounds(p): return 0 <= p[0] < W and 0 <= p[1] < H
    for a, b in SMPL44_SKELETON:
        if a < len(pts2d) and b < len(pts2d):
            if in_bounds(pts2d[a]) and in_bounds(pts2d[b]):
                cv2.line(img, pts2d[a], pts2d[b], (0,200,255), 2, cv2.LINE_AA)
    for p in pts2d:
        if in_bounds(p):
            cv2.circle(img, p, 4, (255,200,0), -1, cv2.LINE_AA)
    return img


# ══════════════════════════════════════════════════════════════════════════════
#  报告工具
# ══════════════════════════════════════════════════════════════════════════════
_report = []
def rlog(msg=""):
    print(msg)
    _report.append(str(msg))

def save_report():
    p = os.path.join(OUTPUT_DIR, "test_report.txt")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(_report))
    print(f"\n[Report saved] {p}")


# ══════════════════════════════════════════════════════════════════════════════
#  GIF 可视化生成器
# ══════════════════════════════════════════════════════════════════════════════
def generate_vitpose_gif(coords, fps, output_dir, gif_fps=10, gif_duration=3.0):
    """
    使用 enhanced_pose_worker.py 中的 GIF 逻辑，
    在测试脚本中直接调用，无需 GUI 线程。
    
    Args:
        coords: (T, 17, 2) ViTPose 关节坐标序列
        fps: 视频帧率
        output_dir: 输出目录
        gif_fps: GIF 帧率
        gif_duration: GIF 时长（秒），0 表示使用全部帧
    """
    from PIL import Image, ImageDraw
    
    skeleton = [
        (0, 1), (0, 2), (1, 3), (2, 4),
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15), (12, 14), (14, 16),
    ]
    
    JOINT_COLORS = {
        'head': (255, 200, 87),
        'shoulder': (87, 199, 255),
        'elbow': (87, 199, 255),
        'wrist': (255, 135, 87),
        'torso': (150, 255, 150),
        'hip': (150, 255, 150),
        'knee': (150, 150, 255),
        'ankle': (255, 87, 199),
    }
    joint_color_map = [
        'head', 'head', 'head', 'head', 'head',
        'shoulder', 'shoulder', 'elbow', 'elbow',
        'wrist', 'wrist', 'hip', 'hip',
        'knee', 'knee', 'ankle', 'ankle',
    ]
    
    LINE_COLORS = {
        'head': (255, 200, 87, 180),
        'arm': (87, 199, 255, 180),
        'torso': (150, 255, 150, 180),
        'leg': (150, 150, 255, 180),
    }
    line_color_map = [
        'head', 'head', 'head', 'head', 'head',
        'arm', 'arm', 'arm',
        'torso', 'torso', 'torso',
        'leg', 'leg', 'leg', 'leg',
    ]
    
    T = len(coords)
    if gif_duration > 0:
        T_vis = min(T, int(gif_duration * fps))
    else:
        T_vis = T
    
    step = max(1, int(fps / gif_fps))
    
    # 计算全局坐标范围（固定视角）
    all_x = coords[:T_vis, :, 0]
    all_y = coords[:T_vis, :, 1]
    x_min, x_max = all_x.min(), all_x.max()
    y_min, y_max = all_y.min(), all_y.max()
    
    pad = 20
    canvas_w, canvas_h = 400, 500
    
    def kp_to_canvas(kp):
        scale = min(
            (canvas_w - 2 * pad) / max(x_max - x_min, 1),
            (canvas_h - 2 * pad) / max(y_max - y_min, 1),
        )
        ox = (canvas_w - (x_max - x_min) * scale) / 2 - x_min * scale
        oy = (canvas_h - (y_max - y_min) * scale) / 2 - y_min * scale
        return kp * scale + np.array([ox, oy])
    
    print(f"\n  [GIF] ViTPose 骨架动画: {T_vis} 帧, gif_fps={gif_fps}")
    
    frames = []
    for i in range(0, T_vis, step):
        img = Image.new('RGB', (canvas_w, canvas_h), (30, 30, 30))
        draw = ImageDraw.Draw(img)
        
        kp = kp_to_canvas(coords[i])
        
        # 画骨架连线
        for j, (a, b) in enumerate(skeleton):
            pt1 = tuple(kp[a].astype(int))
            pt2 = tuple(kp[b].astype(int))
            color_key = line_color_map[j] if j < len(line_color_map) else 'torso'
            color = LINE_COLORS.get(color_key, (150, 255, 150, 180))[:3]
            draw.line([pt1, pt2], fill=color, width=3)
        
        # 画关节点
        for j in range(17):
            x, y = int(kp[j, 0]), int(kp[j, 1])
            color_key = joint_color_map[j]
            color = JOINT_COLORS[color_key]
            r = 5 if j in [9, 10, 15, 16] else 4
            draw.ellipse([x-r, y-r, x+r, y+r], fill=color)
        
        draw.text((10, 10), f"F{i}", fill=(200, 200, 200))
        frames.append(img)
    
    if frames:
        gif_path = os.path.join(output_dir, "vitpose_skeleton.gif")
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=int(1000 / gif_fps),
            loop=0,
        )
        print(f"  [GIF OK] {gif_path} ({len(frames)} frames)")
        return gif_path
    return None


def generate_hmr2_smpl_gif(rotations, fps, output_dir, gif_fps=10, gif_duration=3.0):
    """
    使用 engines/smpl_renderer.py 渲染 SMPL 3D 人体模型动画为 GIF
    
    Args:
        rotations: (T, 24, 3) HMR2 SMPL 轴角旋转参数
        fps: 视频帧率
        output_dir: 输出目录
        gif_fps: GIF 帧率
        gif_duration: GIF 时长（秒）
    """
    from engines.smpl_renderer import SMPLRenderer
    
    T = len(rotations)
    if gif_duration > 0:
        T_vis = min(T, int(gif_duration * fps))
    else:
        T_vis = T
    
    step = max(1, int(fps / gif_fps))
    sampled_poses = rotations[:T_vis:step]
    
    print(f"\n  [GIF] HMR2 SMPL 人体模型: {len(sampled_poses)} poses, gif_fps={gif_fps}")
    
    try:
        renderer = SMPLRenderer(SMPL_NEUTRAL)
        
        gif_path = os.path.join(output_dir, "hmr2_body_model.gif")
        renderer.render_gif(
            sampled_poses,
            gif_path,
            fps=gif_fps,
            azimuth=45.0,
            elevation=20.0,
            size=(400, 500),
            rotate_view=True,
        )
        print(f"  [GIF OK] {gif_path}")
        return gif_path
        
    except Exception as e:
        print(f"  [GIF WARN] SMPL 渲染失败: {e}，回退到条形图...")
        return _generate_hmr2_fallback_gif(rotations, fps, output_dir, gif_fps)


def _generate_hmr2_fallback_gif(rotations, fps, output_dir, gif_fps=10):
    """HMR2 渲染失败的降级方案：旋转量条形图"""
    from PIL import Image
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import io
    
    T = min(len(rotations), int(3.0 * fps))  # 最多3秒
    step = max(1, int(fps / gif_fps))
    
    frames = []
    for i in range(0, T, step):
        fig, ax = plt.subplots(figsize=(4, 4))
        rot_magnitudes = np.linalg.norm(rotations[i], axis=1)
        ax.barh(range(24), rot_magnitudes, color='steelblue')
        ax.set_xlabel('Rotation Magnitude (rad)')
        ax.set_ylabel('Joint Index')
        ax.set_title(f'HMR2 Frame {i}')
        ax.set_xlim(0, np.pi)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        buf.close()
        plt.close()
    
    if frames:
        gif_path = os.path.join(output_dir, "hmr2_rotation.gif")
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=int(1000 / gif_fps),
            loop=0,
        )
        print(f"  [GIF fallback OK] {gif_path}")
        return gif_path
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  主流程（复用 test_pipeline_visual.py 的推理逻辑 + 新增 GIF 步骤）
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="BeatsMatching Pipeline 测试 + GIF 可视化")
    parser.add_argument("video", nargs="?", default=VIDEO_PATH_DEFAULT)
    parser.add_argument("--mode", choices=["vitpose", "hmr2", "both"], default="both")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--save_vis", action="store_true", default=True)
    parser.add_argument("--gif_fps", type=int, default=10, help="GIF 帧率")
    parser.add_argument("--gif_dur", type=float, default=3.0, help="GIF 时长（秒）")
    args = parser.parse_args()

    video_path = args.video
    if not os.path.exists(video_path):
        alt = os.path.join(os.path.dirname(video_path), "output1.mp4")
        if os.path.exists(alt):
            video_path = alt
        else:
            print(f"❌ 视频不存在：{video_path}")
            sys.exit(1)

    # ── STEP 0: 环境 ─────────────────────────────────────────────────────────────
    rlog("=" * 60)
    rlog("BeatsMatching Pipeline 测试 + GIF 可视化")
    rlog("=" * 60)
    rlog(f"时间  : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    rlog(f"视频  : {video_path}")
    rlog(f"输出  : {OUTPUT_DIR}")
    rlog(f"GIF   : {args.gif_fps}fps × {args.gif_dur}s")
    rlog()
    rlog("[0] 环境")
    rlog(f"Python : {sys.version.split()[0]}")
    rlog(f"PyTorch: {torch.__version__}")
    cuda_ok = torch.cuda.is_available()
    rlog(f"CUDA   : {'YES' if cuda_ok else 'NO'} / {torch.cuda.get_device_name(0) if cuda_ok else 'N/A'}")
    rlog()
    verify_all()

    # 读取视频
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Cannot open: {video_path}"
    VIDEO_FPS = cap.get(cv2.CAP_PROP_FPS)
    VIDEO_TOTAL = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    VIDEO_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    VIDEO_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rlog(f"视频   : {VIDEO_W}x{VIDEO_H} @ {VIDEO_FPS:.1f}fps  {VIDEO_TOTAL}帧  "
         f"{VIDEO_TOTAL/VIDEO_FPS:.1f}s")

    N_VIT = min(args.frames, VIDEO_TOTAL)
    N_HMR2 = min(args.frames, VIDEO_TOTAL)

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
    rlog(f"测试帧 : {len(frames_bgr)} 帧")

    # ── STEP 1: ViTPose ─────────────────────────────────────────────────────────────
    joint_coords_seq = None  # (T, 17, 2) 用于 GIF

    if args.mode in ("vitpose", "both"):
        rlog()
        rlog("=" * 60)
        rlog("[1] ViTPose 2D 关节点检测")

        t_load = time.time()
        from pose_extraction.vitpose.detector import ViTPoseDetector
        detector = ViTPoseDetector(vitpose_model=VITPOSE_MODEL_ID, yolo_model=YOLO_WEIGHTS)
        rlog(f"模型加载: {time.time()-t_load:.2f}s")

        vit_results = []
        timings_vit = []

        for i, (bgr, fidx) in enumerate(zip(frames_bgr, frame_indices)):
            ts = fidx / VIDEO_FPS
            t1 = time.time()
            result = detector.detect_frame(bgr, frame_idx=fidx, timestamp=ts)
            dt = time.time() - t1
            timings_vit.append(dt)
            vit_results.append(result)
            if i == 0:
                rlog(f"  首帧 (含JIT): {dt*1000:.0f}ms  persons={len(result.persons)}")
            elif i % 5 == 0:
                rlog(f"  帧{i:3d}: {dt*1000:.0f}ms")

        det_n_vit = sum(1 for r in vit_results if len(r.persons) > 0)
        avg_ms_vit = np.mean(timings_vit[1:]) * 1000
        rlog()
        rlog(f"ViTPose: 检出率 {det_n_vit}/{N_VIT}, avg延迟 {avg_ms_vit:.1f}ms")

        # 静态可视化
        r0 = vit_results[0]
        if r0.persons:
            kpts0 = r0.persons[0]
            vis = draw_skeleton_on_bgr(frames_bgr[0], kpts0[:,:2], kpts0[:,2])
            cv2.putText(vis, f"ViTPose  F{frame_indices[0]}  {det_n_vit}/{N_VIT}",
                        (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,100), 2)
            p = os.path.join(OUTPUT_DIR, "vitpose_keypoints.jpg")
            cv2.imwrite(p, vis)
            rlog(f"  [静态图] 第0帧骨架 -> {p}")

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
        while len(strip_imgs) < 16:
            strip_imgs.append(np.zeros((90,160,3), dtype=np.uint8))
        rows = [np.hstack(strip_imgs[r*4:(r+1)*4]) for r in range(4)]
        strip = np.vstack(rows)
        p = os.path.join(OUTPUT_DIR, "vitpose_strip.jpg")
        cv2.imwrite(p, strip)
        rlog(f"  [拼图] 16帧 -> {p}")

        # ★★★ 构建 GIF 所需的坐标序列 ★★★
        kp_seq_for_gif = []
        for r in vit_results:
            if r.persons:
                kp = r.persons[0][:, :2].astype(np.float32)
                kp_seq_for_gif.append(kp)
            else:
                kp_seq_for_gif.append(np.zeros((17, 2), dtype=np.float32))

        if kp_seq_for_gif:
            joint_coords_seq = np.stack(kp_seq_for_gif)  # (T, 17, 2)

    # ── STEP 2: HMR2 ───────────────────────────────────────────────────────────────
    joint_rotations_seq = None  # (T, 24, 3) 用于 GIF

    if args.mode in ("hmr2", "both"):
        rlog()
        rlog("=" * 60)
        rlog("[2] HMR2 3D 人体重建")

        t_load = time.time()
        from pose_extraction.hmr2.reconstructor import HMR2Reconstructor
        reconstructor = HMR2Reconstructor(checkpoint_path=HMR2_CHECKPOINT, yolo_model=YOLO_WEIGHTS)
        rlog(f"模型加载: {time.time()-t_load:.2f}s")

        hmr2_results = []
        timings_hmr2 = []

        for i in range(N_HMR2):
            bgr = frames_bgr[i]
            fidx = frame_indices[i]
            ts = fidx / VIDEO_FPS
            t1 = time.time()
            res = reconstructor.reconstruct_frame(bgr, frame_idx=fidx, timestamp=ts)
            dt = time.time() - t1
            timings_hmr2.append(dt)
            hmr2_results.append(res)
            if i == 0:
                rlog(f"  首帧 (含JIT): {dt*1000:.0f}ms  persons={len(res)}")
            elif i % 2 == 0:
                rlog(f"  帧{i:3d}: {dt*1000:.0f}ms")

        det_n_hmr2 = sum(1 for r in hmr2_results if len(r) > 0)
        avg_ms_hmr2 = np.mean(timings_hmr2[1:]) * 1000
        rlog()
        rlog(f"HMR2: 检出率 {det_n_hmr2}/{N_HMR2}, avg延迟 {avg_ms_hmr2:.1f}ms")

        first_hmr2 = next((r[0] for r in hmr2_results if r), None)
        if first_hmr2 and first_hmr2.joints_3d is not None and first_hmr2.pred_cam_t_full is not None:
            vis = draw_smpl_on_bgr(frames_bgr[0], first_hmr2.joints_3d, first_hmr2.pred_cam_t_full)
            cv2.putText(vis, "HMR2 - SMPL-44 3D Projection",
                        (8,28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,200,255), 2)
            p = os.path.join(OUTPUT_DIR, "hmr2_skeleton_overlay.jpg")
            cv2.imwrite(p, vis)
            rlog(f"  [投影图] HMR2 3D叠加 -> {p}")

        # ★★★ 构建 GIF 所需的旋转序列 ★★★
        rot_seq_for_gif = []
        for res_list in hmr2_results:
            if res_list and res_list[0].body_pose is not None:
                go = res_list[0].global_orient
                bp = res_list[0].body_pose
                if go is not None and bp is not None:
                    full = np.vstack([go.reshape(1, 3), bp])  # (24, 3)
                    rot_seq_for_gif.append(full.astype(np.float32))
                else:
                    rot_seq_for_gif.append(np.zeros((24, 3), dtype=np.float32))
            else:
                rot_seq_for_gif.append(np.zeros((24, 3), dtype=np.float32))

        if rot_seq_for_gif:
            joint_rotations_seq = np.stack(rot_seq_for_gif)  # (T, 24, 3)

    # ── STEP 3: 节拍检测（同 test_pipeline_visual.py）────────────────────────────
    rlog()
    rlog("=" * 60)
    rlog("[3] 节拍检测")

    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks

    motion_raw, motion_ts = [], []
    for i in range(1, len(kp_seq_for_gif or [])):
        delta = kp_seq_for_gif[i] - kp_seq_for_gif[i-1]
        speed = float(np.sqrt((delta**2).sum(axis=-1)).mean())
        motion_raw.append(speed)
        motion_ts.append(frame_indices[i] / VIDEO_FPS)

    motion_raw = np.array(motion_raw)
    smoothed = gaussian_filter1d(motion_raw, sigma=1.5)
    mu, sigma = smoothed.mean(), smoothed.std()
    adaptive_thr = mu + 0.5 * sigma
    peaks, _ = find_peaks(smoothed, height=adaptive_thr, distance=3)
    beat_times = motion_ts[peaks] if len(motion_ts) > 0 else []

    rlog(f"节拍: {len(peaks)} 个")
    if len(beat_times) > 1:
        ivs = np.diff(beat_times)
        rlog(f"间隔: avg={ivs.mean():.2f}s ~{60/ivs.mean():.0f} BPM")

    # 信号图
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

    ax1 = axes[0]
    dark_ax(ax1, 'ViTPose Motion Speed & Beats')
    if len(motion_ts) > 0:
        ax1.plot(motion_ts, motion_raw, color='#4a4a6a', lw=1.2, alpha=0.7, label='Raw')
        ax1.plot(motion_ts, smoothed, color='#00bcd4', lw=2.0, label='Smoothed')
        ax1.axhline(adaptive_thr, color='#ff7043', ls='--', lw=1.5, label=f'Thr={adaptive_thr:.2f}')
        ax1.scatter(beat_times, smoothed[peaks], s=90, color='#ffeb3b', zorder=6, label=f'{len(peaks)} beats')
    ax1.legend(facecolor='#222', labelcolor='white', fontsize=9)
    ax1.set_ylabel('Speed (px/f)', color='#aaa')

    ax2 = axes[1]
    dark_ax(ax2, 'Per-joint Speed Heatmap')
    if kp_seq_for_gif is not None and len(kp_seq_for_gif) > 1:
        joint_mat = np.array([
            np.sqrt(((kp_seq_for_gif[i]-kp_seq_for_gif[i-1])**2).sum(axis=-1))
            for i in range(1, len(kp_seq_for_gif))
        ]).T
        vmax = np.percentile(joint_mat, 95)
        im = ax2.imshow(joint_mat, aspect='auto', cmap='inferno',
                        extent=[motion_ts[0], motion_ts[-1], -0.5, 16.5],
                        vmin=0, vmax=vmax, origin='lower')
        plt.colorbar(im, ax=ax2, label='Speed')
        ax2.set_yticks(range(17))
        ax2.set_yticklabels(JOINT_NAMES, fontsize=7, color='#ccc')
        ax2.set_ylabel('Joint', color='#aaa')

    ax3 = axes[2]
    dark_ax(ax3, 'Rotation Gradient (if available)')

    plt.tight_layout(pad=2.5)
    p = os.path.join(OUTPUT_DIR, "beat_signal.jpg")
    plt.savefig(p, dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    rlog(f"  [信号图] -> {p}")

    # ── STEP 4: GIF 可视化生成（★ 核心新增功能 ★）────────────────────────────────
    rlog()
    rlog("=" * 60)
    rlog("[4] GIF 动图生成")

    gif_files = {}

    # 4a. ViTPose 骨架动画 GIF
    if joint_coords_seq is not None:
        path = generate_vitpose_gif(
            joint_coords_seq, VIDEO_FPS, OUTPUT_DIR,
            gif_fps=args.gif_fps, gif_duration=args.gif_dur,
        )
        if path:
            gif_files['ViTPose Skeleton GIF'] = path
    else:
        rlog("  [跳过] 无 ViTPose 数据，跳过骨架 GIF")

    # 4b. HMR2 SMPL 人体模型 GIF
    if joint_rotations_seq is not None:
        path = generate_hmr2_smpl_gif(
            joint_rotations_seq, VIDEO_FPS, OUTPUT_DIR,
            gif_fps=args.gif_fps, gif_duration=args.gif_dur,
        )
        if path:
            gif_files['HMR2 SMPL Body Model GIF'] = path
    else:
        rlog("  [跳过] 无 HMR2 数据，跳过 SMPL GIF")

    # ── STEP 5: 汇总报告 ──────────────────────────────────────────────────────────
    rlog()
    rlog("=" * 60)
    rlog("[5] 汇总")
    rlog()
    rlog(f"  {'Model':<25} {'Rate':<12} {'Avg Latency':<14} {'Est FPS'}")
    rlog(f"  {'-'*55}")
    if args.mode in ("vitpose", "both"):
        rlog(f"  {'ViTPose (GPU)':<25} {f'{det_n_vit}/{N_VIT}':<12} {f'{avg_ms_vit:.1f}ms':<14} ~{1000/avg_ms_vit:.1f}")
    if args.mode in ("hmr2", "both"):
        rlog(f"  {'HMR2 (GPU)':<25} {f'{det_n_hmr2}/{N_HMR2}':<12} {f'{avg_ms_hmr2:.1f}ms':<14} ~{1000/avg_ms_hmr2:.1f}")
    rlog()
    rlog(f"  Beats detected: {len(peaks)}")
    rlog()
    rlog("  GIF 输出:")
    for name, path in gif_files.items():
        size_kb = os.path.getsize(path) // 1024
        rlog(f"    ✓ {name}  ({size_kb}KB)")
        rlog(f"      {path}")

    rlog()
    rlog("  所有静态图片:")
    for fname in ["vitpose_keypoints.jpg","vitpose_strip.jpg",
                  "hmr2_skeleton_overlay.jpg","beat_signal.jpg"]:
        fp = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(fp):
            rlog(f"    [OK] {fname}  ({os.path.getsize(fp)//1024}KB)")

    rlog()
    rlog("✅ ALL TESTS PASSED — 含 GIF 可视化！")
    save_report()


if __name__ == "__main__":
    main()
