"""
端到端算法管线测试脚本
======================
从加载模型 → 视频推理 → 异常检测 → 平滑 → 节拍检测 → 生成GIF

用法:
    conda activate pose_unified
    python test_algorithm_pipeline.py                          # 默认测试
    python test_algorithm_pipeline.py --video path/to/video.mp4
    python test_algorithm_pipeline.py --mode vitpose           # 仅 ViTPose
    python test_algorithm_pipeline.py --mode hmr2              # 仅 HMR2
    python test_algorithm_pipeline.py --mode joint             # 联合
    python test_algorithm_pipeline.py --max_frames 60          # 限制帧数
"""

import os
import sys
import time
import argparse
import numpy as np

# 路径设置
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _PARENT)

# 输出目录
OUTPUT_DIR = os.path.join(_HERE, "test_output", "algorithm_test")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def log(msg: str):
    print(f"[TEST] {msg}")


def _make_mock_worker(config):
    """
    创建一个轻量级模拟 Worker 对象，绑定 EnhancedPoseExtractionWorker 的算法方法，
    但不触发 QThread 初始化。
    """
    import types
    from engines.enhanced_pose_worker import EnhancedPoseExtractionWorker as W

    class _Sig:
        def emit(self, *a):
            pass

    mock = types.SimpleNamespace(config=config, status=_Sig(), video_path="")

    # @staticmethod 方法直接引用（不绑定 self）
    static_methods = [
        "_find_outlier_segments",
        "_savgol_smooth",
        "_robust_normalize",
        "_compute_local_snr",
        "_check_joint_limits",
    ]

    # 实例方法需要绑定 self
    instance_methods = [
        "_handle_outliers",
        "_compute_weighted_velocity",
        "_compute_rotation_metric",
        "_detect_beats",
        "_detect_multiscale",
        "_detect_weighted_fusion",
        "_detect_adaptive_fusion",
        "_adaptive_peak_detect",
        "_compute_angular_velocity",
        "_interpolate_outliers_spline",
    ]

    for name in static_methods:
        fn = getattr(W, name, None)
        if fn is not None:
            setattr(mock, name, fn)

    for name in instance_methods:
        # 从类字典取原始函数（避免 descriptor 协议）
        fn = W.__dict__.get(name)
        if fn is not None:
            setattr(mock, name, types.MethodType(fn, mock))

    return mock


def find_default_video() -> str:
    """查找默认测试视频"""
    candidates = [
        os.path.join(_PARENT, "test_videos", "aist_gBR_sBM_c01_d04_mBR0_ch01.mp4"),
        os.path.join(_PARENT, "test_videos", "aist_gHO_sBM_c01_d19_mHO0_ch01.mp4"),
        os.path.join(_HERE, "output.mp4"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError("找不到测试视频，请用 --video 指定")


def test_smpl_renderer():
    """测试 SMPL 渲染器基本功能"""
    log("=" * 60)
    log("测试 1: SMPL 渲染器")
    log("=" * 60)

    from engines.smpl_renderer import SMPLRenderer
    from pose_extraction.weights_config import SMPL_NEUTRAL
    from PIL import Image

    assert os.path.exists(SMPL_NEUTRAL), f"SMPL_NEUTRAL.pkl 不存在: {SMPL_NEUTRAL}"
    renderer = SMPLRenderer(SMPL_NEUTRAL)

    # T-pose
    t0 = time.time()
    verts = renderer.forward(np.zeros((24, 3), dtype=np.float32))
    log(f"  Forward pass: {time.time()-t0:.3f}s, verts shape: {verts.shape}")

    assert verts.shape == (6890, 3), f"顶点形状错误: {verts.shape}"
    # SMPL Y-up: 高度应在 Y 轴
    height = verts[:, 1].max() - verts[:, 1].min()
    assert 1.5 < height < 2.0, f"T-pose 高度异常: {height:.2f}m"
    log(f"  T-pose 高度: {height:.2f}m ✓")

    # 渲染 T-pose
    t0 = time.time()
    img = renderer.render_frame(np.zeros((24, 3), dtype=np.float32), size=(400, 500))
    log(f"  Render T-pose: {time.time()-t0:.3f}s, shape: {img.shape}")
    assert img.shape == (500, 400, 3), f"图像尺寸错误: {img.shape}"
    Image.fromarray(img).save(os.path.join(OUTPUT_DIR, "smpl_tpose.png"))

    # 渲染弯曲姿势
    pose = np.zeros((24, 3), dtype=np.float32)
    pose[4] = [1.2, 0, 0]     # 左膝弯曲
    pose[19] = [0, 0, -1.5]   # 右肘弯曲
    pose[17] = [0, 0, -0.8]   # 右肩抬起
    img2 = renderer.render_frame(pose, size=(400, 500), azimuth=30, elevation=15)
    Image.fromarray(img2).save(os.path.join(OUTPUT_DIR, "smpl_posed.png"))
    log(f"  渲染结果已保存 ✓")

    # 短 GIF 测试 (5帧)
    poses = np.zeros((5, 24, 3), dtype=np.float32)
    for i in range(5):
        poses[i, 4] = [0.3 * i, 0, 0]  # 渐进膝弯
    t0 = time.time()
    renderer.render_gif(poses, os.path.join(OUTPUT_DIR, "smpl_test.gif"),
                        fps=5, size=(300, 400), rotate_view=False)
    log(f"  GIF (5帧) 渲染: {time.time()-t0:.1f}s ✓")
    log("")


def test_vitpose_pipeline(video_path: str, max_frames: int = 0):
    """测试 ViTPose 提取管线"""
    log("=" * 60)
    log("测试 2: ViTPose 管线")
    log("=" * 60)
    import cv2
    from pose_extraction.vitpose.detector import ViTPoseDetector
    from pose_extraction.weights_config import VITPOSE_MODEL_ID, YOLO_WEIGHTS, HF_HOME

    os.environ["HF_HOME"] = HF_HOME
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

    # 加载模型
    t0 = time.time()
    detector = ViTPoseDetector(
        vitpose_model=VITPOSE_MODEL_ID,
        yolo_model=YOLO_WEIGHTS,
    )
    log(f"  ViTPose 模型加载: {time.time()-t0:.1f}s")

    # 读取视频
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_frames = total if max_frames <= 0 else min(total, max_frames)
    log(f"  视频: {n_frames}/{total} 帧, {fps:.1f} fps, 时长 {n_frames/fps:.1f}s")

    # 推理
    kp_list = []
    valid_mask = np.ones(n_frames, dtype=bool)
    t0 = time.time()

    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            valid_mask[i] = False
            kp_list.append(np.zeros((17, 2), dtype=np.float32))
            continue
        result = detector.detect_frame(frame, frame_idx=i, timestamp=i / fps)
        n_persons = len(result.persons) if result and result.persons else 0
        if n_persons >= 1:
            # 多人时取第一个（通常是置信度最高的主体）
            kp = list(result.persons.values())[0]  # Dict[int, (J,3)]
            kp_list.append(kp[:, :2].astype(np.float32))
        else:
            valid_mask[i] = False
            kp_list.append(np.zeros((17, 2), dtype=np.float32))
        if (i + 1) % 100 == 0 or i == n_frames - 1:
            elapsed = time.time() - t0
            log(f"  ViTPose 推理: {i+1}/{n_frames} ({(i+1)/elapsed:.1f} fps)")

    cap.release()
    elapsed = time.time() - t0
    log(f"  推理 {n_frames} 帧: {elapsed:.1f}s ({n_frames/elapsed:.1f} fps)")
    log(f"  有效帧: {valid_mask.sum()}/{n_frames} ({valid_mask.mean()*100:.1f}%)")

    coords = np.stack(kp_list)  # (T, 17, 2)
    return coords, valid_mask, fps


def test_hmr2_pipeline(video_path: str, max_frames: int = 0):
    """测试 HMR2 提取管线"""
    log("=" * 60)
    log("测试 3: HMR2 管线")
    log("=" * 60)
    import cv2
    from pose_extraction.hmr2.reconstructor import HMR2Reconstructor
    from pose_extraction.weights_config import HMR2_CHECKPOINT, YOLO_WEIGHTS

    # 加载模型
    t0 = time.time()
    reconstructor = HMR2Reconstructor(
        checkpoint_path=HMR2_CHECKPOINT,
        yolo_model=YOLO_WEIGHTS,
    )
    log(f"  HMR2 模型加载: {time.time()-t0:.1f}s")

    # 读取视频
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_frames = total if max_frames <= 0 else min(total, max_frames)
    log(f"  视频: {n_frames}/{total} 帧, {fps:.1f} fps, 时长 {n_frames/fps:.1f}s")

    rot_list = []
    valid_mask = np.ones(n_frames, dtype=bool)
    t0 = time.time()

    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            valid_mask[i] = False
            rot_list.append(np.zeros((24, 3), dtype=np.float32))
            continue
        results = reconstructor.reconstruct_frame(frame, frame_idx=i, timestamp=i / fps)
        n_persons = len(results) if results else 0
        if n_persons >= 1:
            # 多人时取第一个（通常是置信度最高的主体）
            r = results[0]
            fp = r.full_pose   # property: (24, 3)
            if fp is not None:
                rot_list.append(fp.astype(np.float32))
            else:
                valid_mask[i] = False
                rot_list.append(np.zeros((24, 3), dtype=np.float32))
        else:
            valid_mask[i] = False
            rot_list.append(np.zeros((24, 3), dtype=np.float32))
        if (i + 1) % 100 == 0 or i == n_frames - 1:
            elapsed = time.time() - t0
            log(f"  HMR2 推理: {i+1}/{n_frames} ({(i+1)/elapsed:.1f} fps)")

    cap.release()
    elapsed = time.time() - t0
    log(f"  推理 {n_frames} 帧: {elapsed:.1f}s ({n_frames/elapsed:.1f} fps)")
    log(f"  有效帧: {valid_mask.sum()}/{n_frames} ({valid_mask.mean()*100:.1f}%)")

    rotations = np.stack(rot_list)  # (T, 24, 3)
    return rotations, valid_mask, fps


def test_outlier_detection_and_smoothing(
    coords: np.ndarray, valid_vit: np.ndarray,
    rotations: np.ndarray, valid_hmr: np.ndarray,
    fps: float,
):
    """
    测试异常检测 + 平滑管线
    这里复用 EnhancedPoseExtractionWorker 的静态/实例方法
    """
    log("=" * 60)
    log("测试 4: 异常检测 + 平滑")
    log("=" * 60)
    from core.algorithm_config import AlgorithmConfig

    config = AlgorithmConfig(
        outlier_threshold=3.0,
        gaussian_sigma=1.5,
        peak_height_factor=0.5,
        min_peak_distance=3,
    )

    worker = _make_mock_worker(config)

    # ViTPose 异常检测
    if coords is not None:
        raw_coords = coords.copy()
        t0 = time.time()
        clean_coords, clean_vit_mask = worker._handle_outliers(
            coords.copy(), valid_vit.copy(), "ViTPose"
        )
        log(f"  ViTPose 异常检测: {time.time()-t0:.2f}s")

        n_outlier = (~clean_vit_mask).sum()
        log(f"  ViTPose 异常帧: {n_outlier}/{len(coords)}")

        # 对比平滑前后的差异
        diff = np.linalg.norm(
            (clean_coords - raw_coords).reshape(len(coords), -1), axis=1
        )
        log(f"  ViTPose 平滑修正量: mean={diff.mean():.2f}, max={diff.max():.2f}")
    else:
        clean_coords, clean_vit_mask = None, None

    # HMR2 异常检测
    if rotations is not None:
        raw_rot = rotations.copy()
        t0 = time.time()
        clean_rot, clean_hmr_mask = worker._handle_outliers(
            rotations.copy(), valid_hmr.copy(), "HMR2"
        )
        log(f"  HMR2 异常检测: {time.time()-t0:.2f}s")

        n_outlier = (~clean_hmr_mask).sum()
        log(f"  HMR2 异常帧: {n_outlier}/{len(rotations)}")

        diff = np.linalg.norm(
            (clean_rot - raw_rot).reshape(len(rotations), -1), axis=1
        )
        log(f"  HMR2 平滑修正量: mean={diff.mean():.4f}, max={diff.max():.4f}")
    else:
        clean_rot, clean_hmr_mask = None, None

    log("")
    return clean_coords, clean_vit_mask, clean_rot, clean_hmr_mask


def test_beat_detection(
    coords: np.ndarray, rotations: np.ndarray, fps: float,
    coords_raw: np.ndarray = None, rotations_raw: np.ndarray = None,
):
    """测试节拍/突变帧检测（带完整诊断可视化）"""
    log("=" * 60)
    log("测试 5: 节拍/突变帧检测（带诊断可视化）")
    log("=" * 60)
    from core.algorithm_config import AlgorithmConfig

    config = AlgorithmConfig(
        fusion_strategy="weighted_sum",
        gaussian_sigma=1.5,
        peak_height_factor=0.5,
        min_peak_distance=3,
        vitpose_weight=0.6,
    )

    worker = _make_mock_worker(config)

    # 计算特征
    velocities = None
    rot_grads = None
    velocities_raw = None
    rot_grads_raw = None

    if coords is not None:
        velocities = worker._compute_weighted_velocity(coords)
        log(f"  ViTPose 速度信号: shape={velocities.shape}, mean={velocities.mean():.2f}")
    if coords_raw is not None:
        velocities_raw = worker._compute_weighted_velocity(coords_raw)

    if rotations is not None:
        rot_grads = worker._compute_rotation_metric(rotations)
        log(f"  HMR2 旋转梯度: shape={rot_grads.shape}, mean={rot_grads.mean():.4f}")
    if rotations_raw is not None:
        rot_grads_raw = worker._compute_rotation_metric(rotations_raw)

    # 节拍检测（带诊断）
    t0 = time.time()
    diagnostics = None
    if velocities is not None and rot_grads is not None:
        keyframes, beat_times, diagnostics = worker._detect_weighted_fusion(
            velocities, rot_grads, fps, return_diagnostics=True
        )
    elif velocities is not None:
        keyframes, beat_times, diagnostics = worker._detect_multiscale(
            velocities, fps, "velocity", return_diagnostics=True
        )
    elif rot_grads is not None:
        keyframes, beat_times, diagnostics = worker._detect_multiscale(
            rot_grads, fps, "rotation", return_diagnostics=True
        )
    else:
        keyframes, beat_times = [0], [0.0]

    log(f"  检测耗时: {time.time()-t0:.3f}s")
    log(f"  检测到关键帧: {len(keyframes)} 个")
    log(f"  关键帧索引 (前20): {keyframes[:20]}")
    if beat_times:
        log(f"  节拍时间 (前10): {[f'{t:.2f}s' for t in beat_times[:10]]}")

    # 提取关键帧来源标签（峰/谷/边界）
    kf_sources = None
    if diagnostics is not None and "keyframe_sources" in diagnostics:
        kf_sources = diagnostics["keyframe_sources"]
        n_peaks = sum(1 for s in kf_sources if s == 'peak')
        n_valleys = sum(1 for s in kf_sources if s == 'valley')
        n_boundary = sum(1 for s in kf_sources if s == 'boundary')
        log(f"  来源分布: 峰值={n_peaks}, 谷值={n_valleys}, 边界={n_boundary}")

    # ── 生成全部诊断图 ──
    _plot_signals(velocities, rot_grads, keyframes, fps, OUTPUT_DIR, kf_sources=kf_sources)

    if velocities_raw is not None or rot_grads_raw is not None:
        _plot_raw_vs_smoothed(velocities_raw, velocities, rot_grads_raw, rot_grads,
                              keyframes, fps, OUTPUT_DIR, kf_sources=kf_sources)

    if diagnostics is not None:
        _plot_multiscale_channels(diagnostics, keyframes, fps, OUTPUT_DIR,
                                  kf_sources=kf_sources)

        if "v_norm" in diagnostics and "r_norm" in diagnostics:
            _plot_fusion_weights(diagnostics, keyframes, fps, OUTPUT_DIR,
                                 kf_sources=kf_sources)

    if coords is not None or rotations is not None:
        _plot_joint_heatmaps(coords, rotations, keyframes, fps, OUTPUT_DIR,
                             kf_sources=kf_sources)

    log("")
    return keyframes, beat_times, velocities, rot_grads


def _plot_signals(velocities, rot_grads, keyframes, fps, output_dir, kf_sources=None):
    """绘制信号 + 关键帧的可视化图（峰/谷视觉区分）"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    # 使用中文字体
    zh_fonts = [f.name for f in fm.fontManager.ttflist
                if any(k in f.name for k in ['SimHei', 'Microsoft YaHei', 'PingFang', 'Noto Sans CJK'])]
    if zh_fonts:
        plt.rcParams['font.sans-serif'] = [zh_fonts[0]] + plt.rcParams['font.sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig.patch.set_facecolor('#1e1e1e')

    for ax in axes:
        ax.set_facecolor('#2a2a2a')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('#555')

    T = 0
    if velocities is not None:
        T = len(velocities)
        t_axis = np.arange(T) / fps
        axes[0].plot(t_axis, velocities, color='#22c55e', linewidth=0.8, alpha=0.8)
        axes[0].set_ylabel('ViTPose Velocity', color='white')
        axes[0].set_title('ViTPose 加权关节速度', color='white', fontsize=11)
        _add_keyframe_lines(axes[0], keyframes, fps, T, kf_sources=kf_sources)

    if rot_grads is not None:
        T = max(T, len(rot_grads))
        t_axis = np.arange(len(rot_grads)) / fps
        axes[1].plot(t_axis, rot_grads, color='#5b8cff', linewidth=0.8, alpha=0.8)
        axes[1].set_ylabel('HMR2 Rotation Grad', color='white')
        axes[1].set_title('HMR2 旋转梯度', color='white', fontsize=11)
        _add_keyframe_lines(axes[1], keyframes, fps, len(rot_grads), kf_sources=kf_sources)

    # 融合信号
    if velocities is not None and rot_grads is not None:
        min_len = min(len(velocities), len(rot_grads))
        v_n = (velocities[:min_len] - velocities[:min_len].min()) / \
              (velocities[:min_len].max() - velocities[:min_len].min() + 1e-8)
        r_n = (rot_grads[:min_len] - rot_grads[:min_len].min()) / \
              (rot_grads[:min_len].max() - rot_grads[:min_len].min() + 1e-8)
        fused = 0.6 * v_n + 0.4 * r_n
        t_axis = np.arange(min_len) / fps
        axes[2].plot(t_axis, fused, color='#e879f9', linewidth=0.8, alpha=0.8)
        axes[2].set_ylabel('Fused Signal', color='white')
        axes[2].set_title('融合信号 + 关键帧', color='white', fontsize=11)
        _add_keyframe_lines(axes[2], keyframes, fps, min_len, alpha=0.7, lw=1.0, kf_sources=kf_sources)

    axes[-1].set_xlabel('Time (s)', color='white')

    # 添加峰/谷图例
    if kf_sources is not None:
        from matplotlib.lines import Line2D
        legend_elems = [
            Line2D([0], [0], color='#FFB84D', linewidth=1.5, label='峰值 (peak)'),
            Line2D([0], [0], color='#5B8CFF', linewidth=1.5, linestyle='--', label='谷值 (valley)'),
            Line2D([0], [0], color='#888888', linewidth=1.5, linestyle=':', label='边界 (boundary)'),
        ]
        axes[0].legend(handles=legend_elems, loc='upper right', fontsize=8,
                       facecolor='#333', edgecolor='#555', labelcolor='white')

    plt.tight_layout()
    path = os.path.join(output_dir, "signal_plot.png")
    plt.savefig(path, dpi=150, facecolor='#1e1e1e')
    plt.close()
    log(f"  信号图已保存: {path}")


def _init_matplotlib():
    """初始化 matplotlib 中文字体与暗色风格"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    zh_fonts = [f.name for f in fm.fontManager.ttflist
                if any(k in f.name for k in ['SimHei', 'Microsoft YaHei', 'PingFang', 'Noto Sans CJK'])]
    if zh_fonts:
        plt.rcParams['font.sans-serif'] = [zh_fonts[0]] + plt.rcParams['font.sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    return plt


def _style_axes(axes, fig):
    """统一暗色主题"""
    fig.patch.set_facecolor('#1e1e1e')
    for ax in (axes if hasattr(axes, '__iter__') else [axes]):
        ax.set_facecolor('#2a2a2a')
        ax.tick_params(colors='white', labelsize=9)
        for spine in ax.spines.values():
            spine.set_color('#555')
        ax.yaxis.label.set_color('white')
        ax.xaxis.label.set_color('white')
        ax.title.set_color('white')


def _add_keyframe_lines(ax, keyframes, fps, N, color='#ff6b6b', alpha=0.5, lw=0.8,
                        kf_sources=None):
    """在子图上标记关键帧竖线（支持峰/谷视觉区分）"""
    _SRC_COLORS = {'peak': '#FFB84D', 'valley': '#5B8CFF', 'boundary': '#888888'}
    _SRC_STYLES = {'peak': '-', 'valley': '--', 'boundary': ':'}
    for i, kf in enumerate(keyframes):
        if 0 <= kf < N:
            if kf_sources is not None and i < len(kf_sources):
                src = kf_sources[i]
                c = _SRC_COLORS.get(src, color)
                ls = _SRC_STYLES.get(src, '-')
                ax.axvline(kf / fps, color=c, alpha=alpha, linewidth=lw, linestyle=ls)
            else:
                ax.axvline(kf / fps, color=color, alpha=alpha, linewidth=lw)


def _plot_raw_vs_smoothed(vel_raw, vel_smooth, rot_raw, rot_smooth,
                          keyframes, fps, output_dir, kf_sources=None):
    """图1: 原始信号 vs 平滑后信号对比"""
    plt = _init_matplotlib()

    n_plots = sum([vel_raw is not None, rot_raw is not None])
    if n_plots == 0:
        return

    fig, axes = plt.subplots(n_plots, 1, figsize=(16, 4 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]
    _style_axes(axes, fig)

    idx = 0
    if vel_raw is not None and vel_smooth is not None:
        N = min(len(vel_raw), len(vel_smooth))
        t_axis = np.arange(N) / fps
        axes[idx].plot(t_axis, vel_raw[:N], color='#666', linewidth=0.5, alpha=0.6, label='原始')
        axes[idx].plot(t_axis, vel_smooth[:N], color='#22c55e', linewidth=1.0, alpha=0.9, label='平滑后')
        axes[idx].set_title('ViTPose 加权关节速度: 原始 vs 平滑', fontsize=11)
        axes[idx].set_ylabel('Velocity')
        axes[idx].legend(loc='upper right', fontsize=9, facecolor='#333', edgecolor='#555', labelcolor='white')
        _add_keyframe_lines(axes[idx], keyframes, fps, N, kf_sources=kf_sources)
        idx += 1

    if rot_raw is not None and rot_smooth is not None:
        N = min(len(rot_raw), len(rot_smooth))
        t_axis = np.arange(N) / fps
        axes[idx].plot(t_axis, rot_raw[:N], color='#666', linewidth=0.5, alpha=0.6, label='原始')
        axes[idx].plot(t_axis, rot_smooth[:N], color='#5b8cff', linewidth=1.0, alpha=0.9, label='平滑后')
        axes[idx].set_title('HMR2 旋转梯度: 原始 vs 平滑', fontsize=11)
        axes[idx].set_ylabel('Rotation Gradient')
        axes[idx].legend(loc='upper right', fontsize=9, facecolor='#333', edgecolor='#555', labelcolor='white')
        _add_keyframe_lines(axes[idx], keyframes, fps, N, kf_sources=kf_sources)

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    path = os.path.join(output_dir, "signal_raw_vs_smoothed.png")
    plt.savefig(path, dpi=150, facecolor='#1e1e1e')
    plt.close()
    log(f"  原始vs平滑图已保存: {path}")


def _plot_multiscale_channels(diagnostics, keyframes, fps, output_dir, kf_sources=None):
    """图2: 多尺度检测三通道 + 投票热力图"""
    plt = _init_matplotlib()

    smooth_fine = diagnostics.get("smooth_fine")
    smooth_coarse = diagnostics.get("smooth_coarse")
    acceleration = diagnostics.get("acceleration")
    vote_map = diagnostics.get("vote_map")
    peaks_fine = diagnostics.get("peaks_fine", [])
    peaks_coarse = diagnostics.get("peaks_coarse", [])
    zero_crossings = diagnostics.get("zero_crossings", [])

    if smooth_fine is None:
        return

    N = len(smooth_fine)
    t_axis = np.arange(N) / fps

    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
    _style_axes(axes, fig)

    # 通道A: 细粒度
    axes[0].plot(t_axis, smooth_fine, color='#22c55e', linewidth=0.8, alpha=0.8, label='细粒度平滑')
    if len(peaks_fine) > 0:
        pf = np.array(peaks_fine)
        pf = pf[pf < N]
        axes[0].scatter(pf / fps, smooth_fine[pf], color='#ff6b6b', s=20, zorder=5, label=f'峰值 ({len(peaks_fine)})')
    axes[0].set_title(f'通道A: 细粒度峰值检测 (σ=0.5×σ_config)', fontsize=11)
    axes[0].set_ylabel('Signal')
    axes[0].legend(loc='upper right', fontsize=8, facecolor='#333', edgecolor='#555', labelcolor='white')
    _add_keyframe_lines(axes[0], keyframes, fps, N, alpha=0.3, kf_sources=kf_sources)

    # 通道B: 粗粒度
    axes[1].plot(t_axis, smooth_coarse, color='#5b8cff', linewidth=0.8, alpha=0.8, label='粗粒度平滑')
    if len(peaks_coarse) > 0:
        pc = np.array(peaks_coarse)
        pc = pc[pc < N]
        axes[1].scatter(pc / fps, smooth_coarse[pc], color='#ff6b6b', s=20, zorder=5, label=f'峰值 ({len(peaks_coarse)})')
    axes[1].set_title(f'通道B: 粗粒度峰值检测 (σ=2.0×σ_config)', fontsize=11)
    axes[1].set_ylabel('Signal')
    axes[1].legend(loc='upper right', fontsize=8, facecolor='#333', edgecolor='#555', labelcolor='white')
    _add_keyframe_lines(axes[1], keyframes, fps, N, alpha=0.3, kf_sources=kf_sources)

    # 通道C: 加速度零交叉
    if acceleration is not None:
        axes[2].plot(t_axis[:len(acceleration)], acceleration[:N], color='#e879f9', linewidth=0.6, alpha=0.7, label='加速度')
        axes[2].axhline(0, color='#666', linewidth=0.5, linestyle='--')
        if len(zero_crossings) > 0:
            zc = np.array(zero_crossings)
            zc = zc[zc < N]
            axes[2].scatter(zc / fps, np.zeros(len(zc)), color='#ffb84d', s=25, zorder=5, marker='v',
                           label=f'零交叉 ({len(zero_crossings)})')
    axes[2].set_title(f'通道C: 加速度零交叉点 (动作方向反转)', fontsize=11)
    axes[2].set_ylabel('Acceleration')
    axes[2].legend(loc='upper right', fontsize=8, facecolor='#333', edgecolor='#555', labelcolor='white')
    _add_keyframe_lines(axes[2], keyframes, fps, N, alpha=0.3, kf_sources=kf_sources)

    # 投票热力图
    if vote_map is not None:
        axes[3].fill_between(t_axis[:len(vote_map)], 0, vote_map[:N],
                            color='#e879f9', alpha=0.4)
        axes[3].plot(t_axis[:len(vote_map)], vote_map[:N], color='#e879f9', linewidth=1.0, alpha=0.9)
        axes[3].axhline(1.5, color='#ff6b6b', linewidth=1.0, linestyle='--', label='确认阈值 (1.5)')
        # 标记最终关键帧（区分峰/谷）
        _SRC_COLORS = {'peak': '#FFB84D', 'valley': '#5B8CFF', 'boundary': '#888888'}
        for i, kf in enumerate(keyframes):
            if 0 < kf < N:
                src = kf_sources[i] if kf_sources and i < len(kf_sources) else 'peak'
                c = _SRC_COLORS.get(src, '#ff6b6b')
                axes[3].axvline(kf / fps, color=c, alpha=0.7, linewidth=1.2)
                label_text = f'{kf/fps:.2f}s'
                if src == 'valley':
                    label_text += '▽'
                axes[3].annotate(label_text, xy=(kf/fps, vote_map[min(kf, len(vote_map)-1)]),
                               fontsize=7, color=c, ha='center', va='bottom',
                               rotation=90)
    axes[3].set_title('投票热力图 + 最终关键帧', fontsize=11)
    axes[3].set_ylabel('Vote Score')
    axes[3].legend(loc='upper right', fontsize=8, facecolor='#333', edgecolor='#555', labelcolor='white')

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    path = os.path.join(output_dir, "multiscale_channels.png")
    plt.savefig(path, dpi=150, facecolor='#1e1e1e')
    plt.close()
    log(f"  多尺度通道图已保存: {path}")


def _plot_fusion_weights(diagnostics, keyframes, fps, output_dir, kf_sources=None):
    """图3: 归一化信号 + 自适应alpha权重 + SNR"""
    plt = _init_matplotlib()

    v_norm = diagnostics.get("v_norm")
    r_norm = diagnostics.get("r_norm")
    alpha = diagnostics.get("alpha_per_frame")
    v_conf = diagnostics.get("v_conf")
    r_conf = diagnostics.get("r_conf")
    fused = diagnostics.get("fused")

    if v_norm is None or r_norm is None:
        return

    N = len(v_norm)
    t_axis = np.arange(N) / fps

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    _style_axes(axes, fig)

    # 归一化信号
    axes[0].plot(t_axis, v_norm, color='#22c55e', linewidth=0.8, alpha=0.8, label='v_norm (ViTPose)')
    axes[0].plot(t_axis, r_norm, color='#5b8cff', linewidth=0.8, alpha=0.8, label='r_norm (HMR2)')
    if fused is not None:
        axes[0].plot(t_axis[:len(fused)], fused[:N], color='#e879f9', linewidth=1.0, alpha=0.9, label='fused')
    axes[0].set_title('归一化信号 (鲁棒百分位截断)', fontsize=11)
    axes[0].set_ylabel('Normalized Value')
    axes[0].legend(loc='upper right', fontsize=8, facecolor='#333', edgecolor='#555', labelcolor='white')
    _add_keyframe_lines(axes[0], keyframes, fps, N, kf_sources=kf_sources)

    # Alpha权重曲线
    if alpha is not None:
        axes[1].plot(t_axis[:len(alpha)], alpha[:N], color='#ffb84d', linewidth=1.0, alpha=0.9)
        axes[1].axhline(0.5, color='#666', linewidth=0.5, linestyle='--', label='均衡线 (0.5)')
        axes[1].fill_between(t_axis[:len(alpha)], 0.5, alpha[:N],
                            where=alpha[:N] > 0.5, color='#22c55e', alpha=0.15, label='ViTPose主导')
        axes[1].fill_between(t_axis[:len(alpha)], 0.5, alpha[:N],
                            where=alpha[:N] < 0.5, color='#5b8cff', alpha=0.15, label='HMR2主导')
    axes[1].set_title('逐帧自适应权重 α(t): ViTPose占比', fontsize=11)
    axes[1].set_ylabel('α(t)')
    axes[1].set_ylim(0, 1)
    axes[1].legend(loc='upper right', fontsize=8, facecolor='#333', edgecolor='#555', labelcolor='white')
    _add_keyframe_lines(axes[1], keyframes, fps, N, kf_sources=kf_sources)

    # SNR曲线
    if v_conf is not None and r_conf is not None:
        axes[2].plot(t_axis[:len(v_conf)], v_conf[:N], color='#22c55e', linewidth=0.8, alpha=0.8, label='SNR_v (ViTPose)')
        axes[2].plot(t_axis[:len(r_conf)], r_conf[:N], color='#5b8cff', linewidth=0.8, alpha=0.8, label='SNR_r (HMR2)')
    axes[2].set_title('局部信噪比 (SNR): 信号可靠性', fontsize=11)
    axes[2].set_ylabel('SNR')
    axes[2].legend(loc='upper right', fontsize=8, facecolor='#333', edgecolor='#555', labelcolor='white')
    _add_keyframe_lines(axes[2], keyframes, fps, N, kf_sources=kf_sources)

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    path = os.path.join(output_dir, "fusion_weights.png")
    plt.savefig(path, dpi=150, facecolor='#1e1e1e')
    plt.close()
    log(f"  融合权重图已保存: {path}")


def _plot_joint_heatmaps(coords, rotations, keyframes, fps, output_dir, kf_sources=None):
    """图4: 关节级热力图 (Joint × Time)"""
    plt = _init_matplotlib()

    n_plots = sum([coords is not None, rotations is not None])
    if n_plots == 0:
        return

    fig, axes = plt.subplots(n_plots, 1, figsize=(16, 5 * n_plots))
    if n_plots == 1:
        axes = [axes]
    fig.patch.set_facecolor('#1e1e1e')

    coco_names = ['鼻', '左眼', '右眼', '左耳', '右耳',
                  '左肩', '右肩', '左肘', '右肘', '左腕', '右腕',
                  '左髋', '右髋', '左膝', '右膝', '左踝', '右踝']

    idx = 0
    if coords is not None and len(coords) > 1:
        T = len(coords)
        # 逐关节逐帧速度
        joint_vel = np.sqrt(np.sum(np.diff(coords, axis=0) ** 2, axis=-1))  # (T-1, 17)
        t_axis = np.arange(joint_vel.shape[0]) / fps

        im = axes[idx].imshow(joint_vel.T, aspect='auto', cmap='magma',
                             extent=[0, t_axis[-1], 16.5, -0.5], interpolation='bilinear')
        axes[idx].set_yticks(range(17))
        axes[idx].set_yticklabels(coco_names, fontsize=8, color='white')
        axes[idx].set_title('ViTPose 逐关节速度热力图 (Joint × Time)', color='white', fontsize=11)
        axes[idx].set_xlabel('Time (s)', color='white')
        axes[idx].tick_params(colors='white')
        fig.colorbar(im, ax=axes[idx], label='Speed (px/frame)', fraction=0.02)

        # 标记关键帧
        _add_keyframe_lines(axes[idx], keyframes, fps, joint_vel.shape[0], alpha=0.6,
                            kf_sources=kf_sources)
        idx += 1

    if rotations is not None and len(rotations) > 1:
        T = len(rotations)
        # 逐关节逐帧旋转梯度 (排除根关节)
        body_rot = rotations[:, 1:, :]  # (T, 23, 3)
        rot_diff = np.sum(np.diff(body_rot, axis=0) ** 2, axis=-1)  # (T-1, 23)
        t_axis = np.arange(rot_diff.shape[0]) / fps

        smpl_short = [f'J{i}' for i in range(1, 24)]

        im = axes[idx].imshow(rot_diff.T, aspect='auto', cmap='inferno',
                             extent=[0, t_axis[-1], 22.5, -0.5], interpolation='bilinear')
        axes[idx].set_yticks(range(23))
        axes[idx].set_yticklabels(smpl_short, fontsize=7, color='white')
        axes[idx].set_title('HMR2 逐关节旋转梯度热力图 (Joint × Time)', color='white', fontsize=11)
        axes[idx].set_xlabel('Time (s)', color='white')
        axes[idx].tick_params(colors='white')
        fig.colorbar(im, ax=axes[idx], label='Rotation Gradient (rad²)', fraction=0.02)

        _add_keyframe_lines(axes[idx], keyframes, fps, rot_diff.shape[0],
                            color='#22c55e', alpha=0.6, kf_sources=kf_sources)

    plt.tight_layout()
    path = os.path.join(output_dir, "joint_heatmaps.png")
    plt.savefig(path, dpi=150, facecolor='#1e1e1e')
    plt.close()
    log(f"  关节热力图已保存: {path}")


def test_gif_generation(
    coords: np.ndarray, rotations: np.ndarray, fps: float,
):
    """测试 GIF 生成（使用平滑后的数据，渲染完整视频）"""
    log("=" * 60)
    log("测试 6: GIF 生成（完整视频）")
    log("=" * 60)

    gif_fps = 10
    T_total = len(coords) if coords is not None else len(rotations)
    step = max(1, int(fps / gif_fps))
    n_gif_frames = len(range(0, T_total, step))

    log(f"  总帧数: {T_total}, GIF 帧率: {gif_fps}, 采样步长: {step}, GIF 帧数: {n_gif_frames}")

    # ── ViTPose 骨架 GIF ──
    if coords is not None:
        log(f"  生成 ViTPose 骨架 GIF...")
        t0 = time.time()
        _generate_vitpose_gif(coords, T_total, step, gif_fps, OUTPUT_DIR)
        log(f"  ViTPose GIF: {time.time()-t0:.1f}s ✓")

    # ── HMR2 SMPL 人体模型 GIF ──
    if rotations is not None:
        log(f"  生成 HMR2 SMPL 人体模型 GIF ({n_gif_frames} 帧)...")
        t0 = time.time()
        _generate_hmr2_smpl_gif(rotations, T_total, step, gif_fps, OUTPUT_DIR)
        log(f"  HMR2 SMPL GIF: {time.time()-t0:.1f}s ✓")

    log("")


def _generate_vitpose_gif(coords, T, step, gif_fps, output_dir):
    """ViTPose 骨架动画 GIF"""
    from PIL import Image, ImageDraw

    skeleton = [
        (0, 1), (0, 2), (1, 3), (2, 4),
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15), (12, 14), (14, 16),
    ]
    joint_colors = [
        (255, 200, 87),   (255, 200, 87),   (255, 200, 87),
        (255, 200, 87),   (255, 200, 87),
        (87, 199, 255),   (87, 199, 255),   (87, 199, 255),
        (87, 199, 255),   (255, 135, 87),    (255, 135, 87),
        (150, 255, 150),  (150, 255, 150),
        (150, 150, 255),  (150, 150, 255),
        (255, 87, 199),   (255, 87, 199),
    ]

    all_x = coords[:T, :, 0]
    all_y = coords[:T, :, 1]
    x_min, x_max = np.nanmin(all_x), np.nanmax(all_x)
    y_min, y_max = np.nanmin(all_y), np.nanmax(all_y)

    pad, canvas_w, canvas_h = 20, 400, 500

    def kp_to_canvas(kp):
        scale = min(
            (canvas_w - 2 * pad) / max(x_max - x_min, 1),
            (canvas_h - 2 * pad) / max(y_max - y_min, 1),
        )
        ox = (canvas_w - (x_max - x_min) * scale) / 2 - x_min * scale
        oy = (canvas_h - (y_max - y_min) * scale) / 2 - y_min * scale
        return kp * scale + np.array([ox, oy])

    frames = []
    for i in range(0, T, step):
        img = Image.new('RGB', (canvas_w, canvas_h), (30, 30, 30))
        draw = ImageDraw.Draw(img)
        kp = kp_to_canvas(coords[i])

        for a, b in skeleton:
            pt1 = tuple(kp[a].astype(int))
            pt2 = tuple(kp[b].astype(int))
            draw.line([pt1, pt2], fill=(120, 180, 255), width=3)

        for j in range(17):
            x, y = int(kp[j, 0]), int(kp[j, 1])
            r = 5 if j in [9, 10, 15, 16] else 4
            draw.ellipse([x - r, y - r, x + r, y + r], fill=joint_colors[j])

        draw.text((10, 10), f"F{i}", fill=(200, 200, 200))
        draw.text((10, canvas_h - 20), "Smoothed", fill=(100, 200, 100))
        frames.append(img)

    if frames:
        gif_path = os.path.join(output_dir, "vitpose_skeleton.gif")
        frames[0].save(
            gif_path, save_all=True, append_images=frames[1:],
            duration=int(1000 / gif_fps), loop=0,
        )
        log(f"  -> {gif_path}")


def _generate_hmr2_smpl_gif(rotations, T, step, gif_fps, output_dir):
    """HMR2 SMPL 人体模型 GIF（flip_y=True 修正 HMR2 相机坐标颠倒）"""
    from engines.smpl_renderer import SMPLRenderer
    from pose_extraction.weights_config import SMPL_NEUTRAL
    from PIL import Image

    renderer = SMPLRenderer(SMPL_NEUTRAL)
    indices = list(range(0, T, step))
    n_total = len(indices)

    frames = []
    t0 = time.time()
    for fi, idx in enumerate(indices):
        img = renderer.render_frame(
            rotations[idx], size=(400, 500),
            azimuth=90.0, elevation=5.0,
            flip_y=True,
            stabilize_root=True,
        )
        frames.append(Image.fromarray(img))
        if (fi + 1) % 20 == 0 or fi == n_total - 1:
            elapsed = time.time() - t0
            log(f"    SMPL 渲染: {fi+1}/{n_total} ({elapsed:.1f}s)")

    gif_path = os.path.join(output_dir, "hmr2_body_model.gif")
    if frames:
        frames[0].save(
            gif_path, save_all=True, append_images=frames[1:],
            duration=int(1000 / gif_fps), loop=0,
        )
    log(f"  -> {gif_path}")


def write_report(
    video_path, mode, max_frames, fps,
    n_vit_valid, n_hmr_valid,
    n_vit_outlier, n_hmr_outlier,
    keyframes, beat_times,
    total_time,
):
    """写入测试报告"""
    report_path = os.path.join(OUTPUT_DIR, "test_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("  BeatsMatching 算法管线测试报告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"视频: {video_path}\n")
        f.write(f"模式: {mode}\n")
        f.write(f"测试帧数: {max_frames}\n")
        f.write(f"FPS: {fps:.1f}\n")
        f.write(f"总耗时: {total_time:.1f}s\n\n")

        f.write("--- 推理结果 ---\n")
        if n_vit_valid is not None:
            f.write(f"ViTPose 有效帧: {n_vit_valid}/{max_frames}\n")
        if n_hmr_valid is not None:
            f.write(f"HMR2 有效帧: {n_hmr_valid}/{max_frames}\n")

        f.write("\n--- 异常检测 ---\n")
        if n_vit_outlier is not None:
            f.write(f"ViTPose 异常帧: {n_vit_outlier}\n")
        if n_hmr_outlier is not None:
            f.write(f"HMR2 异常帧: {n_hmr_outlier}\n")

        f.write(f"\n--- 关键帧检测 ---\n")
        f.write(f"关键帧数量: {len(keyframes)}\n")
        f.write(f"关键帧索引: {keyframes}\n")
        if beat_times:
            f.write(f"节拍时间: {[f'{t:.2f}s' for t in beat_times]}\n")

        f.write(f"\n--- 输出文件 ---\n")
        for fname in sorted(os.listdir(OUTPUT_DIR)):
            fpath = os.path.join(OUTPUT_DIR, fname)
            size_kb = os.path.getsize(fpath) / 1024
            f.write(f"  {fname} ({size_kb:.1f} KB)\n")

    log(f"测试报告: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="BeatsMatching 算法管线端到端测试")
    parser.add_argument("--video", type=str, default="test1.mp4", help="测试视频路径")
    parser.add_argument("--mode", type=str, default="joint",
                        choices=["vitpose", "hmr2", "joint"],
                        help="测试模式 (默认: joint)")
    parser.add_argument("--max_frames", type=int, default=0,
                        help="最大测试帧数 (0=全部帧，默认: 0)")
    parser.add_argument("--skip_smpl_test", action="store_true",
                        help="跳过 SMPL 渲染器基础测试")
    args = parser.parse_args()

    video_path = args.video or find_default_video()
    max_frames = args.max_frames  # 0 = all frames
    log(f"视频: {video_path}")
    log(f"模式: {args.mode}")
    log(f"最大帧数: {max_frames if max_frames > 0 else '全部'}")
    log(f"输出目录: {OUTPUT_DIR}")
    log("")

    total_t0 = time.time()

    # 1. SMPL 渲染器基础测试
    if not args.skip_smpl_test:
        test_smpl_renderer()

    # 2-3. ViTPose / HMR2 推理
    coords, valid_vit = None, None
    rotations, valid_hmr = None, None
    fps = 30.0

    if args.mode in ("vitpose", "joint"):
        coords, valid_vit, fps = test_vitpose_pipeline(video_path, max_frames)
        log("")

    if args.mode in ("hmr2", "joint"):
        rotations, valid_hmr, fps = test_hmr2_pipeline(video_path, max_frames)
        log("")

    # 4. 异常检测 + 平滑
    clean_coords, clean_vit_mask, clean_rot, clean_hmr_mask = \
        test_outlier_detection_and_smoothing(
            coords, valid_vit if valid_vit is not None else np.ones(0, dtype=bool),
            rotations, valid_hmr if valid_hmr is not None else np.ones(0, dtype=bool),
            fps,
        )

    # 5. 节拍检测（传入原始数据用于诊断对比）
    keyframes, beat_times, velocities, rot_grads = test_beat_detection(
        clean_coords, clean_rot, fps,
        coords_raw=coords, rotations_raw=rotations,
    )

    # 5.5 缓存信号数据供后续分析
    cache_path = os.path.join(OUTPUT_DIR, "cached_signals.npz")
    cache_data = {"fps": np.array(fps)}
    if velocities is not None:
        cache_data["vel_smooth"] = velocities
    if rot_grads is not None:
        cache_data["rot_smooth"] = rot_grads
    if coords is not None:
        from core.algorithm_config import AlgorithmConfig as _AC
        _cfg = _AC()
        _w = _make_mock_worker(_cfg)
        cache_data["vel_raw"] = _w._compute_weighted_velocity(coords)
    if rotations is not None:
        if "_w" not in dir():
            from core.algorithm_config import AlgorithmConfig as _AC
            _cfg = _AC()
            _w = _make_mock_worker(_cfg)
        cache_data["rot_raw"] = _w._compute_rotation_metric(rotations)
    np.savez(cache_path, **cache_data)
    log(f"信号缓存已保存: {cache_path}")

    # 6. GIF 生成（使用平滑后数据）
    test_gif_generation(clean_coords, clean_rot, fps)

    total_time = time.time() - total_t0

    # 7. 报告
    n_vit_outlier = (~clean_vit_mask).sum() if clean_vit_mask is not None else None
    n_hmr_outlier = (~clean_hmr_mask).sum() if clean_hmr_mask is not None else None
    n_vit_valid = valid_vit.sum() if valid_vit is not None else None
    n_hmr_valid = valid_hmr.sum() if valid_hmr is not None else None

    n_total = len(coords) if coords is not None else (len(rotations) if rotations is not None else 0)

    write_report(
        video_path, args.mode, n_total, fps,
        n_vit_valid, n_hmr_valid,
        n_vit_outlier, n_hmr_outlier,
        keyframes, beat_times,
        total_time,
    )

    log("")
    log("=" * 60)
    log(f"全部测试完成！总耗时: {total_time:.1f}s")
    log(f"输出目录: {OUTPUT_DIR}")
    log("=" * 60)


if __name__ == "__main__":
    main()
