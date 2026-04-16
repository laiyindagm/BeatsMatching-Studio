"""
benchmark_layered_accel.py
===========================
推理加速逐层叠加基准测试 —— 用于论文表 5-7

测量 5 种配置（逐层叠加）下 ViTPose / HMR2 的 ms/帧和总吞吐量：
  0) 基线：单帧推理（独立 YOLO）
  1) +SharedYOLO
  2) +批量推理 (bs=8)
  3) +FP16（仅 HMR2）
  4) +帧预取流水线

用法：
  conda activate pose_unified
  cd D:\\毕业设计\\BeatsMatching
  python benchmark_layered_accel.py
"""

import os, sys, time, json, warnings
import numpy as np
import cv2
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

# ── 查找测试视频 ────────────────────────────────────────────────────────────
def _find_video():
    candidates = [
        os.path.join(os.path.dirname(_HERE), "test_videos",
                     "aist_gBR_sBM_c01_d04_mBR0_ch01.mp4"),
        os.path.join(_HERE, "test1.mp4"),
        os.path.join(_HERE, "output.mp4"),
    ]
    td = os.path.join(os.path.dirname(_HERE), "test_videos")
    if os.path.isdir(td):
        for f in sorted(os.listdir(td)):
            if f.endswith(".mp4"):
                candidates.append(os.path.join(td, f))
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


# ── 读取 N 帧 ───────────────────────────────────────────────────────────────
def _load_frames(video_path, n=100):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames = []
    for _ in range(n):
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()
    return frames, fps


# ══════════════════════════════════════════════════════════════════════════════
#  模型加载（只加载一次）
# ══════════════════════════════════════════════════════════════════════════════
def _load_models():
    from pose_extraction.weights_config import (
        VITPOSE_MODEL_ID, HMR2_CHECKPOINT, YOLO_WEIGHTS, HF_HOME,
    )
    os.environ["HF_HOME"] = HF_HOME
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

    print("[Bench] 加载 ViTPose…")
    from pose_extraction.vitpose.detector import ViTPoseDetector
    vit = ViTPoseDetector(vitpose_model=VITPOSE_MODEL_ID,
                          yolo_model=YOLO_WEIGHTS)

    print("[Bench] 加载 HMR2…")
    from pose_extraction.hmr2.reconstructor import HMR2Reconstructor
    hmr = HMR2Reconstructor(checkpoint_path=HMR2_CHECKPOINT,
                            yolo_model=YOLO_WEIGHTS)

    return vit, hmr, YOLO_WEIGHTS


# ── GPU warmup ───────────────────────────────────────────────────────────────
def _warmup(vit, hmr, frames, n_warmup=3):
    """预热模型，让 CUDA 分配内存 / JIT 等初始化开销消失"""
    print("[Bench] 预热…")
    for i in range(min(n_warmup, len(frames))):
        vit.detect_frame(frames[i], frame_idx=i)
        hmr.reconstruct_frame(frames[i], frame_idx=i)
    torch.cuda.synchronize()


# ══════════════════════════════════════════════════════════════════════════════
#  逐层基准
# ══════════════════════════════════════════════════════════════════════════════

def bench_baseline(vit, hmr, frames, N):
    """层0: 基线——单帧推理，各自独立 YOLO"""
    torch.cuda.synchronize()

    # ViTPose 逐帧
    t0 = time.perf_counter()
    for i in range(N):
        vit.detect_frame(frames[i], frame_idx=i)
    torch.cuda.synchronize()
    vit_ms = (time.perf_counter() - t0) / N * 1000

    # HMR2 逐帧
    t0 = time.perf_counter()
    for i in range(N):
        hmr.reconstruct_frame(frames[i], frame_idx=i)
    torch.cuda.synchronize()
    hmr_ms = (time.perf_counter() - t0) / N * 1000

    return vit_ms, hmr_ms


def bench_shared_yolo(vit, hmr, frames, N, yolo_path):
    """层1: +SharedYOLO（同一帧只检测一次）"""
    from engines.inference_accelerator import SharedYOLO
    SharedYOLO._instance = None   # 重置单例
    shared = SharedYOLO(yolo_path)
    torch.cuda.synchronize()

    # ViTPose + SharedYOLO
    shared.clear_cache()
    t0 = time.perf_counter()
    for i in range(N):
        img_rgb = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
        boxes = shared.detect(img_rgb, frame_idx=i)
        vit.detect_frame(frames[i], frame_idx=i, predetected_boxes=boxes)
    torch.cuda.synchronize()
    vit_ms = (time.perf_counter() - t0) / N * 1000

    # HMR2 + SharedYOLO（缓存命中）
    t0 = time.perf_counter()
    for i in range(N):
        img_rgb = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
        boxes = shared.detect(img_rgb, frame_idx=i)  # should hit cache
        hmr.reconstruct_frame(frames[i], frame_idx=i, predetected_boxes=boxes)
    torch.cuda.synchronize()
    hmr_ms = (time.perf_counter() - t0) / N * 1000

    print(f"    {shared.cache_stats}")
    return vit_ms, hmr_ms


def bench_batch(vit, hmr, frames, N, yolo_path, bs=8):
    """层2: +SharedYOLO + 批量推理 (bs=8)"""
    from engines.inference_accelerator import (
        SharedYOLO, AcceleratedViTPose, AcceleratedHMR2, InferenceConfig,
    )
    SharedYOLO._instance = None
    shared = SharedYOLO(yolo_path)
    cfg = InferenceConfig(use_batch=True, batch_size=bs,
                          use_fp16=False, use_compile=False,
                          shared_yolo=True)
    acc_vit = AcceleratedViTPose(vit, cfg, shared)
    torch.cuda.synchronize()

    # ViTPose 批推理
    shared.clear_cache()
    t0 = time.perf_counter()
    for i in range(N):
        acc_vit.detect_frame(frames[i], frame_idx=i)
    acc_vit.flush()
    torch.cuda.synchronize()
    vit_ms = (time.perf_counter() - t0) / N * 1000

    # HMR2 批推理 (FP16=False)
    acc_hmr = AcceleratedHMR2(hmr, cfg, shared)
    shared.clear_cache()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(N):
        acc_hmr.reconstruct_frame(frames[i], frame_idx=i)
    acc_hmr.flush()
    torch.cuda.synchronize()
    hmr_ms = (time.perf_counter() - t0) / N * 1000

    return vit_ms, hmr_ms


def bench_batch_fp16(vit, hmr, frames, N, yolo_path, bs=8):
    """层3: +SharedYOLO + 批量推理 + FP16 (HMR2 only)"""
    from engines.inference_accelerator import (
        SharedYOLO, AcceleratedViTPose, AcceleratedHMR2, InferenceConfig,
    )
    SharedYOLO._instance = None
    shared = SharedYOLO(yolo_path)

    # ViTPose: 同层2（FP16 不适用）
    cfg_vit = InferenceConfig(use_batch=True, batch_size=bs,
                              use_fp16=False, use_compile=False,
                              shared_yolo=True)
    acc_vit = AcceleratedViTPose(vit, cfg_vit, shared)
    torch.cuda.synchronize()

    shared.clear_cache()
    t0 = time.perf_counter()
    for i in range(N):
        acc_vit.detect_frame(frames[i], frame_idx=i)
    acc_vit.flush()
    torch.cuda.synchronize()
    vit_ms = (time.perf_counter() - t0) / N * 1000

    # HMR2: 批推理 + FP16
    cfg_hmr = InferenceConfig(use_batch=True, batch_size=bs,
                              use_fp16=True, use_compile=False,
                              shared_yolo=True)
    acc_hmr = AcceleratedHMR2(hmr, cfg_hmr, shared)
    shared.clear_cache()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(N):
        acc_hmr.reconstruct_frame(frames[i], frame_idx=i)
    acc_hmr.flush()
    torch.cuda.synchronize()
    hmr_ms = (time.perf_counter() - t0) / N * 1000

    return vit_ms, hmr_ms


def bench_full_pipeline(vit, hmr, frames, N, yolo_path, video_path, bs=8):
    """层4: +SharedYOLO + 批量推理 + FP16 + 帧预取"""
    from engines.inference_accelerator import (
        SharedYOLO, AcceleratedViTPose, AcceleratedHMR2,
        InferenceConfig, FramePrefetcher,
    )
    SharedYOLO._instance = None
    shared = SharedYOLO(yolo_path)
    cfg_vit = InferenceConfig(use_batch=True, batch_size=bs,
                              use_fp16=False, use_compile=False,
                              shared_yolo=True)
    cfg_hmr = InferenceConfig(use_batch=True, batch_size=bs,
                              use_fp16=True, use_compile=False,
                              shared_yolo=True)
    acc_vit = AcceleratedViTPose(vit, cfg_vit, shared)
    acc_hmr = AcceleratedHMR2(hmr, cfg_hmr, shared)
    shared.clear_cache()

    # 帧预取
    pf = FramePrefetcher(video_path, max_queue=32, end_frame=N)
    pf.start()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    count = 0
    while True:
        idx, frame = pf.get(timeout=15.0)
        if frame is None:
            break
        acc_vit.detect_frame(frame, frame_idx=idx)
        acc_hmr.reconstruct_frame(frame, frame_idx=idx)
        count += 1
        if count >= N:
            break
    acc_vit.flush()
    acc_hmr.flush()
    torch.cuda.synchronize()
    total_s = time.perf_counter() - t0

    pf.stop()
    vit_ms = total_s / max(count, 1) * 1000 * 0.4   # 近似 40%归ViTPose
    hmr_ms = total_s / max(count, 1) * 1000 * 0.6   # 近似 60%归HMR2
    total_fps = count / max(total_s, 1e-9)
    return vit_ms, hmr_ms, total_fps, total_s, count


# ══════════════════════════════════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════════════════════════════════

def main():
    video_path = _find_video()
    if not video_path:
        print("错误: 未找到测试视频")
        sys.exit(1)

    N = 50  # 测试帧数（兼顾准确性与时间）
    BS = 8

    print(f"═══ 推理加速逐层基准测试 ═══")
    print(f"GPU : {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
    print(f"PyTorch: {torch.__version__}")
    print(f"视频: {video_path}")
    print(f"测试帧数: {N}, batch_size: {BS}")

    frames, fps = _load_frames(video_path, N)
    if len(frames) < N:
        N = len(frames)
        print(f"  实际可用帧数: {N}")

    vit, hmr, yolo_path = _load_models()
    _warmup(vit, hmr, frames)

    results = {}

    # ── 层0: 基线 ──
    print("\n[层0] 基线（单帧推理，独立 YOLO）")
    v0, h0 = bench_baseline(vit, hmr, frames, N)
    fps0 = 1000.0 / (v0 + h0)
    results["baseline"] = {"vit_ms": v0, "hmr_ms": h0, "fps": fps0}
    print(f"    ViTPose: {v0:.1f} ms/帧, HMR2: {h0:.1f} ms/帧, 总: {fps0:.1f} fps")

    # ── 层1: +SharedYOLO ──
    print("\n[层1] +SharedYOLO")
    v1, h1 = bench_shared_yolo(vit, hmr, frames, N, yolo_path)
    fps1 = 1000.0 / (v1 + h1)
    results["shared_yolo"] = {"vit_ms": v1, "hmr_ms": h1, "fps": fps1}
    print(f"    ViTPose: {v1:.1f} ms/帧, HMR2: {h1:.1f} ms/帧, 总: {fps1:.1f} fps")

    # ── 层2: +批量推理 ──
    print("\n[层2] +批量推理 (bs=8)")
    v2, h2 = bench_batch(vit, hmr, frames, N, yolo_path, BS)
    fps2 = 1000.0 / (v2 + h2)
    results["batch"] = {"vit_ms": v2, "hmr_ms": h2, "fps": fps2}
    print(f"    ViTPose: {v2:.1f} ms/帧, HMR2: {h2:.1f} ms/帧, 总: {fps2:.1f} fps")

    # ── 层3: +FP16 ──
    print("\n[层3] +FP16（仅 HMR2）")
    v3, h3 = bench_batch_fp16(vit, hmr, frames, N, yolo_path, BS)
    fps3 = 1000.0 / (v3 + h3)
    results["fp16"] = {"vit_ms": v3, "hmr_ms": h3, "fps": fps3}
    print(f"    ViTPose: {v3:.1f} ms/帧, HMR2: {h3:.1f} ms/帧, 总: {fps3:.1f} fps")

    # ── 层4: +帧预取 ──
    print("\n[层4] +帧预取流水线")
    v4, h4, fps4, total_s, cnt = bench_full_pipeline(
        vit, hmr, frames, N, yolo_path, video_path, BS)
    results["prefetch"] = {"vit_ms": v4, "hmr_ms": h4, "fps": fps4,
                           "total_s": total_s, "frames": cnt}
    print(f"    端到端: {cnt} 帧 / {total_s:.2f}s = {fps4:.1f} fps")
    print(f"    (ViTPose≈{v4:.1f} ms, HMR2≈{h4:.1f} ms, 均为近似)")

    # ── 汇总表 ──
    print("\n" + "=" * 72)
    print(f"{'配置':<24} {'ViTPose(ms)':>12} {'HMR2(ms)':>12} {'总吞吐(fps)':>14}")
    print("-" * 72)
    rows = [
        ("基线（单帧推理）",   results["baseline"]),
        ("+SharedYOLO",       results["shared_yolo"]),
        ("+批量推理(bs=8)",   results["batch"]),
        ("+FP16(仅HMR2)",    results["fp16"]),
        ("+帧预取流水线",     results["prefetch"]),
    ]
    for label, d in rows:
        print(f"  {label:<22} {d['vit_ms']:>10.1f}  {d['hmr_ms']:>10.1f}  {d['fps']:>12.1f}")
    print("=" * 72)
    print(f"总加速比: {results['prefetch']['fps'] / results['baseline']['fps']:.2f}x")

    # ── 保存 JSON ──
    out_dir = os.path.join(_HERE, "test_output", "thesis_experiments")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "accel_benchmark.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
