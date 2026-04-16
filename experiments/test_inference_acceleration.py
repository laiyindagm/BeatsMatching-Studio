"""
test_inference_acceleration.py
===================================
推理加速效果基准测试

测试内容：
  1. SharedYOLO 缓存命中率
  2. ViTPose 真正批推理 vs 逐帧推理
  3. HMR2 真正批推理 vs 逐帧推理
  4. FramePrefetcher 帧预取效果
  5. 端到端对比：加速前 vs 加速后

用法：
  conda activate pose_unified
  cd D:\毕业设计\BeatsMatching
  python test_inference_acceleration.py [video_path]
"""

import os
import sys
import time
import argparse

import numpy as np
import cv2
import torch

# 路径设置
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE) if os.path.basename(_HERE) != "BeatsMatching" else os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _REPO)


def find_test_video():
    """查找可用的测试视频"""
    candidates = [
        os.path.join(_HERE, "test_output", "test_video.mp4"),
        os.path.join(os.path.dirname(_HERE), "test_output", "test_video.mp4"),
    ]
    # 搜索 test_output 目录
    test_dir = os.path.join(_HERE, "test_output")
    if os.path.isdir(test_dir):
        for f in os.listdir(test_dir):
            if f.endswith((".mp4", ".avi", ".mov")):
                candidates.append(os.path.join(test_dir, f))
    # 搜索上级目录
    parent_test = os.path.join(os.path.dirname(_HERE), "test_output")
    if os.path.isdir(parent_test):
        for f in os.listdir(parent_test):
            if f.endswith((".mp4", ".avi", ".mov")):
                candidates.append(os.path.join(parent_test, f))
    
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def benchmark_prefetcher(video_path: str, n_frames: int = 100):
    """对比 FramePrefetcher 和 cv2.VideoCapture 的读帧速度"""
    from engines.inference_accelerator import FramePrefetcher
    
    print("\n" + "=" * 60)
    print("  基准测试 1: 帧预取 (FramePrefetcher vs cv2.VideoCapture)")
    print("=" * 60)
    
    # cv2 顺序读取
    cap = cv2.VideoCapture(video_path)
    total = min(n_frames, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    
    t0 = time.time()
    frames_cv2 = []
    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        frames_cv2.append(frame)
    cap.release()
    t_cv2 = time.time() - t0
    
    # FramePrefetcher
    pf = FramePrefetcher(video_path, max_queue=32, end_frame=total)
    t0 = time.time()
    pf.start()
    frames_pf = []
    while True:
        idx, frame = pf.get()
        if frame is None:
            break
        frames_pf.append(frame)
    pf.stop()
    t_pf = time.time() - t0
    
    print(f"  cv2 顺序读取: {len(frames_cv2)} 帧, {t_cv2*1000:.1f}ms ({len(frames_cv2)/t_cv2:.0f} fps)")
    print(f"  FramePrefetcher: {len(frames_pf)} 帧, {t_pf*1000:.1f}ms ({len(frames_pf)/t_pf:.0f} fps)")
    print(f"  加速比: {t_cv2/t_pf:.2f}x")
    
    return frames_cv2[:total]


def benchmark_shared_yolo(frames: list, video_path: str):
    """测试 SharedYOLO 缓存机制"""
    print("\n" + "=" * 60)
    print("  基准测试 2: SharedYOLO 缓存命中率")
    print("=" * 60)
    
    from engines.inference_accelerator import SharedYOLO
    from pose_extraction.weights_config import YOLO_WEIGHTS
    
    shared = SharedYOLO(YOLO_WEIGHTS)
    n = min(len(frames), 30)
    
    # 第一轮：所有 miss
    t0 = time.time()
    for i in range(n):
        image_rgb = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
        shared.detect(image_rgb, frame_idx=i)
    t_first = time.time() - t0
    
    # 第二轮：所有 hit
    t0 = time.time()
    for i in range(n):
        image_rgb = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
        shared.detect(image_rgb, frame_idx=i)
    t_second = time.time() - t0
    
    print(f"  首次检测 ({n} 帧): {t_first*1000:.1f}ms ({t_first/n*1000:.1f}ms/帧)")
    print(f"  缓存命中 ({n} 帧): {t_second*1000:.1f}ms ({t_second/n*1000:.1f}ms/帧)")
    print(f"  缓存加速比: {t_first/max(t_second, 1e-9):.0f}x")
    print(f"  {shared.cache_stats}")
    
    return shared


def benchmark_vitpose(frames: list, shared_yolo=None):
    """测试 ViTPose 真正批推理 vs 逐帧"""
    print("\n" + "=" * 60)
    print("  基准测试 3: ViTPose 推理 (批推理 vs 逐帧)")
    print("=" * 60)
    
    try:
        from pose_extraction.vitpose.detector import ViTPoseDetector
        from pose_extraction.weights_config import VITPOSE_MODEL_ID, YOLO_WEIGHTS, HF_HOME
        from engines.inference_accelerator import (
            AcceleratedViTPose, InferenceConfig,
        )
        
        os.environ["HF_HOME"] = HF_HOME
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        
        n = min(len(frames), 16)
        
        print(f"  加载 ViTPose 模型...")
        detector = ViTPoseDetector(
            vitpose_model=VITPOSE_MODEL_ID,
            yolo_model=YOLO_WEIGHTS,
        )
        
        # 预热
        print(f"  预热 (1 帧)...")
        _ = detector.detect_frame(frames[0], frame_idx=0, timestamp=0.0)
        
        # 逐帧推理
        t0 = time.time()
        results_seq = []
        for i in range(n):
            r = detector.detect_frame(frames[i], frame_idx=i, timestamp=float(i))
            results_seq.append(r)
        t_seq = time.time() - t0
        
        # 真正批推理（通过 AcceleratedViTPose）
        config = InferenceConfig(use_batch=True, batch_size=n, use_compile=False)
        acc = AcceleratedViTPose(detector, config, shared_yolo)
        
        t0 = time.time()
        for i in range(n):
            acc.detect_frame(frames[i], frame_idx=i, timestamp=float(i))
        results_batch = acc.flush()
        t_batch = time.time() - t0
        
        print(f"  逐帧推理 ({n} 帧): {t_seq*1000:.1f}ms ({t_seq/n*1000:.1f}ms/帧)")
        print(f"  批量推理 ({n} 帧): {t_batch*1000:.1f}ms ({t_batch/n*1000:.1f}ms/帧)")
        print(f"  加速比: {t_seq/max(t_batch, 1e-9):.2f}x")
        print(f"  {acc.stats}")
        
        # 验证结果一致性
        if len(results_batch) == len(results_seq):
            match = 0
            for r_s, r_b in zip(results_seq, results_batch):
                if len(r_s.persons) == len(r_b.persons):
                    match += 1
            print(f"  结果一致性: {match}/{len(results_seq)} 帧匹配")
    
    except Exception as e:
        print(f"  ViTPose 测试跳过: {e}")


def benchmark_hmr2(frames: list, shared_yolo=None):
    """测试 HMR2 批推理 vs 逐帧"""
    print("\n" + "=" * 60)
    print("  基准测试 4: HMR2 推理 (批推理 vs 逐帧)")
    print("=" * 60)
    
    try:
        from pose_extraction.hmr2.reconstructor import HMR2Reconstructor
        from pose_extraction.weights_config import HMR2_CHECKPOINT, YOLO_WEIGHTS
        from engines.inference_accelerator import (
            AcceleratedHMR2, InferenceConfig,
        )
        
        n = min(len(frames), 8)
        
        print(f"  加载 HMR2 模型...")
        recon = HMR2Reconstructor(
            checkpoint_path=HMR2_CHECKPOINT,
            yolo_model=YOLO_WEIGHTS,
        )
        
        # 预热
        print(f"  预热 (1 帧)...")
        _ = recon.reconstruct_frame(frames[0], frame_idx=0, timestamp=0.0)
        
        # 逐帧推理
        t0 = time.time()
        results_seq = []
        for i in range(n):
            r = recon.reconstruct_frame(frames[i], frame_idx=i, timestamp=float(i))
            results_seq.append(r)
        t_seq = time.time() - t0
        
        # 批推理
        config = InferenceConfig(use_batch=True, batch_size=n, use_fp16=True, use_compile=False)
        acc = AcceleratedHMR2(recon, config, shared_yolo)
        
        t0 = time.time()
        for i in range(n):
            acc.reconstruct_frame(frames[i], frame_idx=i, timestamp=float(i))
        results_batch = acc.flush()
        t_batch = time.time() - t0
        
        print(f"  逐帧推理 ({n} 帧): {t_seq*1000:.1f}ms ({t_seq/n*1000:.1f}ms/帧)")
        print(f"  批量推理 ({n} 帧): {t_batch*1000:.1f}ms ({t_batch/n*1000:.1f}ms/帧)")
        print(f"  加速比: {t_seq/max(t_batch, 1e-9):.2f}x")
        print(f"  {acc.stats}")
    
    except Exception as e:
        print(f"  HMR2 测试跳过: {e}")


def main():
    parser = argparse.ArgumentParser(description="推理加速基准测试")
    parser.add_argument("video", nargs="?", default=None, help="视频文件路径")
    parser.add_argument("-n", type=int, default=50, help="测试帧数")
    args = parser.parse_args()
    
    video_path = args.video or find_test_video()
    if video_path is None:
        print("错误: 未找到测试视频。请提供视频路径作为参数。")
        sys.exit(1)
    
    print(f"测试视频: {video_path}")
    print(f"GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
    print(f"PyTorch: {torch.__version__}")
    
    # 基准测试 1: 帧预取
    frames = benchmark_prefetcher(video_path, n_frames=args.n)
    
    if len(frames) < 2:
        print("错误: 视频帧数不足")
        sys.exit(1)
    
    # 基准测试 2: SharedYOLO
    shared_yolo = benchmark_shared_yolo(frames, video_path)
    
    # 基准测试 3: ViTPose
    benchmark_vitpose(frames, shared_yolo)
    
    # 基准测试 4: HMR2
    benchmark_hmr2(frames, shared_yolo)
    
    print("\n" + "=" * 60)
    print("  基准测试完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
