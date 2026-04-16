"""
端到端推理测试脚本
==================

测试 ViTPose（2D 关节） + HMR2（3D SMPL）两条管线对视频的完整推理。

使用方法：
    conda activate pose_unified
    python pose_extraction/scripts/test_pipeline.py [视频路径] [--mode vitpose|hmr2|both] [--frames N] [--save_vis]

示例：
    # 测试 ViTPose（不需要下载 HMR2 权重）
    python pose_extraction/scripts/test_pipeline.py d:/毕业设计/BeatsMatching/output.mp4 --mode vitpose --frames 30

    # 测试 HMR2（需先下载权重）
    python pose_extraction/scripts/test_pipeline.py d:/毕业设计/BeatsMatching/output.mp4 --mode hmr2 --frames 10

    # 两条管线都测试
    python pose_extraction/scripts/test_pipeline.py d:/毕业设计/BeatsMatching/output.mp4 --mode both --frames 20 --save_vis
"""

import sys
import os
import time
import argparse
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch

# ── 确保 pose_extraction 可被导入 ────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent  # d:/毕业设计
sys.path.insert(0, str(_ROOT))


def test_vitpose(video_path: str, max_frames: int = 30, save_vis: bool = False) -> dict:
    """测试 ViTPose 管线。"""
    print("\n" + "="*60)
    print("测试 ViTPose 2D 关节检测管线")
    print("="*60)

    from pose_extraction.vitpose.detector import ViTPoseDetector

    # 初始化检测器（自动下载 HuggingFace 权重）
    t0 = time.time()
    detector = ViTPoseDetector(
        vitpose_model="usyd-community/vitpose-base-simple",
        device="auto",
        yolo_model="yolov8n.pt",
    )
    print(f"✅ 模型加载耗时：{time.time()-t0:.1f}s  |  设备：{detector.device}")

    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频：{video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"视频：{os.path.basename(video_path)}")
    print(f"  分辨率：{W}×{H}，FPS：{fps:.1f}，总帧数：{total_frames}")
    print(f"  测试帧数：{min(max_frames, total_frames)}")

    # 可视化输出
    out_writer = None
    if save_vis:
        out_path = str(Path(video_path).parent / f"vitpose_vis_{Path(video_path).stem}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
        print(f"  可视化输出：{out_path}")

    # 逐帧推理
    frame_results = []
    inference_times = []
    frame_count = 0

    while frame_count < max_frames:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        timestamp = frame_count / fps
        t1 = time.time()
        result = detector.detect_frame(frame_bgr, frame_idx=frame_count, timestamp=timestamp)
        dt = time.time() - t1
        inference_times.append(dt)

        frame_results.append(result)

        if save_vis and out_writer is not None:
            vis = detector.visualize_frame(frame_bgr, result)
            out_writer.write(vis)

        if frame_count % 10 == 0:
            n_persons = len(result.persons)
            print(f"  帧 {frame_count:4d} | {dt*1000:5.1f}ms | 检测到人数：{n_persons}")

        frame_count += 1

    cap.release()
    if out_writer:
        out_writer.release()

    # 统计
    total_detections = sum(len(r.persons) for r in frame_results)
    avg_time = np.mean(inference_times) * 1000
    max_time = np.max(inference_times) * 1000
    print(f"\n✅ ViTPose 测试完成")
    print(f"  处理帧数：{frame_count}")
    print(f"  总检测人次：{total_detections}")
    print(f"  平均推理时间：{avg_time:.1f}ms/帧")
    print(f"  最大推理时间：{max_time:.1f}ms/帧")
    print(f"  估计实时帧率：{1000/avg_time:.1f} FPS")

    # 输出示例关节数据
    if frame_results:
        first_result = frame_results[0]
        if first_result.persons:
            pid = list(first_result.persons.keys())[0]
            kpts = first_result.persons[pid]
            print(f"\n  第0帧第0人关节数据（前3个关节）：")
            for j in range(min(3, kpts.shape[0])):
                print(f"    关节{j}: x={kpts[j,0]:.1f}, y={kpts[j,1]:.1f}, conf={kpts[j,2]:.3f}")

    return {
        "frames": frame_count,
        "avg_ms": avg_time,
        "total_detections": total_detections,
        "results": frame_results,
    }


def test_hmr2(video_path: str, max_frames: int = 10, save_vis: bool = False) -> dict:
    """测试 HMR2 3D 重建管线。"""
    print("\n" + "="*60)
    print("测试 HMR2 3D 人体重建管线")
    print("="*60)

    from pose_extraction.hmr2.reconstructor import HMR2Reconstructor

    # 检查权重
    from hmr2.models import DEFAULT_CHECKPOINT
    if not Path(DEFAULT_CHECKPOINT).exists():
        print(f"⚠️  权重文件不存在：{DEFAULT_CHECKPOINT}")
        print("   请先运行下载脚本：")
        print("   conda activate pose_unified")
        print("   python pose_extraction/scripts/download_weights.py")
        return {"error": "weights not found"}

    # 初始化重建器
    t0 = time.time()
    reconstructor = HMR2Reconstructor(
        device="auto",
        yolo_model="yolov8n.pt",
    )
    print(f"✅ 模型加载耗时：{time.time()-t0:.1f}s  |  设备：{reconstructor.device}")

    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频：{video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"视频：{os.path.basename(video_path)}")
    print(f"  分辨率：{W}×{H}，FPS：{fps:.1f}，总帧数：{total_frames}")
    print(f"  测试帧数：{min(max_frames, total_frames)}")

    # 逐帧推理
    all_results = []
    inference_times = []
    frame_count = 0

    while frame_count < max_frames:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        timestamp = frame_count / fps
        t1 = time.time()
        results = reconstructor.reconstruct_frame(frame_bgr, frame_idx=frame_count, timestamp=timestamp)
        dt = time.time() - t1
        inference_times.append(dt)

        all_results.append(results)

        if frame_count % 5 == 0:
            n_persons = len(results)
            print(f"  帧 {frame_count:4d} | {dt*1000:5.1f}ms | 重建人数：{n_persons}")

        frame_count += 1

    cap.release()

    # 统计
    total_recon = sum(len(r) for r in all_results)
    avg_time = np.mean(inference_times) * 1000 if inference_times else 0
    print(f"\n✅ HMR2 测试完成")
    print(f"  处理帧数：{frame_count}")
    print(f"  总重建人次：{total_recon}")
    print(f"  平均推理时间：{avg_time:.1f}ms/帧")

    # 输出示例 SMPL 参数
    for frame_results in all_results:
        if frame_results:
            r = frame_results[0]
            print(f"\n  第0帧第0人 SMPL 参数示例：")
            if r.global_orient is not None:
                print(f"    global_orient (轴角): {r.global_orient.ravel()[:3]}")
            if r.body_pose is not None:
                print(f"    body_pose shape: {r.body_pose.shape}")
            if r.betas is not None:
                print(f"    betas (前5): {r.betas[:5]}")
            if r.joints_3d is not None:
                print(f"    joints_3d shape: {r.joints_3d.shape}")
            break

    return {
        "frames": frame_count,
        "avg_ms": avg_time,
        "total_recon": total_recon,
        "results": all_results,
    }


def test_beat_detection(video_path: str, mode: str = "joint") -> None:
    """测试完整节拍检测管线（ViTPose→节拍）。"""
    print("\n" + "="*60)
    print("测试节拍检测管线（ViTPose → 节拍时间点）")
    print("="*60)

    # 用视频处理器处理整个视频
    from pose_extraction.vitpose.video_processor import ViTPoseVideoProcessor
    from pose_extraction.core.beat_detector import BeatDetector, BeatDetectorConfig, DetectionMode

    processor = ViTPoseVideoProcessor(device="auto")
    print("处理视频中...")
    seq = processor.process_video(video_path, max_frames=60, skip_frames=3)
    print(f"✅ 提取了 {seq.num_frames} 帧的关节数据")

    # 节拍检测
    cfg = BeatDetectorConfig(mode=DetectionMode.JOINT_VELOCITY)
    detector = BeatDetector(cfg)
    beat_frames = detector.detect(seq)
    fps = seq.fps
    beat_times = [f / fps for f in beat_frames]

    print(f"✅ 检测到 {len(beat_frames)} 个节拍帧")
    if beat_times:
        print(f"  节拍时间点（秒）: {[f'{t:.2f}' for t in beat_times[:10]]}")
        print(f"  平均间隔：{np.diff(beat_times).mean():.3f}s" if len(beat_times) > 1 else "")


def main():
    parser = argparse.ArgumentParser(description="BeatsMatching 姿态管线端到端测试")
    parser.add_argument("video", nargs="?",
                        default=r"d:\毕业设计\BeatsMatching\output.mp4",
                        help="视频文件路径")
    parser.add_argument("--mode", choices=["vitpose", "hmr2", "both", "beat"],
                        default="vitpose", help="测试模式")
    parser.add_argument("--frames", type=int, default=30,
                        help="最大测试帧数")
    parser.add_argument("--save_vis", action="store_true",
                        help="保存可视化视频")
    args = parser.parse_args()

    # 确认视频存在
    if not os.path.exists(args.video):
        # 尝试 output1.mp4
        alt = os.path.join(os.path.dirname(args.video), "output1.mp4")
        if os.path.exists(alt):
            args.video = alt
        else:
            print(f"❌ 视频不存在：{args.video}")
            sys.exit(1)

    print(f"视频路径：{args.video}")
    print(f"GPU：{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    if args.mode in ("vitpose", "both"):
        test_vitpose(args.video, args.frames, args.save_vis)

    if args.mode in ("hmr2", "both"):
        test_hmr2(args.video, min(args.frames, 10), args.save_vis)

    if args.mode == "beat":
        test_beat_detection(args.video)

    print("\n✅ 所有测试完成！")


if __name__ == "__main__":
    main()
