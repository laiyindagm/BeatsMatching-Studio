"""
pose_extraction/scripts/run_hmr2.py

HMR2 命令行入口：对单个视频文件运行三维人体重建，提取 SMPL 关节旋转量时序序列。

用法：
    conda activate pose_hmr2
    python -m pose_extraction.scripts.run_hmr2 \\
        --video dance.mp4 \\
        --output output/dance_hmr2.pkl \\
        --device cuda

    # 使用更大的模型：
    python -m pose_extraction.scripts.run_hmr2 \\
        --video dance.mp4 \\
        --model hmr2l \\
        --skip_frames 1

    # 指定本地 checkpoint：
    python -m pose_extraction.scripts.run_hmr2 \\
        --video dance.mp4 \\
        --checkpoint ./checkpoints/hmr2_b.ckpt
"""

import argparse
import pickle
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pose_extraction.hmr2 import HMR2VideoProcessor


def parse_args():
    p = argparse.ArgumentParser(description="HMR2 视频三维人体重建")
    p.add_argument("--video", required=True, help="输入视频路径")
    p.add_argument("--output", default=None, help="输出文件路径（.pkl 或 .npz）")
    p.add_argument("--format", default="pkl", choices=["pkl", "npz"], help="输出格式")
    p.add_argument("--model", default="hmr2b", choices=["hmr2b", "hmr2l"], help="HMR2 模型版本")
    p.add_argument("--checkpoint", default=None, help="本地 .ckpt 文件路径（None=自动下载）")
    p.add_argument("--device", default="auto", help="计算设备：auto / cuda / cpu")
    p.add_argument("--yolo_model", default="yolov8n.pt", help="YOLO 权重路径")
    p.add_argument("--conf_threshold", type=float, default=0.35, help="YOLO 置信度阈值")
    p.add_argument("--skip_frames", type=int, default=0, help="跳帧数（0=逐帧）")
    p.add_argument("--start_frame", type=int, default=0, help="起始帧")
    p.add_argument("--end_frame", type=int, default=None, help="结束帧（默认末尾）")
    p.add_argument("--person_id", type=int, default=0, help="目标人物 ID")
    p.add_argument("--save_vertices", action="store_true", help="是否保存 SMPL 网格顶点")
    return p.parse_args()


def main():
    args = parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"[错误] 视频文件不存在：{args.video}")
        sys.exit(1)

    if args.output is None:
        ext = ".npz" if args.format == "npz" else ".pkl"
        args.output = str(video_path.parent / (video_path.stem + f"_hmr2{ext}"))

    print(f"[run_hmr2] 视频：{args.video}")
    print(f"[run_hmr2] 输出：{args.output}")
    print(f"[run_hmr2] 模型：{args.model}，设备：{args.device}，跳帧：{args.skip_frames}")

    processor = HMR2VideoProcessor(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        device=args.device,
        yolo_model=args.yolo_model,
        conf_threshold=args.conf_threshold,
        skip_frames=args.skip_frames,
        save_vertices=args.save_vertices,
    )

    sequence = processor.process_video(
        video_path=args.video,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        person_id=args.person_id,
    )

    processor.save(sequence, args.output, format=args.format)

    # 导出 PoseTimeSeries
    pts = processor.to_pose_time_series(sequence, person_id=args.person_id)
    pts_path = str(Path(args.output).with_suffix("")) + "_pts.pkl"
    with open(pts_path, "wb") as f:
        pickle.dump(pts, f)
    print(f"[run_hmr2] PoseTimeSeries 已保存：{pts_path}")
    print(f"[run_hmr2] 时序形状：timestamps={pts.timestamps.shape}，"
          f"joint_rotations={pts.joint_rotations.shape if pts.joint_rotations is not None else None}")

    # 打印关节旋转量预览
    if pts.joint_rotations is not None:
        T, J, _ = pts.joint_rotations.shape
        print(f"\n关节旋转量 θ_j(t) 预览（前5帧，前6关节）：")
        from pose_extraction.core.data_types import SMPL_JOINT_NAMES
        for j in range(min(6, J)):
            vals = pts.joint_rotations[:min(5, T), j, :]
            print(f"  关节 {j:2d} {SMPL_JOINT_NAMES[j]:15s}: {vals.tolist()}")


if __name__ == "__main__":
    main()
