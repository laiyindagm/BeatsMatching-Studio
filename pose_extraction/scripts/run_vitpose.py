"""
pose_extraction/scripts/run_vitpose.py

ViTPose 命令行入口：对单个视频文件运行关节检测，输出 .pkl / .npz 文件。

用法：
    # 使用 conda 激活 pose_vitpose 环境后运行：
    conda activate pose_vitpose
    python -m pose_extraction.scripts.run_vitpose \\
        --video dance.mp4 \\
        --output output/dance_vitpose.pkl \\
        --device cuda \\
        --visualize

    # 指定自定义模型：
    python -m pose_extraction.scripts.run_vitpose \\
        --video dance.mp4 \\
        --vitpose_model usyd-community/vitpose-plus-large \\
        --skip_frames 1
"""

import argparse
import sys
from pathlib import Path

# 将项目根目录加入 Python 路径（方便直接运行脚本）
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pose_extraction.vitpose import ViTPoseVideoProcessor


def parse_args():
    p = argparse.ArgumentParser(description="ViTPose 视频关节检测")
    p.add_argument("--video", required=True, help="输入视频路径")
    p.add_argument("--output", default=None, help="输出文件路径（.pkl 或 .npz）")
    p.add_argument("--format", default="pkl", choices=["pkl", "npz"], help="输出格式")
    p.add_argument(
        "--vitpose_model",
        default="usyd-community/vitpose-base-simple",
        help="ViTPose 模型（HuggingFace 名称或本地路径）",
    )
    p.add_argument("--device", default="auto", help="计算设备：auto / cuda / cpu")
    p.add_argument("--yolo_model", default="yolov8n.pt", help="YOLO 权重路径")
    p.add_argument("--conf_threshold", type=float, default=0.35, help="YOLO 置信度阈值")
    p.add_argument("--skip_frames", type=int, default=0, help="跳帧数（0=逐帧）")
    p.add_argument("--start_frame", type=int, default=0, help="起始帧")
    p.add_argument("--end_frame", type=int, default=None, help="结束帧（默认末尾）")
    p.add_argument("--person_id", type=int, default=0, help="目标人物 ID")
    p.add_argument("--visualize", action="store_true", help="是否生成可视化视频")
    p.add_argument("--vis_output", default=None, help="可视化视频输出路径")
    return p.parse_args()


def main():
    args = parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"[错误] 视频文件不存在：{args.video}")
        sys.exit(1)

    # 默认输出路径
    if args.output is None:
        ext = ".npz" if args.format == "npz" else ".pkl"
        args.output = str(video_path.parent / (video_path.stem + f"_vitpose{ext}"))

    print(f"[run_vitpose] 视频：{args.video}")
    print(f"[run_vitpose] 输出：{args.output}")
    print(f"[run_vitpose] 设备：{args.device}，跳帧：{args.skip_frames}")

    processor = ViTPoseVideoProcessor(
        vitpose_model=args.vitpose_model,
        device=args.device,
        yolo_model=args.yolo_model,
        conf_threshold=args.conf_threshold,
        skip_frames=args.skip_frames,
    )

    sequence = processor.process_video(
        video_path=args.video,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        visualize=args.visualize,
        vis_output_path=args.vis_output,
    )

    processor.save(sequence, args.output, format=args.format)

    # 额外导出 PoseTimeSeries（对接 BeatsMatching 的输入）
    pts = processor.to_pose_time_series(sequence, person_id=args.person_id)
    pts_path = str(Path(args.output).with_suffix("")) + "_pts.pkl"
    import pickle
    with open(pts_path, "wb") as f:
        pickle.dump(pts, f)
    print(f"[run_vitpose] PoseTimeSeries 已保存：{pts_path}")
    print(f"[run_vitpose] 时序形状：timestamps={pts.timestamps.shape}，"
          f"joint_coords={pts.joint_coords.shape if pts.joint_coords is not None else None}")


if __name__ == "__main__":
    main()
