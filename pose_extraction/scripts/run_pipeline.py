"""
pose_extraction/scripts/run_pipeline.py

统一管线命令行入口：一键运行完整的姿态提取 + 节拍检测流程。

用法：
    conda activate pose_vitpose   # 或 pose_hmr2
    python -m pose_extraction.scripts.run_pipeline \\
        --video dance.mp4 \\
        --mode joint \\
        --output output/dance_result.pkl \\
        --detect_beats \\
        --device cuda

    # 快速运行（ViTPose 模式，跳帧2）：
    python -m pose_extraction.scripts.run_pipeline \\
        --video dance.mp4 \\
        --mode vitpose \\
        --skip_frames 2 \\
        --detect_beats

输出文件：
    *.pkl         — PoseTimeSeries 对象（含节拍帧，可直接传入 BeatsMatching）
    *_beats.json  — 节拍时间点 JSON（方便其他程序读取）
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pose_extraction.pipeline import (
    extract_pose,
    extract_and_detect,
    PoseExtractionConfig,
)


def parse_args():
    p = argparse.ArgumentParser(description="姿态提取 + 节拍检测统一管线")
    p.add_argument("--video", required=True, help="输入视频路径")
    p.add_argument("--mode", default="vitpose",
                   choices=["vitpose", "hmr2", "joint"], help="提取模式")
    p.add_argument("--output", default=None, help="输出 .pkl 路径")
    p.add_argument("--device", default="auto", help="计算设备")
    p.add_argument("--skip_frames", type=int, default=0, help="跳帧数")
    p.add_argument("--person_id", type=int, default=0, help="目标人物 ID")
    p.add_argument("--start_frame", type=int, default=0)
    p.add_argument("--end_frame", type=int, default=None)
    p.add_argument("--cache_dir", default=None, help="中间结果缓存目录")
    p.add_argument("--detect_beats", action="store_true", help="是否运行节拍检测")
    p.add_argument("--beat_strategy", default="auto",
                   choices=["auto", "vitpose", "hmr2", "joint"])
    p.add_argument("--vitpose_model", default="usyd-community/vitpose-base-simple")
    p.add_argument("--hmr2_model", default="hmr2b", choices=["hmr2b", "hmr2l"])
    p.add_argument("--hmr2_checkpoint", default=None)
    return p.parse_args()


def main():
    args = parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"[错误] 视频文件不存在：{args.video}")
        sys.exit(1)

    if args.output is None:
        tag = f"_{args.mode}"
        if args.detect_beats:
            tag += "_beats"
        args.output = str(video_path.parent / (video_path.stem + tag + ".pkl"))

    cfg = PoseExtractionConfig(
        mode=args.mode,
        device=args.device,
        vitpose_model=args.vitpose_model,
        hmr2_model=args.hmr2_model,
        hmr2_checkpoint=args.hmr2_checkpoint,
        skip_frames=args.skip_frames,
        person_id=args.person_id,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        cache_dir=args.cache_dir,
        beat_strategy=args.beat_strategy,
    )

    print(f"\n{'='*60}")
    print(f" 视频：{args.video}")
    print(f" 模式：{args.mode}，节拍检测：{args.detect_beats}")
    print(f"{'='*60}\n")

    if args.detect_beats:
        pts = extract_and_detect(
            video_path=args.video,
            mode=args.mode,
            beat_strategy=args.beat_strategy,
            config=cfg,
        )
    else:
        pts = extract_pose(
            video_path=args.video,
            mode=args.mode,
            config=cfg,
        )

    # 保存 PoseTimeSeries
    with open(args.output, "wb") as f:
        pickle.dump(pts, f)
    print(f"\n[run_pipeline] PoseTimeSeries 已保存：{args.output}")

    # 打印摘要
    print(f"\n{'='*60}")
    print(f" 结果摘要")
    print(f"{'='*60}")
    print(f" 视频：{args.video}")
    print(f" 总帧数：{len(pts.timestamps)}")
    print(f" 帧率：{pts.fps:.2f} fps")
    print(f" 时长：{pts.timestamps[-1]:.2f}s（若有帧）")
    if pts.joint_coords is not None:
        print(f" ViTPose 关节坐标：{pts.joint_coords.shape}（T, J, 2）")
    if pts.joint_rotations is not None:
        print(f" HMR2 关节旋转量：{pts.joint_rotations.shape}（T, J=24, 3）")
    if pts.beat_frames is not None:
        print(f" 节拍动作帧：{len(pts.beat_frames)} 个")
        # 保存 beats JSON
        beats_json = {
            "video": args.video,
            "mode": args.mode,
            "beat_strategy": args.beat_strategy,
            "fps": pts.fps,
            "beat_frames": pts.beat_frames,
            "beat_timestamps": pts.beat_timestamps,
        }
        beats_path = str(Path(args.output).with_suffix("")) + "_beats.json"
        with open(beats_path, "w", encoding="utf-8") as f:
            json.dump(beats_json, f, ensure_ascii=False, indent=2)
        print(f" 节拍 JSON 已保存：{beats_path}")
        print(f" 节拍时间点（秒）：{pts.beat_timestamps[:10]}{'...' if len(pts.beat_timestamps)>10 else ''}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
