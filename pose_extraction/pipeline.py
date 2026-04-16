"""
pose_extraction/pipeline.py

统一的姿态提取管线顶层接口。

对外只需调用两个函数：
    extract_pose(video_path, mode, ...)        → PoseTimeSeries
    extract_and_detect(video_path, mode, ...)  → PoseTimeSeries（含节拍帧）

以及配置类：
    PoseExtractionConfig

设计原则：
- 按需导入：ViTPose 和 HMR2 的重量级依赖仅在实际使用时才加载，
  保证在未安装某一环境时另一管线仍可正常运行。
- 结果缓存：支持将中间结果（ViTPoseSequence / HMR2Sequence）保存到磁盘，
  避免重复处理耗时视频。
"""

from __future__ import annotations

import os
import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from .core.data_types import PoseTimeSeries
from .core.beat_detector import detect_beats


# ──────────────────────────────────────────────────────────────────────────────
# 配置类
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PoseExtractionConfig:
    """姿态提取管线的全局配置。

    Attributes:
        mode:                 提取模式：
                              "vitpose"  — 仅 2D 关节检测
                              "hmr2"     — 仅 3D 关节旋转量重建
                              "joint"    — 同时运行两条管线（联合模式）
        device:               计算设备，"auto" / "cuda" / "cpu"
        vitpose_model:        ViTPose 模型名称或本地路径
        hmr2_model:           HMR2 模型名称 "hmr2b" 或 "hmr2l"
        hmr2_checkpoint:      HMR2 本地 .ckpt 路径，None 则自动下载
        yolo_model:           YOLO 权重路径
        conf_threshold:       YOLO 人体检测置信度阈值
        skip_frames:          跳帧数（0 = 逐帧，1 = 隔一帧）
        max_persons:          每帧最多处理的人数
        person_id:            目标人物 ID（多人视频时选主角）
        start_frame:          视频起始帧
        end_frame:            视频结束帧（None = 末尾）
        cache_dir:            中间结果缓存目录（None = 不缓存）
        beat_strategy:        节拍检测策略（"auto"/"vitpose"/"hmr2"/"joint"）
        beat_smooth_K:        节拍检测平滑窗口半径
        vitpose_conf_threshold: ViTPose 关节置信度阈值
        save_vertices:        是否保存 SMPL 网格顶点
    """
    mode: str = "vitpose"               # "vitpose" | "hmr2" | "joint"
    device: str = "auto"
    vitpose_model: str = "usyd-community/vitpose-base-simple"
    hmr2_model: str = "hmr2b"
    hmr2_checkpoint: Optional[str] = None
    yolo_model: Optional[str] = "yolov8n.pt"
    conf_threshold: float = 0.35
    skip_frames: int = 0
    max_persons: int = 5
    person_id: int = 0
    start_frame: int = 0
    end_frame: Optional[int] = None
    cache_dir: Optional[str] = None
    beat_strategy: str = "auto"
    beat_smooth_K: int = 4
    vitpose_conf_threshold: float = 0.3
    save_vertices: bool = False


# ──────────────────────────────────────────────────────────────────────────────
# ViTPose 管线
# ──────────────────────────────────────────────────────────────────────────────

def _run_vitpose_pipeline(
    video_path: str,
    cfg: PoseExtractionConfig,
) -> PoseTimeSeries:
    """运行 ViTPose 管线，返回包含 2D 关节坐标的 PoseTimeSeries。"""
    cache_key = _cache_key(video_path, "vitpose", cfg)

    # 尝试从缓存加载
    if cfg.cache_dir is not None:
        cached = _try_load_cache(cache_key, cfg.cache_dir)
        if cached is not None:
            print(f"[pipeline] 从缓存加载 ViTPose 结果：{cache_key}")
            return cached

    # 按需导入
    from .vitpose import ViTPoseVideoProcessor

    processor = ViTPoseVideoProcessor(
        vitpose_model=cfg.vitpose_model,
        device=cfg.device,
        yolo_model=cfg.yolo_model,
        conf_threshold=cfg.conf_threshold,
        skip_frames=cfg.skip_frames,
        max_persons=cfg.max_persons,
    )

    sequence = processor.process_video(
        video_path=video_path,
        start_frame=cfg.start_frame,
        end_frame=cfg.end_frame,
    )

    pts = processor.to_pose_time_series(
        sequence,
        person_id=cfg.person_id,
        conf_threshold=cfg.vitpose_conf_threshold,
    )

    # 缓存
    if cfg.cache_dir is not None:
        _save_cache(pts, cache_key, cfg.cache_dir)

    return pts


# ──────────────────────────────────────────────────────────────────────────────
# HMR2 管线
# ──────────────────────────────────────────────────────────────────────────────

def _run_hmr2_pipeline(
    video_path: str,
    cfg: PoseExtractionConfig,
) -> PoseTimeSeries:
    """运行 HMR2 管线，返回包含 SMPL 旋转量的 PoseTimeSeries。"""
    cache_key = _cache_key(video_path, "hmr2", cfg)

    if cfg.cache_dir is not None:
        cached = _try_load_cache(cache_key, cfg.cache_dir)
        if cached is not None:
            print(f"[pipeline] 从缓存加载 HMR2 结果：{cache_key}")
            return cached

    from .hmr2 import HMR2VideoProcessor

    processor = HMR2VideoProcessor(
        model_name=cfg.hmr2_model,
        checkpoint_path=cfg.hmr2_checkpoint,
        device=cfg.device,
        yolo_model=cfg.yolo_model,
        conf_threshold=cfg.conf_threshold,
        skip_frames=cfg.skip_frames,
        max_persons=cfg.max_persons,
        save_vertices=cfg.save_vertices,
    )

    sequence = processor.process_video(
        video_path=video_path,
        start_frame=cfg.start_frame,
        end_frame=cfg.end_frame,
        person_id=cfg.person_id,
    )

    pts = processor.to_pose_time_series(
        sequence,
        person_id=cfg.person_id,
    )

    if cfg.cache_dir is not None:
        _save_cache(pts, cache_key, cfg.cache_dir)

    return pts


# ──────────────────────────────────────────────────────────────────────────────
# 公开接口
# ──────────────────────────────────────────────────────────────────────────────

def extract_pose(
    video_path: Union[str, Path],
    mode: str = "vitpose",
    config: Optional[PoseExtractionConfig] = None,
    **kwargs,
) -> PoseTimeSeries:
    """从视频中提取人体姿态时序数据。

    这是整个 pose_extraction 模块的**主要入口函数**。
    调用后返回的 PoseTimeSeries 可直接传入 BeatsMatching 算法模块。

    Args:
        video_path:  输入视频路径（.mp4、.avi、.mov 等均支持）
        mode:        提取模式：
                     "vitpose"  — 2D 关节坐标检测（COCO-17 关节）
                     "hmr2"     — 3D SMPL 关节旋转量重建（24 关节）
                     "joint"    — 同时运行两条管线（联合模式）
        config:      PoseExtractionConfig 配置对象，None 则使用默认配置
        **kwargs:    覆盖 config 中对应字段的关键字参数，例如：
                     device="cuda", skip_frames=2, person_id=0

    Returns:
        PoseTimeSeries 对象，包含：
        - timestamps:       帧时间戳数组 (T,)
        - joint_coords:     (T, 17, 2) ViTPose 坐标（mode="vitpose"/"joint" 时）
        - joint_rotations:  (T, 24, 3) HMR2 轴角旋转量（mode="hmr2"/"joint" 时）

    示例：
        # 只用 ViTPose
        pts = extract_pose("dance.mp4", mode="vitpose", device="cuda")

        # 只用 HMR2
        pts = extract_pose("dance.mp4", mode="hmr2", skip_frames=1)

        # 联合模式（最高精度）
        pts = extract_pose("dance.mp4", mode="joint")
    """
    video_path = str(video_path)

    if config is None:
        config = PoseExtractionConfig(mode=mode)
    else:
        config = PoseExtractionConfig(**{
            **config.__dict__,
            "mode": mode,
        })

    # 覆盖 kwargs
    for k, v in kwargs.items():
        if hasattr(config, k):
            setattr(config, k, v)
        else:
            warnings.warn(f"PoseExtractionConfig 中没有字段 '{k}'，已忽略。")

    print(f"[extract_pose] 视频：{video_path}，模式：{config.mode}")

    if config.mode == "vitpose":
        return _run_vitpose_pipeline(video_path, config)

    elif config.mode == "hmr2":
        return _run_hmr2_pipeline(video_path, config)

    elif config.mode == "joint":
        # 两条管线都跑
        pts_vit = _run_vitpose_pipeline(video_path, config)
        pts_hmr = _run_hmr2_pipeline(video_path, config)

        # 合并到同一个 PoseTimeSeries（以 ViTPose 的时间戳为基准）
        # 若两者帧数不一致（因 skip_frames 等原因），以较短的为准
        T_vit = len(pts_vit.timestamps)
        T_hmr = len(pts_hmr.timestamps)
        T = min(T_vit, T_hmr)

        combined = PoseTimeSeries(
            source="joint",
            video_path=video_path,
            fps=pts_vit.fps,
            timestamps=pts_vit.timestamps[:T],
            joint_coords=pts_vit.joint_coords[:T] if pts_vit.joint_coords is not None else None,
            joint_coords_conf=pts_vit.joint_coords_conf[:T] if pts_vit.joint_coords_conf is not None else None,
            joint_names=pts_vit.joint_names,
            joint_rotations=pts_hmr.joint_rotations[:T] if pts_hmr.joint_rotations is not None else None,
        )
        return combined

    else:
        raise ValueError(
            f"未知模式：'{config.mode}'，请选择 'vitpose'、'hmr2' 或 'joint'"
        )


def extract_and_detect(
    video_path: Union[str, Path],
    mode: str = "vitpose",
    beat_strategy: str = "auto",
    config: Optional[PoseExtractionConfig] = None,
    **kwargs,
) -> PoseTimeSeries:
    """从视频中提取姿态时序并检测节拍动作帧。

    在 extract_pose 的基础上，进一步调用节拍检测算法，
    将检测到的节拍帧索引写入 PoseTimeSeries.beat_frames。

    Args:
        video_path:    输入视频路径
        mode:          提取模式（同 extract_pose）
        beat_strategy: 节拍检测策略（"auto"/"vitpose"/"hmr2"/"joint"）
        config:        PoseExtractionConfig 配置对象
        **kwargs:      覆盖 config 字段的关键字参数

    Returns:
        PoseTimeSeries，其中 beat_frames / beat_timestamps 已被填充。

    示例：
        pts = extract_and_detect("dance.mp4", mode="joint", beat_strategy="joint")
        print(f"检测到 {len(pts.beat_frames)} 个节拍动作帧")
        print(f"节拍时间点（秒）：{pts.beat_timestamps}")
    """
    pts = extract_pose(video_path, mode=mode, config=config, **kwargs)
    pts = detect_beats(pts, strategy=beat_strategy)
    print(f"[extract_and_detect] 检测到节拍动作帧：{len(pts.beat_frames)} 个")
    return pts


# ──────────────────────────────────────────────────────────────────────────────
# 缓存工具
# ──────────────────────────────────────────────────────────────────────────────

def _cache_key(video_path: str, pipeline: str, cfg: PoseExtractionConfig) -> str:
    """根据视频路径和配置生成缓存文件名（不含目录）。"""
    import hashlib
    key_str = (
        f"{video_path}|{pipeline}|{cfg.vitpose_model}|{cfg.hmr2_model}|"
        f"{cfg.skip_frames}|{cfg.start_frame}|{cfg.end_frame}|{cfg.person_id}"
    )
    h = hashlib.md5(key_str.encode()).hexdigest()[:12]
    stem = Path(video_path).stem
    return f"{stem}_{pipeline}_{h}.pkl"


def _try_load_cache(cache_key: str, cache_dir: str) -> Optional[PoseTimeSeries]:
    """尝试从缓存目录加载 PoseTimeSeries，失败则返回 None。"""
    cache_path = os.path.join(cache_dir, cache_key)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            warnings.warn(f"缓存读取失败（{cache_path}）：{e}")
    return None


def _save_cache(pts: PoseTimeSeries, cache_key: str, cache_dir: str) -> None:
    """将 PoseTimeSeries 序列化到缓存目录。"""
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, cache_key)
    try:
        with open(cache_path, "wb") as f:
            pickle.dump(pts, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[pipeline] 结果已缓存：{cache_path}")
    except Exception as e:
        warnings.warn(f"缓存写入失败：{e}")
