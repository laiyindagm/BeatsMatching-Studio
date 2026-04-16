"""
pose_extraction/vitpose/detector.py

ViTPose 单帧 / 批次人体 2D 关节检测器。

技术栈：
  - 人体框检测：YOLOv8（ultralytics）或 Faster-RCNN / ViTDet（可扩展）
  - 关节估计：ViTPoseForPoseEstimation（HuggingFace transformers >= 4.38）

支持以下预训练权重（HuggingFace Hub）：
  - "usyd-community/vitpose-base-simple"    ← COCO-17 关节，推荐默认
  - "usyd-community/vitpose-plus-small"
  - "usyd-community/vitpose-plus-base"
  - "usyd-community/vitpose-plus-large"     ← 精度最高，显存要求大
  - 本地路径均可
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image

# ── HuggingFace transformers ──────────────────────────────────────────────────
try:
    from transformers import (
        AutoProcessor,
        VitPoseForPoseEstimation,  # transformers >= 4.51
    )
    _HF_AVAILABLE = True
except ImportError:
    try:
        from transformers import (
            AutoProcessor,
            ViTPoseForPoseEstimation as VitPoseForPoseEstimation,
        )
        _HF_AVAILABLE = True
    except ImportError:
        _HF_AVAILABLE = False
        warnings.warn(
            "transformers 中未找到 ViTPose 相关类，请升级：pip install -U transformers"
        )

# ── YOLOv8 人体检测 ────────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO as _YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False

from ..core.data_types import ViTPoseFrameResult


# COCO-17 关节名称
COCO17_JOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# COCO-17 骨架连接关系
COCO17_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),               # 头部
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),       # 上肢
    (5, 11), (6, 12), (11, 12),                     # 躯干
    (11, 13), (13, 15), (12, 14), (14, 16),         # 下肢
]


class ViTPoseDetector:
    """ViTPose 关节检测器。

    使用流程：
    1. 初始化检测器（加载 YOLO 人体检测 + ViTPose 关节估计模型）
    2. 调用 detect_frame(image) 处理单帧
    3. 或调用 detect_batch(images) 批量处理

    Args:
        vitpose_model:  ViTPose 模型名称或本地路径，
                        默认 "usyd-community/vitpose-base-simple"
        device:         "cuda" / "cpu" / "auto"，默认 "auto"
        yolo_model:     YOLO 权重，默认 "yolov8n.pt"（自动下载），
                        若设为 None 则使用全图作为人体框
        conf_threshold: YOLO 置信度阈值
        bbox_expand:    人体框扩展比例（0.1 表示扩展 10%，避免截边）
    """

    def __init__(
        self,
        vitpose_model: str = "usyd-community/vitpose-base-simple",
        device: str = "auto",
        yolo_model: Optional[str] = "yolov8n.pt",
        conf_threshold: float = 0.35,
        bbox_expand: float = 0.10,
    ) -> None:
        if not _HF_AVAILABLE:
            raise ImportError(
                "请先安装 transformers >= 4.38：pip install -U transformers"
            )

        # 设备
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.conf_threshold = conf_threshold
        self.bbox_expand = bbox_expand

        # 加载 YOLO 人体检测器
        self.yolo = None
        if yolo_model is not None:
            if _YOLO_AVAILABLE:
                self.yolo = _YOLO(yolo_model)
            else:
                warnings.warn(
                    "ultralytics 未安装，将使用全图作为人体框。"
                    "安装命令：pip install ultralytics"
                )

        # 加载 ViTPose 处理器与模型
        print(f"[ViTPoseDetector] 加载模型：{vitpose_model}（设备：{self.device}）")
        self.processor = AutoProcessor.from_pretrained(vitpose_model)
        self.model = VitPoseForPoseEstimation.from_pretrained(vitpose_model)
        self.model.to(self.device)
        self.model.eval()

        # 关节信息
        self.joint_names: List[str] = COCO17_JOINT_NAMES
        self.skeleton: List[Tuple[int, int]] = COCO17_SKELETON
        self.num_joints: int = len(COCO17_JOINT_NAMES)

        print(f"[ViTPoseDetector] 模型加载完成，关节数：{self.num_joints}")

    # ──────────────────────────────────────────────────────────────────────────
    # 人体框检测
    # ──────────────────────────────────────────────────────────────────────────

    def _detect_persons(
        self, image_rgb: np.ndarray
    ) -> List[Tuple[float, float, float, float]]:
        """返回画面中所有人体框 [x1, y1, x2, y2]（像素坐标）。

        若 YOLO 不可用，返回全图框。
        """
        H, W = image_rgb.shape[:2]

        if self.yolo is None:
            return [(0.0, 0.0, float(W), float(H))]

        results = self.yolo(
            image_rgb,
            classes=[0],          # class 0 = person
            conf=self.conf_threshold,
            verbose=False,
        )
        boxes = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                # 扩展框
                dx = (x2 - x1) * self.bbox_expand
                dy = (y2 - y1) * self.bbox_expand
                x1 = max(0.0, x1 - dx)
                y1 = max(0.0, y1 - dy)
                x2 = min(float(W), x2 + dx)
                y2 = min(float(H), y2 + dy)
                boxes.append((x1, y1, x2, y2))

        if not boxes:
            # 兜底：全图
            return [(0.0, 0.0, float(W), float(H))]
        return boxes

    # ──────────────────────────────────────────────────────────────────────────
    # ViTPose 关键点推理
    # ──────────────────────────────────────────────────────────────────────────

    def _run_vitpose(
        self,
        image_rgb: np.ndarray,
        boxes: List[Tuple[float, float, float, float]],
    ) -> List[np.ndarray]:
        """对给定图像和检测框列表运行 ViTPose 推理。

        Args:
            image_rgb: RGB 格式的 numpy 数组，形状 (H, W, 3)
            boxes:     [x1, y1, x2, y2] 列表（像素坐标）

        Returns:
            keypoints_list: 每个人的关键点数组，形状 (J, 3)=[x, y, conf]
        """
        pil_image = Image.fromarray(image_rgb)

        # transformers 的 ViTPose processor 需要 boxes 格式为
        # [[x1, y1, x2, y2], ...]，类型为 list[list[float]]
        boxes_list = [[b[0], b[1], b[2], b[3]] for b in boxes]

        inputs = self.processor(
            images=pil_image,
            boxes=[boxes_list],    # batch 维度
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # 解码关键点
        # outputs.heatmaps 形状：(batch, num_persons, J, H_hm, W_hm)
        # 使用 processor.post_process_pose_estimation 还原到原图坐标
        results = self.processor.post_process_pose_estimation(
            outputs,
            boxes=[boxes_list],
            threshold=0.0,         # 置信度过滤在后续统一处理
        )

        # results[0] 对应 batch 中第 0 张图，是一个 list（每个 person 一个 dict）
        keypoints_list = []
        for person_result in results[0]:
            # keypoints: Tensor (J, 2) 或 (J, 3)
            kpts = person_result["keypoints"].cpu().numpy()    # (J, 2)
            scores = person_result["scores"].cpu().numpy()      # (J,)
            # 合并为 (J, 3)
            kpts_with_conf = np.concatenate(
                [kpts, scores[:, np.newaxis]], axis=-1
            )  # (J, 3)
            keypoints_list.append(kpts_with_conf)

        return keypoints_list

    # ──────────────────────────────────────────────────────────────────────────
    # 公开接口
    # ──────────────────────────────────────────────────────────────────────────

    def detect_frame(
        self,
        image: Union[np.ndarray, Image.Image, str, Path],
        frame_idx: int = 0,
        timestamp: float = 0.0,
        predetected_boxes: Optional[List[Tuple[float, float, float, float]]] = None,
    ) -> ViTPoseFrameResult:
        """处理单帧图像，返回 ViTPoseFrameResult。

        Args:
            image:       输入图像，支持以下类型：
                         - np.ndarray (BGR 或 RGB)
                         - PIL.Image.Image
                         - str / Path（图像文件路径）
            frame_idx:   帧编号
            timestamp:   时间戳（秒）
            predetected_boxes: 预检测的人体框列表 [(x1,y1,x2,y2),...], 若提供则跳过 YOLO

        Returns:
            ViTPoseFrameResult
        """
        # 统一转为 RGB numpy
        image_rgb = self._to_rgb_numpy(image)

        # 人体框检测（可跳过）
        boxes = predetected_boxes if predetected_boxes is not None else self._detect_persons(image_rgb)

        # ViTPose 推理
        keypoints_list = self._run_vitpose(image_rgb, boxes)

        # 构造结果（person_id 按检测顺序 0, 1, 2, ...）
        persons: Dict[int, np.ndarray] = {}
        for pid, kpts in enumerate(keypoints_list):
            persons[pid] = kpts.astype(np.float32)

        return ViTPoseFrameResult(
            frame_idx=frame_idx,
            timestamp=timestamp,
            persons=persons,
            num_joints=self.num_joints,
        )

    def detect_batch(
        self,
        images: List[Union[np.ndarray, Image.Image]],
        frame_indices: Optional[List[int]] = None,
        timestamps: Optional[List[float]] = None,
        predetected_boxes_list: Optional[List[List[Tuple[float, float, float, float]]]] = None,
    ) -> List[ViTPoseFrameResult]:
        """批量处理多帧图像。

        支持两种模式：
          1. 逐帧循环（默认，任意数量人体框）
          2. 真正张量批处理（所有帧的所有 crop 堆叠为一个 batch）

        Args:
            images:        图像列表
            frame_indices: 帧编号列表，默认 [0, 1, 2, ...]
            timestamps:    时间戳列表，默认 [0.0, 1/fps, ...]
            predetected_boxes_list: 每帧对应的预检测人体框，可选

        Returns:
            ViTPoseFrameResult 列表
        """
        N = len(images)
        if frame_indices is None:
            frame_indices = list(range(N))
        if timestamps is None:
            timestamps = [float(i) for i in range(N)]

        # ── 步骤 1: 为每帧获取人体框 ──
        images_rgb = [self._to_rgb_numpy(img) for img in images]
        if predetected_boxes_list is not None:
            all_boxes = predetected_boxes_list
        else:
            all_boxes = [self._detect_persons(img) for img in images_rgb]

        # ── 步骤 2: 真正的张量批推理 ──
        try:
            return self._batch_detect_tensor(
                images_rgb, all_boxes, frame_indices, timestamps
            )
        except Exception as e:
            warnings.warn(f"[ViTPose] 批推理失败 ({e})，回退逐帧模式")
            # 回退：逐帧循环
            results = []
            for i in range(N):
                r = self.detect_frame(
                    images_rgb[i], frame_idx=frame_indices[i],
                    timestamp=timestamps[i], predetected_boxes=all_boxes[i],
                )
                results.append(r)
            return results

    def _batch_detect_tensor(
        self,
        images_rgb: List[np.ndarray],
        boxes_per_image: List[List[Tuple[float, float, float, float]]],
        frame_indices: List[int],
        timestamps: List[float],
    ) -> List[ViTPoseFrameResult]:
        """真正的张量级批推理：所有帧的 crop 堆叠为一个 batch。

        HuggingFace ViTPose processor 原生支持 multi-image batch：
            processor(images=[img1, img2], boxes=[boxes1, boxes2])

        这会将所有 crop 拼成一个 (total_crops, C, H, W) 张量，
        一次模型 forward 处理完毕。
        """
        N = len(images_rgb)
        pil_images = [Image.fromarray(img) for img in images_rgb]
        boxes_lists = [
            [[b[0], b[1], b[2], b[3]] for b in boxes]
            for boxes in boxes_per_image
        ]

        inputs = self.processor(
            images=pil_images,
            boxes=boxes_lists,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # 解码关键点，results[i] 对应第 i 帧的 person list
        raw_results = self.processor.post_process_pose_estimation(
            outputs, boxes=boxes_lists, threshold=0.0,
        )

        # 组装 ViTPoseFrameResult
        results = []
        for i in range(N):
            persons: Dict[int, np.ndarray] = {}
            if i < len(raw_results):
                for pid, person_result in enumerate(raw_results[i]):
                    kpts = person_result["keypoints"].cpu().numpy()
                    scores = person_result["scores"].cpu().numpy()
                    kpts_with_conf = np.concatenate(
                        [kpts, scores[:, np.newaxis]], axis=-1
                    )
                    persons[pid] = kpts_with_conf.astype(np.float32)
            results.append(ViTPoseFrameResult(
                frame_idx=frame_indices[i],
                timestamp=timestamps[i],
                persons=persons,
                num_joints=self.num_joints,
            ))
        return results

    # ──────────────────────────────────────────────────────────────────────────
    # 工具方法
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _to_rgb_numpy(image: Union[np.ndarray, Image.Image, str, Path]) -> np.ndarray:
        """将各种格式的图像转为 RGB numpy 数组。"""
        if isinstance(image, (str, Path)):
            image = Image.open(str(image)).convert("RGB")
        if isinstance(image, Image.Image):
            return np.array(image.convert("RGB"))
        # numpy: 判断是否为 BGR（OpenCV 默认）
        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[2] == 3:
                # 默认假设为 BGR，转 RGB
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
        raise TypeError(f"不支持的图像类型：{type(image)}")

    def visualize_frame(
        self,
        image: Union[np.ndarray, Image.Image],
        result: ViTPoseFrameResult,
        conf_threshold: float = 0.3,
        radius: int = 4,
        thickness: int = 2,
    ) -> np.ndarray:
        """将关节点和骨架绘制到图像上，返回 BGR numpy 数组（用于 cv2.imshow）。

        Args:
            image:          原始图像
            result:         detect_frame 的返回结果
            conf_threshold: 置信度阈值
            radius:         关节点半径
            thickness:      骨架线条粗细

        Returns:
            vis_bgr: BGR 格式的可视化图像
        """
        image_rgb = self._to_rgb_numpy(image)
        vis = image_rgb.copy()

        # 颜色表（按关节分组着色）
        colors = [
            (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
            (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
            (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
            (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255), (255, 0, 170),
        ]

        for pid, kpts in result.persons.items():
            # 画骨架
            for (src, dst) in self.skeleton:
                if src < len(kpts) and dst < len(kpts):
                    s = kpts[src]
                    d = kpts[dst]
                    if s[2] >= conf_threshold and d[2] >= conf_threshold:
                        pt1 = (int(s[0]), int(s[1]))
                        pt2 = (int(d[0]), int(d[1]))
                        cv2.line(vis, pt1, pt2, (0, 255, 255), thickness)

            # 画关节点
            for j, kpt in enumerate(kpts):
                if kpt[2] >= conf_threshold:
                    pt = (int(kpt[0]), int(kpt[1]))
                    color = colors[j % len(colors)]
                    cv2.circle(vis, pt, radius, color, -1)

        return cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
