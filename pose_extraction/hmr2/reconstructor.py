"""
pose_extraction/hmr2/reconstructor.py

HMR2 / 4D-Humans 单帧三维人体重建器（Windows GPU 版）。

Windows 兼容性修复：
  1. pyrender/EGL 不支持 Windows → 禁用渲染器导入（已 patch site-packages）
  2. $HOME 环境变量在 Windows 不存在 → 使用 $USERPROFILE（已 patch）
  3. chumpy 不兼容 Python 3.10+ → 不直接依赖

安装：
    conda activate pose_unified
    # 4D-Humans（已通过 --no-deps 安装）
    pip install pytorch-lightning yacs scikit-image webdataset pandas dill gdown
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image

# ── 4D-Humans / HMR2 ─────────────────────────────────────────────────────────
try:
    from hmr2.models import HMR2 as _HMR2Model, load_hmr2, DEFAULT_CHECKPOINT
    from hmr2.datasets.utils import generate_image_patch_cv2, convert_cvimg_to_tensor
    _HMR2_AVAILABLE = True
except ImportError as _hmr2_err:
    _HMR2_AVAILABLE = False
    _hmr2_err_msg = str(_hmr2_err)
    warnings.warn(
        f"4D-Humans (hmr2) 导入失败: {_hmr2_err_msg}\n"
        "请运行: conda activate pose_unified && pip install git+https://github.com/shubham-goel/4D-Humans.git --no-deps"
    )

# ── YOLOv8 人体框检测 ─────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO as _YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False

# ── 旋转工具 ──────────────────────────────────────────────────────────────────
try:
    from scipy.spatial.transform import Rotation as _Rotation
    _SCIPY_ROT_AVAILABLE = True
except ImportError:
    _SCIPY_ROT_AVAILABLE = False

from ..core.data_types import HMR2FrameResult, SMPL_JOINT_NAMES


# HMR2 图像归一化参数（ImageNet 标准）
_IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMG_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class HMR2Reconstructor:
    """HMR2 单帧三维人体重建器。

    使用流程：
    1. 初始化（可选：调用 download_weights() 下载权重）
    2. 调用 reconstruct_frame(image) 处理单帧
    3. 输出包含完整 SMPL 参数的 HMR2FrameResult 列表

    Args:
        checkpoint_path:  本地 .ckpt 文件路径，None → 使用默认路径（需提前下载）
        device:           "cuda" / "cpu" / "auto"
        yolo_model:       YOLO 权重路径，None 则使用全图
        conf_threshold:   YOLO 检测置信度阈值
        bbox_expand:      人体框扩展比例（防止截边）
        image_size:       HMR2 输入分辨率（默认 256）
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = "auto",
        yolo_model: Optional[str] = "yolov8n.pt",
        conf_threshold: float = 0.35,
        bbox_expand: float = 0.15,
        image_size: int = 256,
    ) -> None:
        if not _HMR2_AVAILABLE:
            raise ImportError(
                "4D-Humans 未安装或存在兼容性问题。\n"
                "安装命令（pose_unified 环境）：\n"
                "  pip install git+https://github.com/shubham-goel/4D-Humans.git --no-deps\n"
                "  pip install pytorch-lightning yacs scikit-image webdataset pandas dill gdown"
            )

        # 设备选择
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.conf_threshold = conf_threshold
        self.bbox_expand = bbox_expand
        self.image_size = image_size

        # YOLO 人体检测
        self.yolo: Optional[Any] = None
        if yolo_model is not None:
            if _YOLO_AVAILABLE:
                self.yolo = _YOLO(yolo_model)
            else:
                warnings.warn("ultralytics 未安装，将使用全图框。")

        # HMR2 模型加载
        ckpt = checkpoint_path or DEFAULT_CHECKPOINT
        print(f"[HMR2Reconstructor] 加载模型（设备：{self.device}）")
        print(f"  权重：{ckpt}")
        self.model, self.model_cfg = load_hmr2(ckpt)
        self.model.to(self.device)
        self.model.eval()

        # 从模型配置读取实际图像尺寸
        if hasattr(self.model_cfg.MODEL, "IMAGE_SIZE"):
            self.image_size = self.model_cfg.MODEL.IMAGE_SIZE
        print(f"[HMR2Reconstructor] 模型加载完成，IMAGE_SIZE={self.image_size}")

    # ──────────────────────────────────────────────────────────────────────────
    # 人体框检测
    # ──────────────────────────────────────────────────────────────────────────

    def _detect_persons(
        self, image_rgb: np.ndarray
    ) -> List[Tuple[float, float, float, float]]:
        """返回 [x1, y1, x2, y2] 列表（像素坐标）。"""
        H, W = image_rgb.shape[:2]
        if self.yolo is None:
            return [(0.0, 0.0, float(W), float(H))]

        results = self.yolo(
            image_rgb, classes=[0], conf=self.conf_threshold, verbose=False
        )
        boxes = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                dx = (x2 - x1) * self.bbox_expand
                dy = (y2 - y1) * self.bbox_expand
                x1 = max(0.0, x1 - dx)
                y1 = max(0.0, y1 - dy)
                x2 = min(float(W), x2 + dx)
                y2 = min(float(H), y2 + dy)
                boxes.append((x1, y1, x2, y2))

        return boxes if boxes else [(0.0, 0.0, float(W), float(H))]

    # ──────────────────────────────────────────────────────────────────────────
    # 图像预处理（裁剪 + 归一化）
    # ──────────────────────────────────────────────────────────────────────────

    def _preprocess_crop(
        self,
        image_rgb: np.ndarray,
        box: Tuple[float, float, float, float],
    ) -> Tuple[torch.Tensor, float, float, float]:
        """将人体框裁剪并归一化为 HMR2 输入格式。

        Args:
            image_rgb: RGB 格式 numpy (H, W, 3), uint8
            box: (x1, y1, x2, y2)

        Returns:
            img_tensor: (1, 3, H, W) float32 on self.device
            cx, cy: 框中心（原图坐标）
            bbox_size: 框的最大边长
        """
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        bbox_w = x2 - x1
        bbox_h = y2 - y1
        bbox_size = float(max(bbox_w, bbox_h))

        # 使用 4D-Humans 的 generate_image_patch_cv2 裁剪
        # 注意：该函数期望 BGR 输入（OpenCV 格式）
        img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        img_patch, _ = generate_image_patch_cv2(
            img_bgr,
            cx, cy,
            bbox_size, bbox_size,
            self.image_size, self.image_size,
            do_flip=False,
            scale=1.0,
            rot=0,
        )
        # img_patch 是 BGR uint8 (H, W, 3)
        img_patch_rgb = cv2.cvtColor(img_patch, cv2.COLOR_BGR2RGB)
        img_float = img_patch_rgb.astype(np.float32) / 255.0

        # ImageNet 归一化
        img_float = (img_float - _IMG_MEAN) / _IMG_STD
        img_chw = np.transpose(img_float, (2, 0, 1))           # (3, H, W)
        img_tensor = torch.from_numpy(img_chw).unsqueeze(0).to(self.device)  # (1, 3, H, W)
        return img_tensor, cx, cy, bbox_size

    # ──────────────────────────────────────────────────────────────────────────
    # HMR2 推理
    # ──────────────────────────────────────────────────────────────────────────

    def _run_hmr2(
        self,
        image_rgb: np.ndarray,
        boxes: List[Tuple[float, float, float, float]],
    ) -> List[Dict]:
        """对给定图像和检测框列表运行 HMR2 推理。

        Returns:
            list of dict，每个人包含：
              global_orient: (1, 3) numpy，轴角
              body_pose:     (23, 3) numpy，轴角
              betas:         (10,) numpy
              pred_cam:      (3,) numpy  [s, tx, ty]
              pred_cam_t_full: (3,) numpy  全图相机平移
              joints_3d:     (44, 3) numpy（可选）
              joints_2d:     (44, 2) numpy（可选）
        """
        H_img, W_img = image_rgb.shape[:2]
        results = []

        for box in boxes:
            try:
                img_tensor, cx, cy, bbox_size = self._preprocess_crop(image_rgb, box)
            except Exception as e:
                warnings.warn(f"[HMR2] 裁剪失败：{e}")
                continue

            # 构造 batch（HMR2 forward 接受 img + bbox_info）
            batch = {
                "img": img_tensor,
                "bbox_info": torch.tensor([[cx, cy, bbox_size]], dtype=torch.float32).to(self.device),
            }

            with torch.no_grad():
                out = self.model(batch)

            # ── 提取 SMPL 参数 ───────────────────────────────────────────────
            pred_smpl = out.get("pred_smpl_params", {})
            global_orient_mat = pred_smpl.get("global_orient", None)  # (1, 1, 3, 3) rotmat
            body_pose_mat     = pred_smpl.get("body_pose",     None)  # (1, 23, 3, 3)
            betas             = pred_smpl.get("betas",         None)  # (1, 10)
            pred_cam          = out.get("pred_cam",            None)  # (1, 3)
            joints_3d         = out.get("pred_keypoints_3d",   None)  # (1, 44, 3)
            joints_2d         = out.get("pred_keypoints_2d",   None)  # (1, 44, 2)

            # ── 旋转矩阵 → 轴角 ─────────────────────────────────────────────
            def rotmat_to_aa(rotmat_t: Optional[torch.Tensor]) -> Optional[np.ndarray]:
                if rotmat_t is None:
                    return None
                # squeeze batch dim：(J, 3, 3)
                rm = rotmat_t.squeeze(0).cpu().numpy()
                if rm.ndim == 2:
                    rm = rm[np.newaxis]  # (1, 3, 3)
                if _SCIPY_ROT_AVAILABLE:
                    return _Rotation.from_matrix(rm).as_rotvec()  # (J, 3)
                else:
                    # fallback: cv2.Rodrigues
                    N = rm.shape[0]
                    aa = np.zeros((N, 3), dtype=np.float32)
                    for i in range(N):
                        rvec, _ = cv2.Rodrigues(rm[i])
                        aa[i] = rvec.ravel()
                    return aa

            go_aa = rotmat_to_aa(global_orient_mat)   # (1, 3)
            bp_aa = rotmat_to_aa(body_pose_mat)       # (23, 3)

            # ── 计算全图相机平移 ─────────────────────────────────────────────
            pred_cam_np = pred_cam.squeeze(0).cpu().numpy() if pred_cam is not None else None
            if pred_cam_np is not None:
                focal = 5000.0
                s, tx, ty = pred_cam_np
                cx_full = W_img / 2.0
                cy_full = H_img / 2.0
                tz = focal / (0.5 * bbox_size * s + 1e-6)
                pred_cam_t_full = np.array([
                    tx * bbox_size / (2.0 * focal) * tz + (cx - cx_full) / focal * tz,
                    ty * bbox_size / (2.0 * focal) * tz + (cy - cy_full) / focal * tz,
                    tz,
                ], dtype=np.float32)
            else:
                pred_cam_t_full = None

            results.append({
                "global_orient":    go_aa,
                "body_pose":        bp_aa,
                "betas":            betas.squeeze(0).cpu().numpy() if betas is not None else None,
                "pred_cam":         pred_cam_np,
                "pred_cam_t_full":  pred_cam_t_full,
                "joints_3d":        joints_3d.squeeze(0).cpu().numpy() if joints_3d is not None else None,
                "joints_2d":        joints_2d.squeeze(0).cpu().numpy() if joints_2d is not None else None,
            })

        return results

    # ──────────────────────────────────────────────────────────────────────────
    # 公开接口
    # ──────────────────────────────────────────────────────────────────────────

    def reconstruct_frame(
        self,
        image: Union[np.ndarray, Image.Image, str, Path],
        frame_idx: int = 0,
        timestamp: float = 0.0,
        predetected_boxes: Optional[List[Tuple[float, float, float, float]]] = None,
    ) -> List[HMR2FrameResult]:
        """处理单帧，返回所有检测人物的 HMR2FrameResult 列表。

        Args:
            predetected_boxes: 预检测的人体框，若提供则跳过 YOLO
        """
        image_rgb = self._to_rgb_numpy(image)
        boxes = predetected_boxes if predetected_boxes is not None else self._detect_persons(image_rgb)
        person_dicts = self._run_hmr2(image_rgb, boxes)

        results = []
        for pid, pd in enumerate(person_dicts):
            r = HMR2FrameResult(
                frame_idx=frame_idx,
                timestamp=timestamp,
                person_id=pid,
                global_orient=pd["global_orient"],
                body_pose=pd["body_pose"],
                betas=pd["betas"],
                pred_cam=pd["pred_cam"],
                pred_cam_t_full=pd["pred_cam_t_full"],
                joints_3d=pd["joints_3d"],
                joints_2d=pd["joints_2d"],
                vertices=None,
            )
            results.append(r)
        return results

    # ──────────────────────────────────────────────────────────────────────────
    # 批量推理（多 crop 单次 forward）
    # ──────────────────────────────────────────────────────────────────────────

    def batch_reconstruct(
        self,
        frames_data: List[Tuple[np.ndarray, List[Tuple[float, float, float, float]], int, float]],
    ) -> List[List[HMR2FrameResult]]:
        """批量处理多帧多人，所有 crop 合并为单次 model forward。

        Args:
            frames_data: [(image_rgb, boxes, frame_idx, timestamp), ...]
                         boxes 可为 None（会调用 _detect_persons）

        Returns:
            List[List[HMR2FrameResult]]，外层对应帧，内层对应人
        """
        if not frames_data:
            return []

        # ── 步骤 1: 为每帧获取人体框，并预处理所有 crop ──
        all_crops = []       # (img_tensor, cx, cy, bbox_size)
        frame_crop_counts = []  # 每帧有几个人
        frame_meta = []      # (image_rgb, frame_idx, timestamp, boxes)

        for image_rgb, boxes, frame_idx, timestamp in frames_data:
            if boxes is None or len(boxes) == 0:
                boxes = self._detect_persons(image_rgb)
            H_img, W_img = image_rgb.shape[:2]

            crop_count = 0
            for box in boxes:
                try:
                    img_tensor, cx, cy, bbox_size = self._preprocess_crop(image_rgb, box)
                    all_crops.append((img_tensor, cx, cy, bbox_size))
                    crop_count += 1
                except Exception:
                    pass
            frame_crop_counts.append(crop_count)
            frame_meta.append((image_rgb.shape[:2], frame_idx, timestamp, boxes))

        if not all_crops:
            return [[] for _ in frames_data]

        # ── 步骤 2: 合并为单个 batch tensor ──
        img_batch = torch.cat([c[0] for c in all_crops], dim=0)       # (N_total, 3, H, W)
        bbox_batch = torch.stack([
            torch.tensor([c[1], c[2], c[3]], dtype=torch.float32)
            for c in all_crops
        ]).to(self.device)                                             # (N_total, 3)

        batch = {"img": img_batch, "bbox_info": bbox_batch}

        with torch.no_grad():
            out = self.model(batch)

        # ── 步骤 3: 解析输出，向量化 rotmat → axis-angle ──
        pred_smpl = out.get("pred_smpl_params", {})
        go_mat_all = pred_smpl.get("global_orient", None)  # (N_total, 1, 3, 3)
        bp_mat_all = pred_smpl.get("body_pose", None)      # (N_total, 23, 3, 3)
        betas_all = pred_smpl.get("betas", None)            # (N_total, 10)
        pred_cam_all = out.get("pred_cam", None)             # (N_total, 3)
        j3d_all = out.get("pred_keypoints_3d", None)
        j2d_all = out.get("pred_keypoints_2d", None)

        N_total = img_batch.shape[0]

        # 向量化 rotmat → axis-angle（所有 crop × 24 关节一次性转换）
        go_aa_all, bp_aa_all = self._batch_rotmat_to_aa(go_mat_all, bp_mat_all, N_total)

        # ── 步骤 4: 分发到各帧各人 ──
        all_results = []
        crop_idx = 0
        for fi, (img_shape, frame_idx, timestamp, boxes) in enumerate(frame_meta):
            H_img, W_img = img_shape
            n_crops = frame_crop_counts[fi]
            frame_results = []
            for pi in range(n_crops):
                ci = crop_idx + pi
                _, cx, cy, bbox_size = all_crops[ci]

                go_aa = go_aa_all[ci] if go_aa_all is not None else None
                bp_aa = bp_aa_all[ci] if bp_aa_all is not None else None
                betas_np = betas_all[ci].cpu().numpy() if betas_all is not None else None
                pred_cam_np = pred_cam_all[ci].cpu().numpy() if pred_cam_all is not None else None
                j3d_np = j3d_all[ci].cpu().numpy() if j3d_all is not None else None
                j2d_np = j2d_all[ci].cpu().numpy() if j2d_all is not None else None

                # 计算全图相机平移
                pred_cam_t_full = None
                if pred_cam_np is not None:
                    focal = 5000.0
                    s, tx, ty = pred_cam_np
                    cx_full, cy_full = W_img / 2.0, H_img / 2.0
                    tz = focal / (0.5 * bbox_size * s + 1e-6)
                    pred_cam_t_full = np.array([
                        tx * bbox_size / (2.0 * focal) * tz + (cx - cx_full) / focal * tz,
                        ty * bbox_size / (2.0 * focal) * tz + (cy - cy_full) / focal * tz,
                        tz,
                    ], dtype=np.float32)

                frame_results.append(HMR2FrameResult(
                    frame_idx=frame_idx, timestamp=timestamp, person_id=pi,
                    global_orient=go_aa, body_pose=bp_aa, betas=betas_np,
                    pred_cam=pred_cam_np, pred_cam_t_full=pred_cam_t_full,
                    joints_3d=j3d_np, joints_2d=j2d_np, vertices=None,
                ))
            crop_idx += n_crops
            all_results.append(frame_results)
        return all_results

    @staticmethod
    def _batch_rotmat_to_aa(
        go_mat: Optional[torch.Tensor],
        bp_mat: Optional[torch.Tensor],
        N: int,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """向量化 rotmat → axis-angle：所有 crop 的所有关节一次性转换。

        Args:
            go_mat: (N, 1, 3, 3)  global orient rotation matrices
            bp_mat: (N, 23, 3, 3) body pose rotation matrices
            N: batch size

        Returns:
            go_aa: (N, 1, 3) or None
            bp_aa: (N, 23, 3) or None
        """
        go_aa_all = None
        bp_aa_all = None

        if _SCIPY_ROT_AVAILABLE:
            if go_mat is not None:
                rm = go_mat.cpu().numpy().reshape(-1, 3, 3)    # (N, 3, 3)
                aa = _Rotation.from_matrix(rm).as_rotvec()     # (N, 3) 一次调用
                go_aa_all = aa.reshape(N, 1, 3)

            if bp_mat is not None:
                rm = bp_mat.cpu().numpy().reshape(-1, 3, 3)    # (N*23, 3, 3)
                aa = _Rotation.from_matrix(rm).as_rotvec()     # (N*23, 3) 一次调用
                bp_aa_all = aa.reshape(N, 23, 3)
        else:
            # fallback: cv2.Rodrigues (较慢但通用)
            if go_mat is not None:
                rm = go_mat.cpu().numpy().reshape(-1, 3, 3)
                go_aa_all = np.zeros((N, 1, 3), dtype=np.float32)
                for i in range(N):
                    rvec, _ = cv2.Rodrigues(rm[i])
                    go_aa_all[i, 0] = rvec.ravel()

            if bp_mat is not None:
                rm = bp_mat.cpu().numpy().reshape(-1, 3, 3)
                bp_aa_all = np.zeros((N, 23, 3), dtype=np.float32)
                for i in range(N * 23):
                    rvec, _ = cv2.Rodrigues(rm[i])
                    bp_aa_all[i // 23, i % 23] = rvec.ravel()

        return go_aa_all, bp_aa_all

    # ──────────────────────────────────────────────────────────────────────────
    # 静态工具
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def download_weights() -> None:
        """下载 HMR2 权重到默认缓存目录（~/.cache/4DHumans）。"""
        from hmr2.models import download_models
        print("[HMR2] 检查/下载权重...")
        download_models()
        print("[HMR2] 权重就绪。")

    @staticmethod
    def _to_rgb_numpy(image: Union[np.ndarray, Image.Image, str, Path]) -> np.ndarray:
        if isinstance(image, (str, Path)):
            image = Image.open(str(image)).convert("RGB")
        if isinstance(image, Image.Image):
            return np.array(image.convert("RGB"), dtype=np.uint8)
        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[2] == 3:
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
        raise TypeError(f"不支持的图像类型：{type(image)}")
