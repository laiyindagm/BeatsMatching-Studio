"""
快速验证 ViTPose GPU 推理和 HMR2 环境完整性。
运行方式：
    D:\\anaconda3\\envs\\pose_unified\\python.exe pose_extraction/scripts/verify_env.py
"""
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import sys
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import cv2
import numpy as np
import torch

print("="*60)
print("BeatsMatching - pose_unified env verification")
print("="*60)
print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU Memory: {mem:.1f} GB")
print(f"NumPy: {np.__version__}")
print()

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. ViTPose load
print("1. Testing ViTPose load...")
model_vit = None
proc = None
try:
    from transformers import AutoProcessor, VitPoseForPoseEstimation
    t0 = time.time()
    proc = AutoProcessor.from_pretrained("usyd-community/vitpose-base-simple")
    model_vit = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple")
    model_vit = model_vit.to(device)
    model_vit.eval()
    print(f"   [OK] ViTPose loaded in {time.time()-t0:.1f}s, device: {next(model_vit.parameters()).device}")
except Exception as e:
    print(f"   [FAIL] ViTPose load error: {e}")

print()

# 2. YOLO load
print("2. Testing YOLO load...")
yolo = None
try:
    from ultralytics import YOLO
    yolo = YOLO("yolov8n.pt")
    print(f"   [OK] YOLO loaded")
except Exception as e:
    print(f"   [FAIL] YOLO error: {e}")

print()

# 3. ViTPose GPU single-frame inference
print("3. Testing ViTPose GPU inference on dummy frame...")
if model_vit is not None and proc is not None:
    try:
        from PIL import Image
        dummy_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        pil_img = Image.fromarray(dummy_frame)
        boxes_list = [[0.0, 0.0, 1280.0, 720.0]]

        inputs = proc(images=pil_img, boxes=[boxes_list], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        t1 = time.time()
        with torch.no_grad():
            outputs = model_vit(**inputs)
        results = proc.post_process_pose_estimation(outputs, boxes=[boxes_list], threshold=0.0)
        dt = (time.time() - t1) * 1000

        if results and results[0]:
            kpts = results[0][0]["keypoints"].cpu().numpy()
            scores = results[0][0]["scores"].cpu().numpy()
            print(f"   [OK] ViTPose GPU inference done! Time: {dt:.1f}ms, joints: {kpts.shape[0]}")
            print(f"      nose: x={kpts[0,0]:.1f}, y={kpts[0,1]:.1f}, conf={scores[0]:.3f}")
        else:
            print(f"   [OK] Inference done (no person in random frame), Time: {dt:.1f}ms")
    except Exception as e:
        import traceback
        print(f"   [FAIL] Inference error: {e}")
        traceback.print_exc()
else:
    print("   [SKIP] ViTPose not loaded")

print()

# 4. HMR2 import
print("4. Testing HMR2 import...")
try:
    from hmr2.models import HMR2, DEFAULT_CHECKPOINT, load_hmr2
    from hmr2.datasets.utils import generate_image_patch_cv2, convert_cvimg_to_tensor
    print(f"   [OK] HMR2 import OK")
    print(f"   Default checkpoint: {DEFAULT_CHECKPOINT}")
    ckpt_exists = os.path.exists(DEFAULT_CHECKPOINT)
    print(f"   Checkpoint exists: {ckpt_exists}")
    if not ckpt_exists:
        print("   To download: python pose_extraction/scripts/download_weights.py")
except Exception as e:
    print(f"   [FAIL] HMR2 import error: {e}")

print()

# 5. Real video frame test
print("5. Testing real video frame inference...")
video_candidates = [
    r"d:\毕业设计\BeatsMatching\output.mp4",
    r"d:\毕业设计\BeatsMatching\output1.mp4",
]
video_path = None
for vp in video_candidates:
    if os.path.exists(vp):
        video_path = vp
        break

if video_path is None or model_vit is None:
    print("   [SKIP] No video or model")
else:
    from PIL import Image
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if ret:
        print(f"   Video: {os.path.basename(video_path)}, size: {frame.shape[1]}x{frame.shape[0]}")
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(frame_rgb)
        H, W = frame.shape[:2]

        boxes_list = [[0.0, 0.0, float(W), float(H)]]
        if yolo is not None:
            try:
                yolo_results = yolo(frame_rgb, classes=[0], conf=0.3, verbose=False)
                detected = []
                for r in yolo_results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                        detected.append([x1, y1, x2, y2])
                if detected:
                    boxes_list = detected
                print(f"   YOLO: {len(boxes_list)} person(s) detected")
            except Exception as e:
                print(f"   YOLO failed: {e}")

        inputs = proc(images=pil_frame, boxes=[boxes_list], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        t2 = time.time()
        with torch.no_grad():
            outputs = model_vit(**inputs)
        results = proc.post_process_pose_estimation(outputs, boxes=[boxes_list], threshold=0.3)
        dt2 = (time.time() - t2) * 1000

        n_persons = len(results[0]) if results else 0
        print(f"   [OK] Real frame inference done!")
        print(f"      Time: {dt2:.1f}ms, persons: {n_persons}")
        if results and results[0]:
            r0 = results[0][0]
            kpts = r0["keypoints"].cpu().numpy()
            scores = r0["scores"].cpu().numpy()
            visible = int((scores > 0.3).sum())
            print(f"      Person 0: visible joints (conf>0.3) = {visible}/17")

print()
print("="*60)
print("Environment verification complete!")
print()
print("Next steps:")
print("  1. Download HMR2 weights: python pose_extraction/scripts/download_weights.py")
print("  2. Run full pipeline test: python pose_extraction/scripts/test_pipeline.py --mode both")
print("="*60)
