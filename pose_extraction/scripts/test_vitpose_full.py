"""
test_vitpose_full.py - ViTPose full pipeline test (output to d:\\tmp)
"""
import os, sys, time
import cv2
import numpy as np
import torch

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

# NOTE: Absolute paths using raw strings with actual Chinese chars stored in UTF-8 source
PROJ = r"d:\毕业设计"
sys.path.insert(0, PROJ)
VIDEO = r"d:\毕业设计\BeatsMatching\output.mp4"
OUT_VIS = r"d:\tmp\vitpose_vis.mp4"
MAX_FRAMES = 60
SAVE_VIS = True

print("=" * 60)
print("ViTPose Full Pipeline Test")
print("=" * 60)
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

t0 = time.time()
from pose_extraction.vitpose.detector import ViTPoseDetector
detector = ViTPoseDetector(
    vitpose_model="usyd-community/vitpose-base-simple",
    device="auto",
    yolo_model="yolov8n.pt",
)
print("[OK] Model loaded in %.1fs  device=%s" % (time.time()-t0, detector.device))

cap = cv2.VideoCapture(VIDEO)
assert cap.isOpened(), "Cannot open: " + VIDEO
fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("Video: output.mp4")
print("  %dx%d  %.1ffps  %d frames" % (W, H, fps, total))
print("  Testing %d frames" % min(MAX_FRAMES, total))

out_writer = None
if SAVE_VIS:
    os.makedirs(r"d:\tmp", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(OUT_VIS, fourcc, fps, (W, H))
    print("  Vis output:", OUT_VIS)

frame_results = []
frame_times = []
frame_count = 0

while frame_count < MAX_FRAMES:
    ret, frame_bgr = cap.read()
    if not ret:
        break
    ts = frame_count / fps
    t1 = time.time()
    result = detector.detect_frame(frame_bgr, frame_idx=frame_count, timestamp=ts)
    dt = time.time() - t1
    frame_times.append(dt)
    frame_results.append(result)

    if SAVE_VIS and out_writer:
        vis = detector.visualize_frame(frame_bgr, result)
        out_writer.write(vis)

    if frame_count % 10 == 0:
        n = len(result.persons)
        print("  Frame %4d | %5.1fms | persons=%d" % (frame_count, dt*1000, n))

    frame_count += 1

cap.release()
if out_writer:
    out_writer.release()

avg_ms = float(np.mean(frame_times)) * 1000
max_ms = float(np.max(frame_times)) * 1000
total_det = sum(len(r.persons) for r in frame_results)
est_fps_val = 1000.0 / avg_ms if avg_ms > 0 else 0

print()
print("=" * 60)
print("ViTPose Summary")
print("=" * 60)
print("  Frames: %d" % frame_count)
print("  Detections: %d" % total_det)
print("  Avg: %.1fms   Max: %.1fms   ~%.1f FPS" % (avg_ms, max_ms, est_fps_val))

if frame_results and frame_results[0].persons:
    pid = list(frame_results[0].persons.keys())[0]
    kpts = frame_results[0].persons[pid]
    vis_j = int(np.sum(kpts[:, 2] > 0.3))
    print("  Frame0 Person0: %d/17 joints visible" % vis_j)
    for j in range(min(5, len(kpts))):
        print("    Joint%d: (%.1f, %.1f, %.3f)" % (j, kpts[j,0], kpts[j,1], kpts[j,2]))

# Beat detection
print()
print("=" * 60)
print("Beat Detection Test")
print("=" * 60)
from pose_extraction.vitpose.video_processor import ViTPoseVideoProcessor
from pose_extraction.core.beat_detector import detect_beats

processor2 = ViTPoseVideoProcessor(device="auto")
print("Processing 60 frames for beat detection...")
seq2 = processor2.process_video(VIDEO, start_frame=0, end_frame=60)
print("[OK] Extracted %d/%d frames" % (len(seq2.frames), seq2.total_frames))

# Convert ViTPoseSequence -> PoseTimeSeries
pts2 = processor2.to_pose_time_series(seq2, person_id=0)
print("[OK] PoseTimeSeries: coords shape=%s" % str(pts2.joint_coords.shape if pts2.joint_coords is not None else "None"))

# Run beat detection
pts2 = detect_beats(pts2, strategy="vitpose")
beat_frames_list = pts2.beat_frames or []
beat_ts_list = pts2.beat_timestamps or []
print("[OK] Detected %d beats" % len(beat_frames_list))
if beat_ts_list:
    print("  Times(s): [%s]" % ", ".join("%.2f" % t for t in beat_ts_list[:10]))
    if len(beat_ts_list) > 1:
        diffs = np.diff(beat_ts_list)
        avg_int = float(np.mean(diffs))
        print("  Avg interval: %.3fs  (~%.0f bpm)" % (avg_int, 60/avg_int))

print()
print("ALL DONE - ViTPose pipeline verified!")
