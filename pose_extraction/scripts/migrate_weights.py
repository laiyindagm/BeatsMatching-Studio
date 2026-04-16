"""
权重文件迁移脚本
将所有模型权重复制到 d:/毕业设计/weights/ 统一管理
"""
import os
import shutil
import sys

WEIGHTS_DIR = r"d:\毕业设计\weights"
USERPROFILE = os.environ.get("USERPROFILE", os.path.expanduser("~"))

# ─────────── 源路径 ───────────
SRC_HMR2 = os.path.join(USERPROFILE, ".cache", "4DHumans",
                         "logs", "train", "multiruns", "hmr2", "0",
                         "checkpoints", "epoch=35-step=1000000.ckpt")
SRC_SMPL  = os.path.join(USERPROFILE, ".cache", "4DHumans",
                         "data", "smpl", "SMPL_NEUTRAL.pkl")
SRC_YOLO  = r"d:\毕业设计\yolov8n.pt"

# ─────────── 目标路径 ───────────
DST_HMR2 = os.path.join(WEIGHTS_DIR, "hmr2", "epoch=35-step=1000000.ckpt")
DST_SMPL  = os.path.join(WEIGHTS_DIR, "smpl", "SMPL_NEUTRAL.pkl")
DST_YOLO  = os.path.join(WEIGHTS_DIR, "yolo", "yolov8n.pt")

# ViTPose: HuggingFace 模型采用 snapshot_download，
# 直接用软链接或者在代码里设置 HF_HOME 指向 weights/vitpose
DST_HF_HOME = os.path.join(WEIGHTS_DIR, "vitpose")

def copy_if_needed(src, dst):
    if not os.path.exists(src):
        print(f"  [SKIP] Source not found: {src}")
        return False
    if os.path.exists(dst):
        src_sz = os.path.getsize(src)
        dst_sz = os.path.getsize(dst)
        if src_sz == dst_sz:
            print(f"  [OK]   Already exists: {dst} ({dst_sz/1e6:.0f} MB)")
            return True
        else:
            print(f"  [WARN] Size mismatch, re-copying: {dst}")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    print(f"  [COPY] {src} -> {dst}  ({os.path.getsize(src)/1e6:.0f} MB) ...")
    shutil.copy2(src, dst)
    print(f"  [DONE] Copied.")
    return True

print("=== BeatsMatching 权重迁移 ===")
print()

print("[1] HMR2 Checkpoint")
copy_if_needed(SRC_HMR2, DST_HMR2)

print("[2] SMPL_NEUTRAL.pkl")
copy_if_needed(SRC_SMPL, DST_SMPL)

print("[3] YOLOv8n")
copy_if_needed(SRC_YOLO, DST_YOLO)

print("[4] ViTPose (HuggingFace cache)")
# 复制整个 HF hub 目录到 weights/vitpose/hub
src_hf = os.path.join(USERPROFILE, ".cache", "huggingface", "hub")
dst_hf = os.path.join(DST_HF_HOME, "hub")
if os.path.exists(src_hf):
    vitpose_dirs = [d for d in os.listdir(src_hf) if "vitpose" in d.lower()]
    for d in vitpose_dirs:
        src_d = os.path.join(src_hf, d)
        dst_d = os.path.join(dst_hf, d)
        if os.path.exists(dst_d):
            print(f"  [OK]   Already exists: {dst_d}")
        else:
            print(f"  [COPY] {src_d} -> {dst_d} ...")
            shutil.copytree(src_d, dst_d)
            print(f"  [DONE] Copied.")
else:
    print(f"  [SKIP] HF cache not found: {src_hf}")

print()
print("=== 最终权重目录结构 ===")
for root, dirs, files in os.walk(WEIGHTS_DIR):
    level = root.replace(WEIGHTS_DIR, '').count(os.sep)
    indent = '  ' * level
    print(f'{indent}{os.path.basename(root)}/')
    subindent = '  ' * (level + 1)
    for f in files:
        sz = os.path.getsize(os.path.join(root, f))
        print(f'{subindent}{f}  ({sz/1e6:.0f} MB)')

print()
print("=== 权重配置 (weights_config.py) ===")
print(f'WEIGHTS_DIR       = r"{WEIGHTS_DIR}"')
print(f'HMR2_CHECKPOINT   = r"{DST_HMR2}"')
print(f'SMPL_NEUTRAL      = r"{DST_SMPL}"')
print(f'YOLO_WEIGHTS      = r"{DST_YOLO}"')
print(f'HF_HOME           = r"{DST_HF_HOME}"')
