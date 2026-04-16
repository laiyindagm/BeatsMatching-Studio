# pose_extraction/weights_config.py
"""
统一权重路径配置
所有模型权重位于项目根目录下的 weights/ 文件夹
"""
import os

# ─── 根目录 ────────────────────────────────────────────────────────────────────
_THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(os.path.dirname(_THIS_DIR), "weights")

# ─── 各模型权重 ────────────────────────────────────────────────────────────────
HMR2_CHECKPOINT = os.path.join(WEIGHTS_DIR, "hmr2", "checkpoints", "epoch=35-step=1000000.ckpt")
SMPL_NEUTRAL    = os.path.join(WEIGHTS_DIR, "smpl", "SMPL_NEUTRAL.pkl")
YOLO_WEIGHTS    = os.path.join(WEIGHTS_DIR, "yolo", "yolov8n.pt")

# ─── HuggingFace 离线缓存 ──────────────────────────────────────────────────────
# 设置此目录为 HF_HOME，使 transformers 从本地加载 ViTPose
HF_HOME      = os.path.join(WEIGHTS_DIR, "vitpose")
VITPOSE_MODEL_ID = "usyd-community/vitpose-base-simple"

def verify_all():
    """启动时验证所有权重文件是否存在"""
    checks = {
        "HMR2 checkpoint": HMR2_CHECKPOINT,
        "SMPL_NEUTRAL":    SMPL_NEUTRAL,
        "YOLOv8n":         YOLO_WEIGHTS,
        "ViTPose HF home": HF_HOME,
    }
    ok = True
    for name, path in checks.items():
        exists = os.path.exists(path)
        status = "OK  " if exists else "MISS"
        print(f"  [{status}] {name}: {path}")
        if not exists:
            ok = False
    return ok


if __name__ == "__main__":
    print("=== 权重文件验证 ===")
    all_ok = verify_all()
    print()
    print("All OK!" if all_ok else "Some weights are MISSING! Run scripts/migrate_weights.py first.")
