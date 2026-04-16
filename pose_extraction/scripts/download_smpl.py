"""
pose_extraction/scripts/download_smpl.py
下载 SMPL_NEUTRAL.pkl（SMPL 中性身体模型）。

来源：camenduru/4D-Humans（HuggingFace）

运行方式：
    conda activate pose_unified
    python pose_extraction/scripts/download_smpl.py
"""
import os, sys
from pathlib import Path

CACHE_DIR = Path(os.environ.get("HOME") or os.environ.get("USERPROFILE") or os.path.expanduser("~")) / ".cache"
CACHE_4DHUMANS = CACHE_DIR / "4DHumans"
DEST_FILE = CACHE_4DHUMANS / "data/smpl/SMPL_NEUTRAL.pkl"


def main():
    DEST_FILE.parent.mkdir(parents=True, exist_ok=True)

    if DEST_FILE.exists():
        print(f"[OK] SMPL_NEUTRAL.pkl already exists: {DEST_FILE}")
        print(f"  Size: {DEST_FILE.stat().st_size / 1024:.0f} KB")
        return

    print("Downloading SMPL_NEUTRAL.pkl from camenduru/4D-Humans...")

    try:
        from huggingface_hub import hf_hub_download
        import shutil

        path = hf_hub_download(
            repo_id="camenduru/4D-Humans",
            filename="smpl/SMPL_NEUTRAL.pkl",
            local_dir=str(DEST_FILE.parent),
            local_dir_use_symlinks=False,
        )
        src = Path(path)
        if src != DEST_FILE:
            shutil.copy2(str(src), str(DEST_FILE))
        print(f"[OK] Downloaded: {DEST_FILE} ({DEST_FILE.stat().st_size / 1024:.0f} KB)")

    except Exception as e:
        print(f"hf_hub_download failed: {e}")
        print("Trying direct HTTP download...")
        import urllib.request

        URL = "https://huggingface.co/camenduru/4D-Humans/resolve/main/smpl/SMPL_NEUTRAL.pkl"
        urllib.request.urlretrieve(URL, str(DEST_FILE))
        print(f"[OK] Downloaded: {DEST_FILE} ({DEST_FILE.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
