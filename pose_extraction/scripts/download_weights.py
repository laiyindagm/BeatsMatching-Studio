"""
下载 HMR2 权重和 SMPL 数据文件。

运行方式：
    conda activate pose_unified
    python pose_extraction/scripts/download_weights.py

文件会被放置在 ~/.cache/4DHumans/
"""
import os
import sys
import urllib.request
import tarfile
from pathlib import Path

CACHE_DIR = Path(os.environ.get("HOME") or os.environ.get("USERPROFILE") or os.path.expanduser("~")) / ".cache"
CACHE_4DHUMANS = CACHE_DIR / "4DHumans"
URL = "https://www.cs.utexas.edu/~pavlakos/4dhumans/hmr2_data.tar.gz"
TAR_PATH = CACHE_4DHUMANS / "hmr2_data.tar.gz"
DEFAULT_CKPT = CACHE_4DHUMANS / "logs/train/multiruns/hmr2/0/checkpoints/epoch=35-step=1000000.ckpt"


def download_with_progress(url: str, dest: Path, chunk_size: int = 1024 * 1024) -> None:
    """带进度条的下载函数，支持断点续传。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"下载: {url}")
    print(f"目标: {dest}")
    
    # 检查已下载的部分
    downloaded = 0
    if dest.exists():
        downloaded = dest.stat().st_size
        print(f"  已存在部分文件: {downloaded / 1024 / 1024:.1f}MB")
    
    # 创建请求，支持断点续传
    req = urllib.request.Request(url)
    if downloaded > 0:
        req.add_header("Range", f"bytes={downloaded}-")
    
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            total_size = int(response.headers.get('Content-Length', 0))
            if downloaded > 0 and response.status == 206:
                total_size += downloaded
            
            mode = 'ab' if downloaded > 0 and response.status == 206 else 'wb'
            with open(dest, mode) as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = min(100, downloaded * 100 // total_size)
                        mb_done = downloaded / 1024 / 1024
                        mb_total = total_size / 1024 / 1024
                        sys.stdout.write(f"\r  {mb_done:.1f}MB / {mb_total:.1f}MB ({pct}%)")
                        sys.stdout.flush()
            print()
    except Exception as e:
        print(f"\n下载出错: {e}")
        print(f"已下载 {downloaded / 1024 / 1024:.1f}MB，下次运行会尝试断点续传")
        raise


def main():
    CACHE_4DHUMANS.mkdir(parents=True, exist_ok=True)

    # 检查是否已存在
    if DEFAULT_CKPT.exists():
        print(f"[OK] HMR2 权重已存在: {DEFAULT_CKPT}")
    else:
        print("HMR2 权重不存在，开始下载...")
        if not TAR_PATH.exists():
            download_with_progress(URL, TAR_PATH)
        else:
            print(f"压缩包已存在: {TAR_PATH}，直接解压...")

        print("解压中...")
        # 注意：服务器实际返回普通 tar（非 gzip），虽然扩展名是 .tar.gz
        try:
            with tarfile.open(str(TAR_PATH), "r") as tar:
                tar.extractall(str(CACHE_4DHUMANS))
        except tarfile.ReadError:
            # 如果不是普通 tar，尝试 gzip
            with tarfile.open(str(TAR_PATH), "r:gz") as tar:
                tar.extractall(str(CACHE_4DHUMANS))
        print(f"[OK] 解压完成: {CACHE_4DHUMANS}")

    # 检查 SMPL 模型
    smpl_path = CACHE_4DHUMANS / "data/smpl/SMPL_NEUTRAL.pkl"
    if smpl_path.exists():
        print(f"[OK] SMPL 模型已存在: {smpl_path}")
    else:
        print(f"[WARN] SMPL 模型不存在: {smpl_path}")
        print("   请从 HuggingFace 下载：camenduru/4D-Humans/blob/main/smpl/SMPL_NEUTRAL.pkl")
        print("   或运行：python d:/tmp/download_smpl.py")

    # 验证加载
    print("\n验证 HMR2 模型加载...")
    try:
        from hmr2.models import load_hmr2, DEFAULT_CHECKPOINT
        model, cfg = load_hmr2(DEFAULT_CHECKPOINT)
        print(f"[OK] HMR2 模型加载成功！IMAGE_SIZE={cfg.MODEL.IMAGE_SIZE}")
        del model
    except Exception as e:
        print(f"[FAIL] 加载失败: {e}")
        raise


if __name__ == "__main__":
    main()
