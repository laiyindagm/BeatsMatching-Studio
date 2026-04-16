"""
健壮下载 HMR2 权重 - 支持断点续传和自动重试。

用法:
    D:\\anaconda3\\envs\\pose_unified\\python.exe d:\\mao\\pose_extraction\\scripts\\download_weights_robust.py
"""
import os, sys, time, tarfile, urllib.request, urllib.error
from pathlib import Path

CACHE_DIR = Path(os.environ.get("HOME") or os.environ.get("USERPROFILE") or os.path.expanduser("~")) / ".cache"
CACHE_4DH = CACHE_DIR / "4DHumans"
URL = "https://www.cs.utexas.edu/~pavlakos/4dhumans/hmr2_data.tar.gz"
TAR = CACHE_4DH / "hmr2_data.tar.gz"
CKPT = CACHE_4DH / "logs/train/multiruns/hmr2/0/checkpoints/epoch=35-step=1000000.ckpt"
CHUNK = 4 * 1024 * 1024   # 4 MB per read

def download(url, dest, retries=20):
    dest.parent.mkdir(parents=True, exist_ok=True)
    done = dest.stat().st_size if dest.exists() else 0
    print(f"Downloading: {url}")
    print(f"Target: {dest}")
    print(f"Already downloaded: {done/1024/1024:.1f} MB")

    for attempt in range(retries):
        try:
            req = urllib.request.Request(url)
            if done > 0:
                req.add_header("Range", f"bytes={done}-")
            with urllib.request.urlopen(req, timeout=120) as resp:
                total = int(resp.headers.get("Content-Length", 0))
                if resp.status == 206:
                    total += done
                mode = "ab" if done > 0 and resp.status == 206 else "wb"
                if mode == "wb":
                    done = 0
                print(f"Attempt {attempt+1}: status={resp.status}, total={total/1024/1024:.1f}MB")
                t0 = time.time()
                with open(dest, mode) as f:
                    while True:
                        chunk = resp.read(CHUNK)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        elapsed = max(time.time() - t0, 0.001)
                        speed = done / elapsed / 1024 / 1024
                        pct = done * 100 // total if total > 0 else 0
                        sys.stdout.write(f"\r  {done/1024/1024:.1f}/{total/1024/1024:.1f}MB ({pct}%) {speed:.2f}MB/s")
                        sys.stdout.flush()
            print("\nDownload complete!")
            return True
        except Exception as e:
            print(f"\nAttempt {attempt+1} failed: {e}")
            done = dest.stat().st_size if dest.exists() else 0
            print(f"Will retry from {done/1024/1024:.1f} MB in 5s...")
            time.sleep(5)
    return False

def verify_tar(path):
    print(f"Verifying tar: {path}")
    size = path.stat().st_size
    print(f"  Size: {size/1024/1024:.1f} MB")
    if size < 2500 * 1024 * 1024:
        print(f"  WARNING: Expected ~2500MB, got {size/1024/1024:.1f}MB - file may be incomplete!")
        return False
    try:
        # Try plain tar first (server sends uncompressed tar despite .tar.gz extension)
        try:
            with tarfile.open(str(path), "r") as t:
                members = t.getmembers()
                print(f"  OK (plain tar): {len(members)} entries in archive")
                return True
        except tarfile.ReadError:
            pass
        # Fallback: try gzip
        with tarfile.open(str(path), "r:gz") as t:
            members = t.getmembers()
            print(f"  OK (gzip tar): {len(members)} entries in archive")
            return True
    except Exception as e:
        print(f"  FAIL: {e}")
        return False

def extract_tar(tar_path, dest_dir):
    print(f"Extracting {tar_path} -> {dest_dir}")
    # Try plain tar first
    try:
        with tarfile.open(str(tar_path), "r") as t:
            t.extractall(str(dest_dir))
        print("Extraction complete! (plain tar)")
        return
    except tarfile.ReadError:
        pass
    with tarfile.open(str(tar_path), "r:gz") as t:
        t.extractall(str(dest_dir))
    print("Extraction complete! (gzip tar)")

def main():
    CACHE_4DH.mkdir(parents=True, exist_ok=True)

    if CKPT.exists():
        print(f"[OK] Checkpoint already exists: {CKPT}")
    else:
        print("[INFO] Checkpoint not found, will download...")
        if TAR.exists() and not verify_tar(TAR):
            print("[INFO] Existing tar seems incomplete/corrupted, deleting...")
            TAR.unlink()

        if not TAR.exists():
            ok = download(URL, TAR)
            if not ok:
                print("Download failed after all retries.")
                sys.exit(1)
        else:
            print(f"[INFO] tar already exists at {TAR}, skipping download")

        if not verify_tar(TAR):
            print("[ERROR] Downloaded tar seems invalid!")
            sys.exit(1)

        extract_tar(TAR, CACHE_4DH)

    smpl = CACHE_4DH / "data/smpl/SMPL_NEUTRAL.pkl"
    if smpl.exists():
        print(f"[OK] SMPL model: {smpl}")
    else:
        print(f"[WARN] SMPL model not found: {smpl}")

    print("\nVerifying model load...")
    try:
        from hmr2.models import load_hmr2, DEFAULT_CHECKPOINT
        model, cfg = load_hmr2(DEFAULT_CHECKPOINT)
        print(f"[OK] HMR2 loaded! IMAGE_SIZE={cfg.MODEL.IMAGE_SIZE}")
        del model
    except Exception as e:
        print(f"[FAIL] Load error: {e}")
        raise

if __name__ == "__main__":
    main()
