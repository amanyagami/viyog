"""Step 1 – Download model weights.

Downloads the four CIFAR-100 fine-tuned weights from HuggingFace.
Falls back to Google Drive (via gdown) if HF download fails.

Run:
    CUDA_VISIBLE_DEVICES=5 uv run python experiments/01_download.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from config import GDRIVE_FOLDER_ID, HF_REPO, HF_WEIGHT_FILES, WEIGHTS_DIR


def download_from_hf() -> list[str]:
    """Download weights from HuggingFace hub. Returns list of failed files."""
    from huggingface_hub import hf_hub_download

    failed: list[str] = []
    for fname in HF_WEIGHT_FILES:
        dest = WEIGHTS_DIR / fname
        if dest.exists():
            print(f"  [skip] {fname} already exists ({dest.stat().st_size / 1e6:.0f} MB)")
            continue
        print(f"  Downloading {fname} from HF…")
        try:
            local = hf_hub_download(
                repo_id=HF_REPO,
                filename=fname,
                local_dir=str(WEIGHTS_DIR),
                local_dir_use_symlinks=False,
            )
            print(f"  ✓ {fname} → {local}")
        except Exception as e:
            print(f"  ✗ HF download failed for {fname}: {e}")
            failed.append(fname)
    return failed


def download_from_gdrive() -> None:
    """Download weights from Google Drive folder using gdown."""
    try:
        import gdown
    except ImportError:
        print("  [error] gdown not installed. Run: uv add gdown")
        return

    print(f"\n  Downloading from Google Drive folder {GDRIVE_FOLDER_ID}…")
    url = f"https://drive.google.com/drive/folders/{GDRIVE_FOLDER_ID}"
    gdown.download_folder(url, output=str(WEIGHTS_DIR), quiet=False, use_cookies=False)


def verify_weights() -> None:
    """Print size of each weight file and warn if any are missing."""
    print("\n--- Weight verification ---")
    all_ok = True
    for fname in HF_WEIGHT_FILES:
        p = WEIGHTS_DIR / fname
        if p.exists():
            print(f"  ✓ {fname:50s}  {p.stat().st_size / 1e6:7.1f} MB")
        else:
            print(f"  ✗ MISSING: {fname}")
            all_ok = False
    if not all_ok:
        print("\n  Some weights are missing. Run step again or check network access.")
        sys.exit(1)
    else:
        print("\n  All weights present.")


if __name__ == "__main__":
    print("=== Step 1: Download weights ===")
    print(f"Target directory: {WEIGHTS_DIR}")

    failed = download_from_hf()

    if failed:
        print(f"\n  {len(failed)} file(s) failed via HuggingFace. Trying Google Drive…")
        download_from_gdrive()

    verify_weights()
