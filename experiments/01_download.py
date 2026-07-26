"""Step 1 – Download model weights.

Fetches the fine-tuned checkpoints from the canonical HuggingFace weights repo
(`config.HF_REPO`) via `snapshot_download`, which preserves the repo's
<dataset>/<model>.pth layout exactly as `config.weight_path()` expects it — no
renaming or moving needed after download. Falls back to Google Drive (via
gdown) only if the HF download fails outright.

These are the "author-created research objects" the AE reproduction pipeline
computes *from* (attacks, features, signatures are all regenerated live on
top of them) — nothing precomputed downstream is fetched here.

Run:
    uv run python experiments/01_download.py               # everything (33 files, ~1.1 GB)
    uv run python experiments/01_download.py --core-only    # just the 6 archs T1 needs (~250 MB)
"""

from __future__ import annotations

import argparse
import sys

from config import GDRIVE_FOLDER_ID, HF_LEGACY_WEIGHT_FILES, HF_REPO, WEIGHTS_DIR

# The 6 architectures the required T1 reproduction tier (see README) needs:
# convnextv2_base, swin_tiny, vit_base (legacy flat names) + densenet121,
# mobilenetv3_l, resnet50 (namespaced under cifar100/).
CORE_MODEL_HF_PATHS = [
    "convnextv2_base_cifar100.pth",
    "swin_tiny_patch4_window7_224_cifar100.pth",
    "vit_b_cifar100.pth",
    "cifar100/densenet121.pth",
    "cifar100/mobilenetv3_l.pth",
    "cifar100/resnet50.pth",
]

# Total .pth files in the canonical repo at authoring time (20-arch cifar100
# panel + cifar10 + gtsrb) — used only as an informational completeness count
# for the optional extended (T2) tier, not a hard requirement.
HF_EXPECTED_TOTAL_FILES = 34


def download_from_hf(core_only: bool) -> bool:
    """Download weights from the canonical HF repo. Returns True on success."""
    from huggingface_hub import hf_hub_download, snapshot_download

    print(f"  Fetching from https://huggingface.co/{HF_REPO} …")
    try:
        if core_only:
            for fname in CORE_MODEL_HF_PATHS:
                dest = WEIGHTS_DIR / fname
                if dest.exists():
                    print(f"  [skip] {fname} already exists ({dest.stat().st_size / 1e6:.0f} MB)")
                    continue
                local = hf_hub_download(
                    repo_id=HF_REPO,
                    filename=fname,
                    local_dir=str(WEIGHTS_DIR),
                    local_dir_use_symlinks=False,
                )
                print(f"  ✓ {fname} → {local}")
        else:
            snapshot_download(
                repo_id=HF_REPO,
                local_dir=str(WEIGHTS_DIR),
                local_dir_use_symlinks=False,
                allow_patterns=["*.pth"],
            )
            print("  ✓ full snapshot downloaded")
        return True
    except Exception as e:
        print(f"  ✗ HF download failed: {e}")
        return False


def download_from_gdrive() -> None:
    """Download weights from Google Drive folder using gdown (fallback only)."""
    try:
        import gdown
    except ImportError:
        print("  [error] gdown not installed. Run: uv add gdown")
        return

    print(f"\n  Downloading from Google Drive folder {GDRIVE_FOLDER_ID}…")
    url = f"https://drive.google.com/drive/folders/{GDRIVE_FOLDER_ID}"
    gdown.download_folder(url, output=str(WEIGHTS_DIR), quiet=False, use_cookies=False)


def verify_weights(core_only: bool) -> None:
    """Confirm the 6 core-model checkpoints exist (required); report total
    .pth coverage for the optional extended tier (informational only)."""
    print("\n--- Weight verification ---")
    all_ok = True
    for fname in CORE_MODEL_HF_PATHS:
        p = WEIGHTS_DIR / fname
        if p.exists():
            print(f"  ✓ {fname:40s}  {p.stat().st_size / 1e6:7.1f} MB")
        else:
            print(f"  ✗ MISSING (required for T1): {fname}")
            all_ok = False

    if not core_only:
        total = len(list(WEIGHTS_DIR.rglob("*.pth")))
        print(f"\n  Extended coverage: {total}/{HF_EXPECTED_TOTAL_FILES} .pth files present"
              f" (informational — only the 6 core models above are required).")

    if not all_ok:
        print("\n  Core weights are missing. Run step again or check network access.")
        sys.exit(1)
    print("\n  All required (core) weights present.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--core-only",
        action="store_true",
        help="Only fetch the 6 architectures the required T1 reproduction tier needs.",
    )
    args = parser.parse_args()

    print("=== Step 1: Download weights ===")
    print(f"Target directory: {WEIGHTS_DIR}")

    ok = download_from_hf(args.core_only)
    if not ok:
        print("\n  HuggingFace download failed. Trying Google Drive fallback…")
        download_from_gdrive()

    verify_weights(args.core_only)
