"""Step 1 – Download model weights.

Fetches fine-tuned checkpoints from the canonical Hugging Face weights repo at
the immutable revision in ``config.HF_REVISION``. The repository's
``<dataset>/<model>.pth`` layout is preserved exactly as ``config.weight_path``
expects it. There is deliberately no secondary download source: legacy mirrors
have incompatible filenames and cannot reproduce this artifact.

These are the "author-created research objects" the AE reproduction pipeline
computes *from* (attacks, features, signatures are all regenerated live on top
of them) — nothing precomputed downstream is fetched here.

Run:
    uv run --frozen python experiments/01_download.py
        Full snapshot: 34 checkpoints, 5.00 GB (4.66 GiB).
    uv run --frozen python experiments/01_download.py --core-only
        T1 subset: 6 checkpoints, 1.58 GB (1.47 GiB).
"""

from __future__ import annotations

import argparse
import sys

from config import HF_REPO, HF_REVISION, WEIGHTS_DIR

# The 6 architectures the required T1 reproduction tier (see README) needs:
# convnextv2_base, swin_tiny, vit_base (legacy flat names) + densenet121,
# mobilenetv3_l, resnet50 (namespaced under cifar100/).
CORE_MODEL_HF_PATHS = (
    "convnextv2_base_cifar100.pth",
    "swin_tiny_patch4_window7_224_cifar100.pth",
    "vit_b_cifar100.pth",
    "cifar100/densenet121.pth",
    "cifar100/mobilenetv3_l.pth",
    "cifar100/resnet50.pth",
)

# Immutable-revision facts verified from the Hugging Face API on 2026-08-08.
HF_EXPECTED_TOTAL_FILES = 34
HF_EXPECTED_TOTAL_BYTES = 5_003_987_510
CORE_EXPECTED_BYTES = 1_579_316_012


def _human_size(size: int) -> str:
    """Return decimal and binary gigabytes for a byte count."""
    return f"{size / 1e9:.2f} GB ({size / 2**30:.2f} GiB)"


def download_from_hf(core_only: bool) -> None:
    """Download weights from the canonical repository and pinned revision."""
    from huggingface_hub import hf_hub_download, snapshot_download

    print(f"  Fetching https://huggingface.co/{HF_REPO}/tree/{HF_REVISION}")
    if core_only:
        for filename in CORE_MODEL_HF_PATHS:
            local = hf_hub_download(
                repo_id=HF_REPO,
                filename=filename,
                revision=HF_REVISION,
                local_dir=str(WEIGHTS_DIR),
            )
            print(f"  ✓ {filename} → {local}")
        return

    snapshot_download(
        repo_id=HF_REPO,
        revision=HF_REVISION,
        local_dir=str(WEIGHTS_DIR),
        allow_patterns=["*.pth"],
    )
    print("  ✓ full 34-checkpoint snapshot downloaded")


def verify_weights(core_only: bool) -> bool:
    """Verify required paths, immutable payload sizes, and full-tier coverage."""
    print("\n--- Weight verification ---")
    all_ok = True
    core_bytes = 0
    for filename in CORE_MODEL_HF_PATHS:
        path = WEIGHTS_DIR / filename
        if path.is_file():
            size = path.stat().st_size
            core_bytes += size
            print(f"  ✓ {filename:40s}  {size / 1e6:7.1f} MB")
        else:
            print(f"  ✗ MISSING (required for T1): {filename}")
            all_ok = False

    if all_ok:
        print(f"\n  Core payload: {_human_size(core_bytes)}")
        if core_bytes != CORE_EXPECTED_BYTES:
            print(
                "  ✗ Core payload differs from pinned revision: "
                f"expected {CORE_EXPECTED_BYTES:,} bytes, found {core_bytes:,}"
            )
            all_ok = False

    if not core_only:
        paths = list(WEIGHTS_DIR.rglob("*.pth"))
        total = len(paths)
        print(f"\n  Extended coverage: {total}/{HF_EXPECTED_TOTAL_FILES} .pth files present")
        if total < HF_EXPECTED_TOTAL_FILES:
            print("  ✗ Full snapshot is incomplete.")
            all_ok = False
        elif total == HF_EXPECTED_TOTAL_FILES:
            total_bytes = sum(path.stat().st_size for path in paths)
            print(f"  Full payload: {_human_size(total_bytes)}")
            if total_bytes != HF_EXPECTED_TOTAL_BYTES:
                print(
                    "  ✗ Full payload differs from pinned revision: "
                    f"expected {HF_EXPECTED_TOTAL_BYTES:,} bytes, found {total_bytes:,}"
                )
                all_ok = False

    if not all_ok:
        print("\n  Weight verification failed. Retry the pinned Hugging Face download.")
        return False
    print("\n  All required (core) weights present.")
    return True


def main(argv: list[str] | None = None) -> int:
    """Download and verify the requested immutable checkpoint tier."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--core-only",
        action="store_true",
        help="Only fetch the 6 architectures the required T1 reproduction tier needs.",
    )
    args = parser.parse_args(argv)

    print("=== Step 1: Download weights ===")
    print(f"Target directory: {WEIGHTS_DIR}")
    print(f"Pinned revision: {HF_REVISION}")

    try:
        download_from_hf(args.core_only)
    except Exception as exc:
        print(f"\n  Pinned Hugging Face download failed: {exc}", file=sys.stderr)
        print("  No alternate source is supported; retry this exact revision.", file=sys.stderr)
        return 1

    return 0 if verify_weights(args.core_only) else 1


if __name__ == "__main__":
    raise SystemExit(main())
