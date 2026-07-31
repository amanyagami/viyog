"""Step 3 – Generate adversarial samples for all attack types.

Normalization strategy (critical for correctness):
  - DataLoader returns raw [0,1] float32 images (no normalization).
  - NormalizedModel wraps the backbone and normalizes internally.
  - torchattacks sees the NormalizedModel as a black-box mapping [0,1]→logits.
  - Attacks perturb images in [0,1] space and clip to [0,1].
  - Generated adversarial images are stored as uint8 [0,255] in HDF5 (4× smaller
    than float32, zero information loss for 8-bit image data).

Batch sizes are per-model, sized for H200 126 GB VRAM (see config.MODEL_ATTACK_BATCH).
torch.cuda.empty_cache() is called only between models/attacks, never inside the
batch loop — repeated calls inside the loop stall the CUDA allocator.

File naming: data/adversarial/<model_name>_<attack_name>.h5

Run (single GPU):
    CUDA_VISIBLE_DEVICES=5 uv run python experiments/03_gen_adversarial.py
Run (subset of models, for multi-GPU):
    CUDA_VISIBLE_DEVICES=5 uv run python experiments/03_gen_adversarial.py --models efficientnetv2_l
    CUDA_VISIBLE_DEVICES=6 uv run python experiments/03_gen_adversarial.py --models convnextv2_base
    CUDA_VISIBLE_DEVICES=7 uv run python experiments/03_gen_adversarial.py --models vit_base swin_tiny
"""

from __future__ import annotations

import argparse
import gc
import os
import time
from pathlib import Path

import h5py
import torch
import torchattacks
from tqdm import tqdm

import config
from config import (
    ADV_DIR,
    ATTACK_MAX_SAMPLES,
    ATTACKS,
    DEVICE,
    MODEL_ATTACK_BATCH,
    MODELS,
)
from data_utils import AdvH5Writer, get_id_loader
from model_utils import load_normalized_model

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

_IMAGE_SHAPE = (3, 224, 224)


def _h5_is_complete(path: Path, expected_n: int) -> bool:
    """Return True if HDF5 exists, has the right number of samples, and is non-empty."""
    if not path.exists() or path.stat().st_size < 4096:
        return False
    try:
        with h5py.File(path, "r") as f:
            n = f["labels"].shape[0]
            return (
                n == expected_n
                and int(f["images"][0].max()) > 0
                and int(f["images"][-1].max()) > 0
            )
    except Exception:
        return False


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate adversarial samples")
    p.add_argument("--dataset", default="cifar100", choices=list(config.DATASET_SPECS),
                   help="ID dataset whose test split is attacked (default: cifar100)")
    p.add_argument(
        "--models",
        nargs="+",
        default=list(config.MODEL_ARCHS),
        choices=list(config.MODEL_ARCHS),
        metavar="MODEL",
        help="Which model(s) to process (default: all)",
    )
    p.add_argument(
        "--attacks",
        nargs="+",
        default=list(ATTACKS.keys()),
        choices=list(ATTACKS.keys()),
        metavar="ATTACK",
        help="Which attack(s) to generate (default: all). Lets DeepFool and CW "
        "run in separate passes at their own optimal batch size.",
    )
    p.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Override the per-model attack batch size (default: config value). "
        "DeepFool fits a large batch (tiny VRAM); CW needs a small one.",
    )
    return p.parse_args()


def _build_attack(name: str, cfg: dict, model: torch.nn.Module) -> torchattacks.Attack:
    cls = getattr(torchattacks, cfg["cls"])
    return cls(model, **cfg["kwargs"])


def generate_for_model(
    model_name: str,
    arch: str,
    weight_path: Path | None,
    attacks: list[str],
    dataset: str,
    batch_override: int | None = None,
) -> None:
    batch = batch_override if batch_override is not None else MODEL_ATTACK_BATCH[model_name]
    print(f"\n  === Model: {model_name}  (attack batch={batch}) ===")

    norm_model = load_normalized_model(arch, weight_path, num_classes=config.NUM_CLASSES, device=DEVICE)
    norm_model.eval()

    # One ID test loader for every attack; its length sets the full-set cap.
    loader = get_id_loader(dataset, batch_size=batch, num_workers=4, train=False)
    id_n = len(loader.dataset)

    for atk_name in attacks:
        atk_cfg = ATTACKS[atk_name]
        out_path = ADV_DIR / f"{model_name}_{atk_name}.h5"
        # Slow attacks (DeepFool, CW) are capped to keep wall-clock sane.
        n_samples = min(ATTACK_MAX_SAMPLES.get(atk_name, id_n), id_n)

        if _h5_is_complete(out_path, n_samples):
            print(f"  [skip] {out_path.name} already complete ({out_path.stat().st_size/1e6:.0f} MB)")
            continue
        elif out_path.exists():
            print(f"  [removing] {out_path.name} is incomplete — regenerating")
            out_path.unlink()

        cap = f"  (cap {n_samples})" if n_samples < id_n else ""
        print(f"\n    Attack: {atk_name}  cls={atk_cfg['cls']}  batch={batch}{cap}")
        atk = _build_attack(atk_name, atk_cfg, norm_model)

        t0 = time.time()
        written = 0
        with AdvH5Writer(out_path, n_samples, _IMAGE_SHAPE) as writer:
            for imgs, labels in tqdm(loader, desc=f"      {atk_name}", dynamic_ncols=True):
                if written >= n_samples:
                    break
                imgs = imgs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                adv_imgs = atk(imgs, labels)  # [0,1] float32
                take = n_samples - written
                writer.write_batch(adv_imgs[:take], labels[:take])
                written += min(take, adv_imgs.shape[0])
                # Do NOT call empty_cache() here — stalls CUDA allocator each batch.

        elapsed = time.time() - t0
        size_mb = out_path.stat().st_size / 1e6
        print(f"    ✓ {elapsed:.0f}s  →  {out_path.name}  ({size_mb:.0f} MB)")

        del atk
        gc.collect()
        torch.cuda.empty_cache()

    del norm_model
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    args = _parse_args()
    selected = args.models
    config.set_dataset(args.dataset)
    global ADV_DIR, MODELS
    ADV_DIR, MODELS = config.ADV_DIR, config.MODELS
    print(f"=== Step 3: Adversarial sample generation [{args.dataset}] ===")
    print(f"  Models:  {selected}")
    print(f"  Attacks: {args.attacks}")
    print(f"  Batch:   {args.batch if args.batch is not None else 'config default'}")
    print(f"  Output:  {ADV_DIR}")

    for model_name in selected:
        arch, weight_path = MODELS[model_name]
        generate_for_model(model_name, arch, weight_path, args.attacks, args.dataset, args.batch)

    print("\n=== Adversarial generation complete ===")
    print(f"{'File':<55} {'Size (MB)':>10}")
    print("-" * 67)
    for p in sorted(ADV_DIR.glob("*.h5")):
        print(f"{p.name:<55} {p.stat().st_size / 1e6:>10.1f}")


if __name__ == "__main__":
    main()
