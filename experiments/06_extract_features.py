"""Step 6 – Extract first-layer features for ID, OOD, and ADV splits.

For each model we extract from the first Conv2d (or patch-embed Conv2d for ViT):
  • filter_means (N, C): per-filter spatial mean |activation|
  • filter_maxs  (N, C): per-filter spatial max  |activation|
  • inf_norms    (N,):   per-sample ‖activation‖_∞ (Viyog statistic)
  • labels       (N,):   ground-truth class index

Extraction runs under torch.no_grad() — no graph is built, so activations are
freed immediately after the hook fires.  This allows batch sizes 4-8× larger than
adversarial generation (see config.MODEL_FEATURE_BATCH).

Features are stored as float16 in HDF5 to halve storage vs float32.

Output naming:
  features/feat_<model>_id.h5
  features/feat_<model>_ood_<ood_name>.h5
  features/feat_<model>_adv_<attack>.h5

Run (single GPU):
    CUDA_VISIBLE_DEVICES=5 uv run python experiments/06_extract_features.py
Run (subset of models, for multi-GPU):
    CUDA_VISIBLE_DEVICES=5 uv run python experiments/06_extract_features.py --models efficientnetv2_l
    CUDA_VISIBLE_DEVICES=6 uv run python experiments/06_extract_features.py --models convnextv2_base
    CUDA_VISIBLE_DEVICES=7 uv run python experiments/06_extract_features.py --models vit_base swin_tiny
"""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path

import torch
import torch.nn as nn
import config
from config import (
    ADV_DIR,
    ATTACKS,
    DEVICE,
    FEATURES_DIR,
    MODEL_FEATURE_BATCH,
    MODELS,
    N_FEATURE_SAMPLES,
    NUM_CLASSES,
    OOD_DATASETS,
)
from data_utils import (
    FeatureH5Writer,
    adv_loader_from_h5,
    get_id_loader,
    get_ood_loader,
)
from model_utils import FirstLayerHook, find_first_conv_in_normalized, load_normalized_model
from tqdm import tqdm

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "5")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract first-layer features")
    p.add_argument("--dataset", default="cifar100", choices=list(config.DATASET_SPECS),
                   help="ID dataset (default: cifar100)")
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
        help="Which adversarial attack split(s) to extract (default: all). "
        "Use to skip attacks still being generated.",
    )
    return p.parse_args()


def _count_filters(norm_model: nn.Module) -> int:
    _, layer = find_first_conv_in_normalized(norm_model)
    if layer is None:
        raise RuntimeError("No conv layer found.")
    return layer.out_channels


@torch.no_grad()
def extract_features(
    norm_model: nn.Module,
    loader: torch.utils.data.DataLoader,
    out_path: Path,
    n_total: int,
    n_filters: int,
    device: str,
    n_classes: int = NUM_CLASSES,
    desc: str = "",
) -> None:
    if out_path.exists():
        print(f"  [skip] {out_path.name}")
        return

    norm_model.eval()
    with (
        FirstLayerHook(norm_model) as hook,
        FeatureH5Writer(out_path, n_total, n_filters, n_classes=n_classes) as writer,
    ):
        written = 0
        for batch in tqdm(loader, desc=f"  {desc}", dynamic_ncols=True, leave=False):
            if written >= n_total:
                break

            if isinstance(batch, (list, tuple)):
                imgs = batch[0]
                labels = batch[1] if len(batch) > 1 else torch.full((imgs.shape[0],), -1)
            else:
                imgs = batch
                labels = torch.full((imgs.shape[0],), -1)

            imgs = imgs.to(device, non_blocking=True)
            logits = norm_model(imgs)  # forward triggers hook AND yields logits (free)
            feats = hook.features      # (B, C, H, W)

            remaining = n_total - written
            feats = feats[:remaining]
            labels = labels[:remaining]
            logits = logits[:remaining]

            writer.write_batch(feats, labels, logits)
            written += feats.shape[0]

    print(f"  ✓ {out_path.name}  [{written} samples]")


def process_model(
    model_name: str, arch: str, weight_path: Path | None, attacks: list[str],
    dataset: str,
) -> None:
    feat_batch = MODEL_FEATURE_BATCH[model_name]
    n_classes = config.NUM_CLASSES
    print(f"\n  === Model: {model_name}  (feature batch={feat_batch}) ===")

    norm_model = load_normalized_model(arch, weight_path, num_classes=n_classes, device=DEVICE)
    n_filters = _count_filters(norm_model)
    print(f"  First-layer filters: {n_filters}")

    # ---- ID test split ----
    id_loader = get_id_loader(dataset, batch_size=feat_batch, num_workers=4, train=False)
    id_n = min(N_FEATURE_SAMPLES, len(id_loader.dataset))
    extract_features(
        norm_model, id_loader,
        out_path=FEATURES_DIR / f"feat_{model_name}_id.h5",
        n_total=id_n, n_filters=n_filters, device=DEVICE, n_classes=n_classes,
        desc=f"{model_name}/id",
    )

    # ---- OOD: the dataset's OOD pool ----
    for ood_name in OOD_DATASETS:
        try:
            ood_loader = get_ood_loader(
                ood_name, batch_size=feat_batch, num_workers=4,
                max_samples=N_FEATURE_SAMPLES,
            )
            ood_n = min(N_FEATURE_SAMPLES, len(ood_loader.dataset))
            extract_features(
                norm_model, ood_loader,
                out_path=FEATURES_DIR / f"feat_{model_name}_ood_{ood_name}.h5",
                n_total=ood_n, n_filters=n_filters, device=DEVICE, n_classes=n_classes,
                desc=f"{model_name}/ood-{ood_name}",
            )
        except Exception as e:
            print(f"  [warn] OOD {ood_name} failed: {e}")

    # ---- ADV: requested attack types ----
    for atk_name in attacks:
        h5_path = ADV_DIR / f"{model_name}_{atk_name}.h5"
        if not h5_path.exists():
            print(f"  [skip] {h5_path.name} not found")
            continue
        adv_loader = adv_loader_from_h5(h5_path, batch_size=feat_batch, num_workers=4)
        adv_n = min(N_FEATURE_SAMPLES, len(adv_loader.dataset))
        extract_features(
            norm_model, adv_loader,
            out_path=FEATURES_DIR / f"feat_{model_name}_adv_{atk_name}.h5",
            n_total=adv_n, n_filters=n_filters, device=DEVICE, n_classes=n_classes,
            desc=f"{model_name}/adv-{atk_name}",
        )
        adv_loader.dataset.close()

    del norm_model
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    args = _parse_args()
    selected = args.models
    config.set_dataset(args.dataset)
    global ADV_DIR, FEATURES_DIR, MODELS, NUM_CLASSES, OOD_DATASETS
    ADV_DIR, FEATURES_DIR = config.ADV_DIR, config.FEATURES_DIR
    MODELS, NUM_CLASSES, OOD_DATASETS = config.MODELS, config.NUM_CLASSES, config.OOD_DATASETS
    print(f"=== Step 6: Feature extraction [{args.dataset}] ===")
    print(f"  Models: {selected}")
    print(f"  Attacks: {args.attacks}")
    print(f"  Output: {FEATURES_DIR}")
    print(f"  Max samples per split: {N_FEATURE_SAMPLES}")

    for model_name in selected:
        arch, weight_path = MODELS[model_name]
        process_model(model_name, arch, weight_path, args.attacks, args.dataset)

    h5_files = sorted(FEATURES_DIR.glob("feat_*.h5"))
    total_mb = sum(p.stat().st_size for p in h5_files) / 1e6
    print(f"\n  {len(h5_files)} feature files, {total_mb:.0f} MB total")
    print("=== Feature extraction complete ===")


if __name__ == "__main__":
    main()
