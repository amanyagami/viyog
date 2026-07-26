"""Step 4 – Evaluate adversarial accuracy from stored HDF5 files.

Loads each model once, then evaluates all 6 attack HDF5 files for that model.
Also computes accuracy drop vs the clean baseline.

Run:
    CUDA_VISIBLE_DEVICES=5 uv run python experiments/04_eval_adversarial.py
Run (subset):
    CUDA_VISIBLE_DEVICES=5 uv run python experiments/04_eval_adversarial.py --models efficientnetv2_l
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.nn as nn
from tqdm import tqdm

import config
from config import ADV_DIR, ANALYSIS_DIR, ATTACKS, DEVICE, EVAL_BATCH, MODELS, NUM_CLASSES
from data_utils import adv_loader_from_h5
from model_utils import load_normalized_model

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "5")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Adversarial accuracy evaluation")
    p.add_argument("--dataset", default="cifar100", choices=list(config.DATASET_SPECS),
                   help="ID dataset (default: cifar100)")
    p.add_argument(
        "--models",
        nargs="+",
        default=list(config.MODEL_ARCHS),
        choices=list(config.MODEL_ARCHS),
        metavar="MODEL",
    )
    p.add_argument(
        "--attacks",
        nargs="+",
        default=list(ATTACKS.keys()),
        choices=list(ATTACKS.keys()),
        metavar="ATTACK",
        help="Which attack(s) to evaluate (default: all). Use to skip "
        "attacks still being generated.",
    )
    return p.parse_args()


@torch.no_grad()
def eval_accuracy(model: nn.Module, loader: torch.utils.data.DataLoader, device: str) -> dict[str, float]:
    model.eval()
    correct1 = correct5 = total = 0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(imgs)
        B = labels.size(0)
        total += B
        pred1 = logits.argmax(dim=1)
        correct1 += (pred1 == labels).sum().item()
        if NUM_CLASSES >= 5:
            _, pred5 = logits.topk(5, dim=1)
            correct5 += pred5.eq(labels.unsqueeze(1)).any(dim=1).sum().item()
    return {
        "top1": 100.0 * correct1 / total,
        "top5": 100.0 * correct5 / total if NUM_CLASSES >= 5 else None,
        "n_samples": total,
    }


def main() -> None:
    args = _parse_args()
    selected = args.models
    atk_subset = args.attacks
    config.set_dataset(args.dataset)
    global ADV_DIR, ANALYSIS_DIR, MODELS, NUM_CLASSES
    ADV_DIR, ANALYSIS_DIR = config.ADV_DIR, config.ANALYSIS_DIR
    MODELS, NUM_CLASSES = config.MODELS, config.NUM_CLASSES
    print(f"=== Step 4: Adversarial accuracy evaluation [{args.dataset}] ===")

    clean_path = ANALYSIS_DIR / "clean_accuracy.json"
    clean_results: dict = {}
    if clean_path.exists():
        with open(clean_path) as f:
            clean_results = json.load(f)

    # Accumulate into any existing partial results.
    out_path = ANALYSIS_DIR / "adversarial_accuracy.json"
    all_results: dict[str, dict] = {}
    if out_path.exists():
        with open(out_path) as f:
            all_results = json.load(f)

    for model_name in selected:
        arch, weight_path = MODELS[model_name]
        print(f"\n  Model: {model_name}")
        model = load_normalized_model(arch, weight_path, num_classes=config.NUM_CLASSES, device=DEVICE)
        model.eval()
        all_results.setdefault(model_name, {})

        clean_top1 = clean_results.get(model_name, {}).get("top1")
        if clean_top1 is not None:
            print(f"  Clean top-1 baseline: {clean_top1:.2f}%")

        for atk_name in atk_subset:
            h5_path = ADV_DIR / f"{model_name}_{atk_name}.h5"
            if not h5_path.exists():
                print(f"    [skip] {h5_path.name} not found")
                continue

            loader = adv_loader_from_h5(h5_path, batch_size=EVAL_BATCH, num_workers=0)
            t0 = time.time()
            metrics = eval_accuracy(
                model,
                tqdm(loader, desc=f"    {atk_name}", leave=False, dynamic_ncols=True),
                DEVICE,
            )
            elapsed = time.time() - t0

            drop = (clean_top1 - metrics["top1"]) if clean_top1 is not None else None
            all_results[model_name][atk_name] = {**metrics, "acc_drop": drop}

            msg = f"    {atk_name:<12} top-1={metrics['top1']:5.2f}%"
            if drop is not None:
                msg += f"  (↓{drop:.2f}%)"
            msg += f"  [{elapsed:.0f}s]"
            print(msg)

            loader.dataset.close()

        del model
        torch.cuda.empty_cache()

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved → {out_path}")

    print("\n  === Summary (top-1 adversarial accuracy %) ===")
    atk_names = list(ATTACKS.keys())
    header = f"  {'Model':<22}" + "".join(f" {a:>10}" for a in atk_names)
    print(header)
    print("  " + "-" * (22 + 11 * len(atk_names)))
    for mname, atk_dict in all_results.items():
        row = f"  {mname:<22}"
        for a in atk_names:
            val = atk_dict.get(a, {}).get("top1")
            row += f" {val:>9.2f}%" if val is not None else f" {'N/A':>9}"
        print(row)


if __name__ == "__main__":
    main()
