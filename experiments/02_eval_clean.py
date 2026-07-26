"""Step 2 – Evaluate clean accuracy on CIFAR-100 test set.

Loads each of the four models and reports top-1 and top-5 accuracy on
all 10,000 CIFAR-100 test images.

Run:
    CUDA_VISIBLE_DEVICES=5 uv run python experiments/02_eval_clean.py
Run (subset):
    CUDA_VISIBLE_DEVICES=5 uv run python experiments/02_eval_clean.py --models efficientnetv2_l
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
from config import ANALYSIS_DIR, DEVICE, EVAL_BATCH, MODELS, NUM_CLASSES
from data_utils import get_id_loader
from model_utils import load_normalized_model

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "5")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean accuracy evaluation")
    p.add_argument("--dataset", default="cifar100", choices=list(config.DATASET_SPECS),
                   help="ID dataset to evaluate (default: cifar100)")
    p.add_argument(
        "--models",
        nargs="+",
        default=list(config.MODEL_ARCHS),
        choices=list(config.MODEL_ARCHS),
        metavar="MODEL",
    )
    return p.parse_args()


@torch.no_grad()
def evaluate(model: nn.Module, loader: torch.utils.data.DataLoader, device: str) -> dict[str, float]:
    model.eval()
    correct1 = correct5 = total = 0

    for imgs, labels in tqdm(loader, desc="  eval", leave=False, dynamic_ncols=True):
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
    config.set_dataset(args.dataset)
    global ANALYSIS_DIR, MODELS, NUM_CLASSES
    ANALYSIS_DIR, MODELS, NUM_CLASSES = config.ANALYSIS_DIR, config.MODELS, config.NUM_CLASSES
    print(f"=== Step 2: Clean accuracy evaluation [{args.dataset}] ===")

    loader = get_id_loader(args.dataset, batch_size=EVAL_BATCH, num_workers=4)
    print(f"  {args.dataset} test: {len(loader.dataset)} samples  batch={EVAL_BATCH}")

    # Load any existing results so partial runs accumulate correctly.
    out_path = ANALYSIS_DIR / "clean_accuracy.json"
    results: dict[str, dict] = {}
    if out_path.exists():
        with open(out_path) as f:
            results = json.load(f)

    for name in selected:
        arch, weight_path = MODELS[name]
        print(f"\n  Model: {name} ({arch})")
        t0 = time.time()
        model = load_normalized_model(arch, weight_path, num_classes=config.NUM_CLASSES, device=DEVICE)
        print(f"  Loaded in {time.time() - t0:.1f}s")

        metrics = evaluate(model, loader, DEVICE)
        results[name] = metrics
        print(f"  Top-1: {metrics['top1']:.2f}%  Top-5: {metrics['top5']:.2f}%")

        del model
        torch.cuda.empty_cache()

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved → {out_path}")

    print(f"\n  {'Model':<22} {'Top-1':>7} {'Top-5':>7}")
    print("  " + "-" * 38)
    for name, m in results.items():
        print(f"  {name:<22} {m['top1']:>6.2f}% {m['top5']:>6.2f}%")


if __name__ == "__main__":
    main()
