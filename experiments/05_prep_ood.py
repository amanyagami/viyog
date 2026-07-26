"""Step 5 – Download and verify the 8 OOD datasets.

All datasets are downloaded via torchvision to data/ood/<name>/.
This script just pre-downloads; actual feature extraction happens in step 6.

OOD taxonomy wrt CIFAR-100:
  near_ood    – semantically related (cifar10, stl10, flowers102, food101)
  far_ood     – structurally different (svhn, mnist, fashionmnist)
  texture_ood – no semantic objects   (dtd)

Run:
    CUDA_VISIBLE_DEVICES=5 uv run python experiments/05_prep_ood.py
"""

from __future__ import annotations

import os

from config import OOD_DATASETS, OOD_ROOT
from data_utils import get_ood_loader

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "5")


def main() -> None:
    print("=== Step 5: OOD dataset preparation ===")
    print(f"  Root: {OOD_ROOT}\n")

    for name, meta in OOD_DATASETS.items():
        print(f"  Dataset: {name:<14}  kind={meta['kind']}")
        print(f"    Note: {meta['note']}")
        try:
            loader = get_ood_loader(name, batch_size=256, num_workers=4, max_samples=None)
            n = len(loader.dataset)
            # Access first batch to confirm dataset is readable
            imgs, *_ = next(iter(loader))
            print(f"    ✓ {n} samples, image shape: {tuple(imgs.shape[1:])}")
        except Exception as e:
            print(f"    ✗ Failed: {e}")
        print()

    print("=== OOD preparation complete ===")


if __name__ == "__main__":
    main()
