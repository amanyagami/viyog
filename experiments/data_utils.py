"""Dataset loading and HDF5 storage utilities.

Covers:
- CIFAR-100 test loader (ID dataset)
- 8 OOD dataset loaders (torchvision, resize to 224×224, normalise to [0,1])
- HDF5 adversarial sample writer/reader
- HDF5 feature writer/reader
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from config import DATA_DIR, DATASET_SPECS, IMAGE_SIZE, OOD_ROOT, OOD_UNIVERSE
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

_BICUBIC = transforms.InterpolationMode.BICUBIC

# ---------------------------------------------------------------------------
# Shared transforms
# ---------------------------------------------------------------------------

# Raw [0,1] – for attack generation (no normalisation; NormalizedModel handles it)
_RAW_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),  # → float32 [0,1]
])

# 3-channel conversion for grayscale datasets (MNIST, FashionMNIST)
_RAW_TRANSFORM_GRAY = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])


def get_cifar100_loader(
    root: str | Path = "/mnt/data1/asing725/viyog/data/cifar",
    batch_size: int = 256,
    num_workers: int = 4,
    train: bool = False,
    shuffle: bool = False,
) -> DataLoader:
    """Return a DataLoader for CIFAR-100 (test split by default).

    Images are resized to IMAGE_SIZE×IMAGE_SIZE and returned as [0,1] float32.
    No normalisation – pass through NormalizedModel before the backbone.
    """
    ds = datasets.CIFAR100(
        root=str(root),
        train=train,
        download=True,
        transform=_RAW_TRANSFORM,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )


# ---------------------------------------------------------------------------
# Generic ID dataset loaders (multi-dataset extension)
# ---------------------------------------------------------------------------
# All loaders return [0,1] float32 tensors (no normalisation — NormalizedModel
# handles it, so finetuning, attack generation, and feature extraction all see
# pixel space). Train loaders add light, dataset-appropriate augmentation;
# horizontal flip is disabled for digit/sign datasets (svhn/gtsrb/mnist).


def _id_transform(train: bool, gray: bool, hflip: bool, aug: bool) -> transforms.Compose:
    """Build a [0,1] transform for an ID dataset split."""
    ops: list = []
    if train:
        ops.append(transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.65, 1.0),
                                                 interpolation=_BICUBIC))
        if hflip:
            ops.append(transforms.RandomHorizontalFlip())
        if aug and not gray:
            ops.append(transforms.RandAugment())  # PIL-space, before ToTensor
    else:
        ops.append(transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=_BICUBIC))
    if gray:
        ops.append(transforms.Grayscale(num_output_channels=3))
    ops.append(transforms.ToTensor())
    return transforms.Compose(ops)


def _id_root(name: str) -> str:
    """Reuse the already-downloaded OOD copy when a dataset doubles as OOD;
    otherwise store under data/id/<name>."""
    base = (OOD_ROOT / name) if name in OOD_UNIVERSE else (DATA_DIR / "id" / name)
    return str(base)


def _build_id_dataset(name: str, train: bool, transform: transforms.Compose) -> Dataset:
    """Instantiate a torchvision ID dataset for the requested split."""
    spec = DATASET_SPECS[name]
    cls = getattr(datasets, spec["tv_cls"])
    root = _id_root(name)

    if spec.get("needs_split"):
        # No native split (e.g. EuroSAT) → deterministic 80/20 partition.
        full = cls(root, download=True, transform=transform)
        n = len(full)
        n_test = int(0.2 * n)
        g = torch.Generator().manual_seed(1234)
        perm = torch.randperm(n, generator=g).tolist()
        idx = perm[n_test:] if train else perm[:n_test]
        return Subset(full, idx)

    kw = dict(spec["train_kw"] if train else spec["test_kw"])
    if name == "pets":
        kw["target_types"] = "category"
    return cls(root, download=True, transform=transform, **kw)


def get_id_train_loader(
    dataset: str,
    batch_size: int = 128,
    num_workers: int = 8,
    aug: bool = True,
) -> DataLoader:
    """Augmented training loader for an ID dataset (for finetuning)."""
    spec = DATASET_SPECS[dataset]
    tfm = _id_transform(True, spec["gray"], spec["hflip"], aug)
    ds = _build_id_dataset(dataset, train=True, transform=tfm)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                      pin_memory=True, drop_last=True,
                      persistent_workers=(num_workers > 0))


def get_id_eval_loader(
    dataset: str,
    batch_size: int = 256,
    num_workers: int = 4,
    train: bool = False,
) -> DataLoader:
    """Deterministic [0,1] loader for an ID split (eval / adv source / features)."""
    spec = DATASET_SPECS[dataset]
    tfm = _id_transform(False, spec["gray"], False, False)
    ds = _build_id_dataset(dataset, train=train, transform=tfm)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                      pin_memory=True, persistent_workers=(num_workers > 0))


def get_id_loader(
    dataset: str,
    batch_size: int = 256,
    num_workers: int = 4,
    train: bool = False,
) -> DataLoader:
    """ID test/train loader used by eval (02), adv generation (03) and feature
    extraction (06). CIFAR-100 routes through the legacy loader so the original
    artifacts/paths are reused exactly."""
    if dataset == "cifar100":
        return get_cifar100_loader(batch_size=batch_size, num_workers=num_workers,
                                   train=train, shuffle=False)
    return get_id_eval_loader(dataset, batch_size=batch_size,
                              num_workers=num_workers, train=train)


# ---------------------------------------------------------------------------
# OOD dataset loaders
# ---------------------------------------------------------------------------

def _make_ood_loader(
    ds: Dataset,
    batch_size: int = 256,
    num_workers: int = 4,
    max_samples: int | None = None,
) -> DataLoader:
    if max_samples is not None and len(ds) > max_samples:
        indices = list(range(max_samples))
        ds = Subset(ds, indices)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=False,
    )


def get_ood_loader(
    name: str,
    batch_size: int = 256,
    num_workers: int = 4,
    max_samples: int | None = 5000,
) -> DataLoader:
    """Return a DataLoader for one of the 8 OOD datasets.

    All datasets are resized to IMAGE_SIZE×IMAGE_SIZE and returned as [0,1]
    float32 tensors. Grayscale datasets are broadcast to 3 channels.

    Parameters
    ----------
    name : str
        One of: cifar10, svhn, stl10, dtd, mnist, fashionmnist,
                flowers102, food101
    """
    root = str(OOD_ROOT / name)
    gray_datasets = {"mnist", "fashionmnist"}
    tfm = _RAW_TRANSFORM_GRAY if name in gray_datasets else _RAW_TRANSFORM

    tv = datasets  # alias

    if name == "cifar10":
        ds = tv.CIFAR10(root, train=False, download=True, transform=tfm)
    elif name == "svhn":
        ds = tv.SVHN(root, split="test", download=True, transform=tfm)
    elif name == "stl10":
        ds = tv.STL10(root, split="test", download=True, transform=tfm)
    elif name == "dtd":
        ds = tv.DTD(root, split="test", download=True, transform=tfm)
    elif name == "mnist":
        ds = tv.MNIST(root, train=False, download=True, transform=tfm)
    elif name == "fashionmnist":
        ds = tv.FashionMNIST(root, train=False, download=True, transform=tfm)
    elif name == "flowers102":
        ds = tv.Flowers102(root, split="test", download=True, transform=tfm)
    elif name == "food101":
        ds = tv.Food101(root, split="test", download=True, transform=tfm)
    elif name in DATASET_SPECS:
        # Generic fallback for newer OOD sets (eurosat, gtsrb, …): reuse the ID
        # builder's test split so EuroSAT's deterministic split etc. are honored.
        ds = _build_id_dataset(name, train=False, transform=tfm)
    else:
        raise ValueError(f"Unknown OOD dataset: {name}")

    return _make_ood_loader(ds, batch_size=batch_size, num_workers=num_workers,
                            max_samples=max_samples)


# ---------------------------------------------------------------------------
# HDF5 adversarial sample storage
# ---------------------------------------------------------------------------
# Layout:
#   adv_<model>_<attack>.h5
#     images   (N, 3, H, W) uint8   – adversarial pixels [0,255]
#     labels   (N,)          int32   – ground-truth class indices
#
# uint8 gives 4× storage reduction vs float32 with zero information loss
# for standard 8-bit image data.

class AdvH5Writer:
    """Streaming HDF5 writer for adversarial uint8 images."""

    def __init__(self, path: Path, n_total: int, image_shape: tuple[int, int, int]) -> None:
        self.path = path
        self.f = h5py.File(path, "w")
        C, H, W = image_shape
        chunk = (min(64, n_total), C, H, W)
        self.images = self.f.create_dataset(
            "images", shape=(n_total, C, H, W), dtype="uint8",
            chunks=chunk, compression="lzf",
        )
        self.labels = self.f.create_dataset(
            "labels", shape=(n_total,), dtype="int32",
            chunks=(min(1024, n_total),),
        )
        self._ptr = 0

    def write_batch(self, adv_imgs: torch.Tensor, labels: torch.Tensor) -> None:
        """Write one batch. adv_imgs ∈ [0,1] float32, labels int."""
        B = adv_imgs.shape[0]
        end = self._ptr + B
        uint8 = (adv_imgs.cpu().float().clamp(0, 1) * 255).byte().numpy()
        self.images[self._ptr:end] = uint8
        self.labels[self._ptr:end] = labels.cpu().numpy().astype(np.int32)
        self._ptr = end

    def close(self) -> None:
        self.f.close()

    def __enter__(self) -> AdvH5Writer:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


class AdvH5Dataset(Dataset):
    """Read adversarial HDF5 and serve (float32 [0,1] tensor, int label) pairs.

    The h5py handle is opened lazily, the first time an item is fetched inside
    a given process. This makes the dataset fork-safe: each DataLoader worker
    opens its own handle after the fork, so ``num_workers > 0`` is safe.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.f: h5py.File | None = None
        self._images = None
        self._labels = None
        # Read length once up front without holding a handle open across fork.
        with h5py.File(path, "r") as f:
            self._len = len(f["labels"])

    def _ensure_open(self) -> None:
        if self.f is None:
            self.f = h5py.File(self.path, "r")
            self._images = self.f["images"]  # uint8
            self._labels = self.f["labels"]  # int32

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        self._ensure_open()
        img = torch.from_numpy(self._images[idx].astype(np.float32)) / 255.0
        lbl = int(self._labels[idx])
        return img, lbl

    def close(self) -> None:
        if self.f is not None:
            self.f.close()
            self.f = None


def adv_loader_from_h5(
    path: Path,
    batch_size: int = 256,
    num_workers: int = 0,  # AdvH5Dataset is fork-safe → >0 OK for read-only use
) -> DataLoader:
    ds = AdvH5Dataset(path)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True,
                      persistent_workers=(num_workers > 0))


# ---------------------------------------------------------------------------
# HDF5 feature storage
# ---------------------------------------------------------------------------
# Layout:
#   features_<model>_<split>.h5
#     filter_means  (N, C)  float16 – per-filter spatial mean |activation|
#     filter_maxs   (N, C)  float16 – per-filter spatial max |activation|
#     filter_l2     (N, C)  float16 – per-filter spatial RMS  activation (energy)
#     inf_norms     (N,)    float32 – per-sample ‖feat‖_∞ (Viyog statistic)
#     logits        (N, K)  float16 – classifier logits (energy / MSP / margin)
#     labels        (N,)    int32   – class labels (-1 if unavailable)
#
# filter_l2 and logits enable the energy/sparsity/logit signature families in
# step 08 at zero extra GPU cost (the forward pass already produces logits;
# the first-layer activation is already in VRAM for the mean/max reduction).

class FeatureH5Writer:
    """Streaming HDF5 writer for first-layer activation statistics + logits."""

    def __init__(
        self, path: Path, n_total: int, n_filters: int, n_classes: int | None = None
    ) -> None:
        self.f = h5py.File(path, "w")
        chunk_N = min(256, n_total)
        self.fmeans = self.f.create_dataset(
            "filter_means", shape=(n_total, n_filters), dtype="float16",
            chunks=(chunk_N, n_filters), compression="lzf",
        )
        self.fmaxs = self.f.create_dataset(
            "filter_maxs", shape=(n_total, n_filters), dtype="float16",
            chunks=(chunk_N, n_filters), compression="lzf",
        )
        self.fl2 = self.f.create_dataset(
            "filter_l2", shape=(n_total, n_filters), dtype="float16",
            chunks=(chunk_N, n_filters), compression="lzf",
        )
        self.norms = self.f.create_dataset(
            "inf_norms", shape=(n_total,), dtype="float32",
            chunks=(min(1024, n_total),),
        )
        self.logits = None
        if n_classes is not None:
            self.logits = self.f.create_dataset(
                "logits", shape=(n_total, n_classes), dtype="float16",
                chunks=(chunk_N, n_classes), compression="lzf",
            )
        self.labels = self.f.create_dataset(
            "labels", shape=(n_total,), dtype="int32",
            chunks=(min(1024, n_total),),
        )
        self._ptr = 0

    def write_batch(
        self,
        feats: torch.Tensor,       # (B, C, H, W) or (B, C)
        labels: torch.Tensor,
        logits: torch.Tensor | None = None,  # (B, K)
    ) -> None:
        B = feats.shape[0]
        end = self._ptr + B
        flat = feats.float().cpu()
        if flat.ndim == 4:
            # Spatial mean / max / RMS of |activation| over H, W
            abs_flat = flat.abs()
            fm = abs_flat.mean(dim=(2, 3))                 # (B, C)
            fx = abs_flat.amax(dim=(2, 3))                 # (B, C)
            fl2 = flat.pow(2).mean(dim=(2, 3)).sqrt()      # (B, C) per-filter RMS
        else:
            fm = flat.abs()
            fx = flat.abs()
            fl2 = flat.abs()

        # ‖feat‖_∞ per sample
        norms = flat.reshape(B, -1).abs().amax(dim=1)  # (B,)

        self.fmeans[self._ptr:end] = fm.numpy().astype(np.float16)
        self.fmaxs[self._ptr:end] = fx.numpy().astype(np.float16)
        self.fl2[self._ptr:end] = fl2.numpy().astype(np.float16)
        self.norms[self._ptr:end] = norms.numpy().astype(np.float32)
        if self.logits is not None and logits is not None:
            self.logits[self._ptr:end] = logits.float().cpu().numpy().astype(np.float16)
        self.labels[self._ptr:end] = labels.cpu().numpy().astype(np.int32)
        self._ptr = end

    def close(self) -> None:
        self.f.close()

    def __enter__(self) -> FeatureH5Writer:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def load_feature_h5(path: Path) -> dict[str, np.ndarray]:
    """Load all arrays from a feature HDF5 into a dict of numpy arrays."""
    with h5py.File(path, "r") as f:
        return {k: f[k][:] for k in f}


# ---------------------------------------------------------------------------
# Rich (full) feature storage — for the one-pass extractor (06b)
# ---------------------------------------------------------------------------
# Stores every per-filter statistic any signature needs, computed from the full
# (C,H,W) map before it is discarded. Names of the first three match the legacy
# layout so the existing step-08 battery reads these files unchanged.
#
# Per-filter (N, C) float16:  filter_means, filter_maxs, filter_l2,
#                             filter_std, filter_tv, filter_hf
# Per-image  (N,)   float32:  inf_norms, gram_offdiag
# Plus logits (N, K) float16 and labels (N,) int32.

FULL_PER_FILTER = ["filter_means", "filter_maxs", "filter_l2",
                   "filter_std", "filter_tv", "filter_hf"]
FULL_PER_IMAGE = ["inf_norms", "gram_offdiag"]


class FullFeatureH5Writer:
    """Streaming writer for the rich first-layer statistic battery."""

    def __init__(self, path: Path, n_total: int, n_filters: int,
                 n_classes: int | None = None) -> None:
        self.f = h5py.File(path, "w")
        cN = min(256, n_total)
        self._pf = {}
        for name in FULL_PER_FILTER:
            self._pf[name] = self.f.create_dataset(
                name, shape=(n_total, n_filters), dtype="float16",
                chunks=(cN, n_filters), compression="lzf")
        self._pi = {}
        for name in FULL_PER_IMAGE:
            self._pi[name] = self.f.create_dataset(
                name, shape=(n_total,), dtype="float32",
                chunks=(min(1024, n_total),))
        self.logits = None
        if n_classes is not None:
            self.logits = self.f.create_dataset(
                "logits", shape=(n_total, n_classes), dtype="float16",
                chunks=(cN, n_classes), compression="lzf")
        self.labels = self.f.create_dataset(
            "labels", shape=(n_total,), dtype="int32", chunks=(min(1024, n_total),))
        self._ptr = 0

    def write_batch(self, per_filter: dict[str, torch.Tensor],
                    per_image: dict[str, torch.Tensor],
                    labels: torch.Tensor,
                    logits: torch.Tensor | None = None) -> None:
        B = labels.shape[0]
        end = self._ptr + B
        for name, ds in self._pf.items():
            ds[self._ptr:end] = per_filter[name].detach().cpu().numpy().astype(np.float16)
        for name, ds in self._pi.items():
            ds[self._ptr:end] = per_image[name].detach().cpu().numpy().astype(np.float32)
        if self.logits is not None and logits is not None:
            self.logits[self._ptr:end] = logits.detach().float().cpu().numpy().astype(np.float16)
        self.labels[self._ptr:end] = labels.detach().cpu().numpy().astype(np.int32)
        self._ptr = end

    def close(self) -> None:
        self.f.close()

    def __enter__(self) -> "FullFeatureH5Writer":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
