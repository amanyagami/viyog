"""Central configuration for the OOD/ADV evaluation pipeline.

All paths, model definitions, attack parameters, and OOD dataset specs
live here. Edit this file to change models, attacks, or paths.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Default = the repo root (parent of this experiments/ dir) so a fresh clone
# works with zero setup; set VIYOG_ROOT to override (e.g. to point at a
# separate large-disk location for data/weights/results).
ROOT = Path(os.environ.get("VIYOG_ROOT", str(Path(__file__).resolve().parent.parent)))
DATA_DIR = Path(os.environ.get("VIYOG_DATA", str(ROOT / "data")))
# Output roots accept an env override (default = canonical paths) so seed/experiment
# runs can be namespaced to a separate directory without clobbering canonical results.
# With no env vars set these are byte-identical to the previous hard-coded values.
WEIGHTS_BASE = Path(os.environ.get("VIYOG_WEIGHTS", str(ROOT / "weights")))  # IMMUTABLE base; WEIGHTS_DIR rebound by set_dataset
WEIGHTS_DIR = WEIGHTS_BASE
RESULTS_DIR = Path(os.environ.get("VIYOG_RESULTS", str(ROOT / "results")))
FEATURES_DIR = RESULTS_DIR / "features"
PLOTS_DIR = RESULTS_DIR / "plots"
ANALYSIS_DIR = RESULTS_DIR / "analysis"
ADV_DIR = Path(os.environ.get("VIYOG_ADV", str(DATA_DIR / "adversarial")))
OOD_ROOT = DATA_DIR / "ood"

for _d in [WEIGHTS_DIR, FEATURES_DIR, PLOTS_DIR, ANALYSIS_DIR, ADV_DIR, OOD_ROOT]:
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# GPU assignment (as of 2026-06-08 – verify with nvidia-smi before running):
#   GPU 5: 135 GB free → efficientnetv2_l  (alone, largest model)
#   GPU 0: 121 GB free → convnextv2_base   (alone)
#   GPU 4:  70 GB free → vit_base + swin_tiny (sequential)
#   GPU 6/7: ~5 GB free → occupied, do NOT use
# Set CUDA_VISIBLE_DEVICES before launching each script.
# Inside the script device is always "cuda:0".
# ---------------------------------------------------------------------------
GPU_ID = 5   # primary / default for serial steps
DEVICE = "cuda:0"

# ---------------------------------------------------------------------------
# Normalisation (ImageNet stats – used by all four timm-finetuned models)
# CIFAR-100 fine-tuning from ImageNet pretraining keeps these stats.
# ---------------------------------------------------------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = 224  # all models trained at 224×224

# --- Normalisation protocol (deliberate, documented) ----------------------
# ONE normalisation for every model: ImageNet mean/std, applied IDENTICALLY at
# finetune, clean-eval, attack-generation and feature-extraction (the
# NormalizedModel wrapper does it once, internally, on [0,1] inputs).
#
# Rationale for the paper: a single fixed input transform makes the first-layer
# activation comparison *comparable across all architectures* — the detector is
# evaluated on identical inputs, not arch-specific ones. A few timm archs
# (tf_efficientnet*, vit_base) were pretrained with mean=std=0.5; using ImageNet
# stats only shifts their finetune starting point and is fully absorbed during
# finetuning. It is NOT a correctness bug, because the SAME stats are used at
# every stage, so the model is self-consistent end-to-end. Documented as a
# design choice; per-arch stats are intentionally NOT used (they would make the
# cross-architecture comparison apples-to-oranges and would invalidate the
# already-reported efficientnet/vit results). See README_REBUTTAL "Normalisation".

# ---------------------------------------------------------------------------
# Models: {logical_name: (timm_arch_name, weight_file_or_None)}
# weight_file=None → use timm pretrained weights (not custom checkpoint)
# ---------------------------------------------------------------------------
MODELS: dict[str, tuple[str, Path | None]] = {
    "convnextv2_base": (
        "convnextv2_base",
        WEIGHTS_DIR / "convnextv2_base_cifar100.pth",
    ),
    "swin_tiny": (
        "swin_tiny_patch4_window7_224",
        WEIGHTS_DIR / "swin_tiny_patch4_window7_224_cifar100.pth",
    ),
    "efficientnetv2_l": (
        "tf_efficientnetv2_l",
        WEIGHTS_DIR / "tf_efficientnetv2_l_cifar100.pth",
    ),
    "vit_base": (
        "vit_base_patch16_224",
        WEIGHTS_DIR / "vit_b_cifar100.pth",
    ),
}
NUM_CLASSES = 100  # CIFAR-100

# Canonical HuggingFace repo for downloading weights: the full 20-architecture
# CIFAR-100 panel + cifar10/gtsrb, laid out exactly as weight_path() expects
# (4 legacy flat files at repo root, everything else namespaced
# <dataset>/<model>.pth) so a plain snapshot_download() needs no renaming.
# amanyagami/Cifar100_Finetuned is an older, incomplete (4-file) mirror kept
# only for provenance — not used by 01_download.py.
HF_REPO = "amanyagami/viyog-weights"
# The original 4 finetuned checkpoints, which keep legacy flat filenames at
# the repo root (see _CIFAR100_WEIGHTS below).
HF_LEGACY_WEIGHT_FILES = [
    "convnextv2_base_cifar100.pth",
    "swin_tiny_patch4_window7_224_cifar100.pth",
    "tf_efficientnetv2_l_cifar100.pth",
    "vit_b_cifar100.pth",
]
# Google Drive folder ID (backup source)
GDRIVE_FOLDER_ID = "1RRzfF7mMa424lbJzp_brA2gg8D4nmaD9"

# ---------------------------------------------------------------------------
# Adversarial attacks  (6: highest acceptance, ordered fastest → slowest)
# ---------------------------------------------------------------------------
# FGSM    – Goodfellow 2014.  Single step L∞.  Universal baseline.  ~2s/model.
# BIM     – Kurakin 2017.     Iterative FGSM L∞, 10 steps.          ~20s/model.
# PGD     – Madry 2018.       Gold-standard L∞, 20 steps+rand start. ~30s/model.
# APGD-CE – Croce 2020.       AutoAttack L∞, auto step-size 100 steps.~3min/model.
# DeepFool– Moosavi 2016.     Min-norm L2, per-sample iterative.     ~5min/model.
# CW      – Carlini 2017.     Optimization-based L2, 100 steps Adam. ~10min/model.
# Dropped: TPGD (TRADES-specific, redundant with PGD for general eval).
# All operate on raw [0,1] images fed to a NormalizedModel wrapper.
ATTACKS: dict[str, dict] = {
    "fgsm": {
        "cls": "FGSM",
        "kwargs": {"eps": 8 / 255},
    },
    "bim": {
        "cls": "BIM",
        "kwargs": {"eps": 8 / 255, "alpha": 2 / 255, "steps": 10},
    },
    "pgd": {
        "cls": "PGD",
        "kwargs": {"eps": 8 / 255, "alpha": 2 / 255, "steps": 20, "random_start": True},
    },
    "apgd_ce": {
        "cls": "APGD",
        "kwargs": {"eps": 8 / 255, "steps": 100, "loss": "ce"},
    },
    "deepfool": {
        "cls": "DeepFool",
        "kwargs": {"steps": 50, "overshoot": 0.02},
    },
    "cw": {
        "cls": "CW",
        "kwargs": {"c": 1e-3, "kappa": 0, "steps": 100, "lr": 0.01},
    },
}

# DeepFool and CW are CPU-bound (per-class gradient loop over 100 CIFAR classes)
# → ~23 min/batch, ~15 h/model at the full 10k test set.  We cap their sample
# count: 2000 samples give stable AUROC (separability) and ~±1.5% accuracy CI,
# cutting wall-clock from ~1.5 days to a few hours.  Fast attacks (FGSM / BIM /
# PGD / APGD) stay at the full test set.  Attacks not listed → full 10k.
ATTACK_MAX_SAMPLES: dict[str, int] = {
    "deepfool": 2000,
    "cw": 2000,
}

# ---------------------------------------------------------------------------
# OOD datasets
# kind: near_ood | far_ood | texture_ood
# tv_cls: torchvision dataset class name
# split_kwarg: keyword arg for selecting test split in that dataset
# ---------------------------------------------------------------------------
OOD_DATASETS: dict[str, dict] = {
    "cifar10": {
        "kind": "near_ood",
        "tv_cls": "CIFAR10",
        "split_kwarg": {"train": False},
        "note": "10-class subset of CIFAR source; coarse semantic overlap",
    },
    "svhn": {
        "kind": "far_ood",
        "tv_cls": "SVHN",
        "split_kwarg": {"split": "test"},
        "note": "Street-view digit photos; structural mismatch",
    },
    "stl10": {
        "kind": "near_ood",
        "tv_cls": "STL10",
        "split_kwarg": {"split": "test"},
        "note": "10 ImageNet-adjacent classes; semantic near-OOD",
    },
    "dtd": {
        "kind": "texture_ood",
        "tv_cls": "DTD",
        "split_kwarg": {"split": "test"},
        "note": "Describable Textures; no semantic objects",
    },
    "mnist": {
        "kind": "far_ood",
        "tv_cls": "MNIST",
        "split_kwarg": {"train": False},
        "note": "Grayscale handwritten digits; converted to 3-ch",
    },
    "fashionmnist": {
        "kind": "far_ood",
        "tv_cls": "FashionMNIST",
        "split_kwarg": {"train": False},
        "note": "Grayscale clothing silhouettes; converted to 3-ch",
    },
    "flowers102": {
        "kind": "near_ood",
        "tv_cls": "Flowers102",
        "split_kwarg": {"split": "test"},
        "note": "102 flower species; fine-grained natural images",
    },
    "food101": {
        "kind": "near_ood",
        "tv_cls": "Food101",
        "split_kwarg": {"split": "test"},
        "note": "101 food categories; fine-grained natural images",
    },
    "eurosat": {
        "kind": "far_ood",
        "tv_cls": "EuroSAT",
        "split_kwarg": {},
        "note": "Sentinel-2 satellite tiles; non-object overhead imagery",
    },
    "gtsrb": {
        "kind": "far_ood",
        "tv_cls": "GTSRB",
        "split_kwarg": {"split": "test"},
        "note": "German traffic signs; structured icons, distinct domain",
    },
}

# ---------------------------------------------------------------------------
# Per-model batch sizes sized for H200 126 GB VRAM.
#
# Adversarial attacks need gradients → all layer activations stay alive during
# the backward pass.  Peak VRAM per step ≈ model_weights + 2×activations.
# Conservative estimates (leaving ~30 % safety margin):
#   efficientnetv2_l  ~120 M params → ~50 GB peak at batch 256
#   convnextv2_base   ~89 M params  → ~50 GB peak at batch 512
#   vit_base          ~86 M params  → ~35 GB peak at batch 512
#   swin_tiny         ~29 M params  → ~35 GB peak at batch 1024
#
# The peak is identical for every attack (same graph per step); only iteration
# count differs.  Larger batches → fewer kernel launches → better H200 TCore
# utilisation (previous config left 60-80 % of compute idle at batch ≤ 64).
MODEL_ATTACK_BATCH: dict[str, int] = {
    # Measured per-image activation memory (224×224, fwd+bwd stored for grad):
    #   efficientnetv2_l ~433 MB/img → batch 256 = 111 GB on GPU 5 (135 GB free)
    #   convnextv2_base  ~232 MB/img → batch 128 =  30 GB on GPU 0 (122 GB free)
    #   vit_base         ~123 MB/img → batch 512 =  63 GB on GPU 4 ( 71 GB free)
    #   swin_tiny         ~40 MB/img → batch 512 =  20 GB on GPU 4 ( 71 GB free)
    "efficientnetv2_l": 48,    # relocated to idle GPU2 (25GB free): 48×433MB≈21GB fits CW;
                               # DeepFool is work-bound so smaller batch ≈ no time cost
    "convnextv2_base":  192,   # 128 starved GPU0 to 39% util; 192 (~45GB) fits 57GB free
    "vit_base":         256,   # was 512 → APGD OOM'd; GPU 4 now shares w/ a 70 GB neighbor
    "swin_tiny":        256,   # runs concurrently w/ vit_base on GPU 4 → keep small
    # Edge models (3–11 M params) — tiny graph, large batch fits trivially.
    "mobilenetv3_l":    512,
    "effnet_lite0":     512,
    "mobileone_s1":     512,
    "fastvit_sa12":     512,
    "mobilenetv4_m":    512,
    "efficientvit_b1":  512,
    "edgenext_small":   512,
    # ResNet / DenseNet families (paper baselines + variants).
    "resnet18":         512,
    "resnet34":         512,
    "resnet50":         256,
    "resnet101":        192,
    "resnet152":        128,
    "densenet121":      256,
    "densenet161":      128,
    "densenet169":      192,
    "densenet201":      128,
}

# Feature extraction uses torch.no_grad() → activations freed immediately after
# the hook fires.  Only current-batch features sit in VRAM → can go 4-8× larger.
MODEL_FEATURE_BATCH: dict[str, int] = {
    "efficientnetv2_l": 1024,
    "convnextv2_base":  2048,
    "vit_base":         2048,
    "swin_tiny":        1024,   # lowered from 4096 — OOM'd on shared GPU0
    "mobilenetv3_l":   4096,
    "effnet_lite0":    4096,
    "mobileone_s1":    2048,   # 4096 overflowed int32 in avg_pool2d (HF stat)
    "fastvit_sa12":    2048,
    "mobilenetv4_m":   4096,
    "efficientvit_b1": 4096,
    "edgenext_small":  4096,
    "resnet18":        2048,   # 4096*64*112*112 > INT_MAX → avg_pool2d overflow
    "resnet34":        2048,   # (same fix; resnet50 already uses 2048 safely)
    "resnet50":        2048,
    "resnet101":       1536,
    "resnet152":       1024,
    "densenet121":     1536,
    "densenet161":     1024,
    "densenet169":     1280,
    "densenet201":     1024,
}

# Batch size for forward-only evaluation (clean + adversarial accuracy).
EVAL_BATCH: int = 1024

# ---------------------------------------------------------------------------
# Feature extraction & analysis
# ---------------------------------------------------------------------------
N_FEATURE_SAMPLES = 5000   # samples per split for feature analysis
FEATURES_DTYPE = "float16"  # HDF5 storage precision

# Neuron grouping thresholds (by filter-level mean |activation| on ID data)
NEURON_LARGE_PCT = 0.10   # top 10% → "large neurons"
NEURON_LOW_PCT = 0.10     # bottom 10% → "low neurons"
# middle 80% → "middle neurons"

# ===========================================================================
# MULTI-DATASET EXTENSION
# ---------------------------------------------------------------------------
# The original pipeline is hard-wired to CIFAR-100 as the in-distribution (ID)
# dataset. To run the same OOD-vs-ADV experiment on other ID datasets we add a
# `dataset` axis. Each pipeline step takes `--dataset <name>` (default
# "cifar100"); `set_dataset(name)` below recomputes the dataset-varying globals
# (MODELS, NUM_CLASSES, FEATURES_DIR, ADV_DIR, ANALYSIS_DIR, PLOTS_DIR,
# OOD_DATASETS). CIFAR-100 keeps the legacy flat layout so existing artifacts
# are reused untouched; every other dataset is namespaced under
# results/<dataset>/, data/adversarial/<dataset>/, weights/<dataset>/.
# ===========================================================================

# Logical model name → timm architecture (decoupled from the weight file, which
# is now per-dataset: weights/<dataset>/<logical>.pth).
MODEL_ARCHS: dict[str, str] = {
    "convnextv2_base": "convnextv2_base",
    "swin_tiny":       "swin_tiny_patch4_window7_224",
    "efficientnetv2_l": "tf_efficientnetv2_l",
    "vit_base":        "vit_base_patch16_224",
    # Edge-deployed backbones (answer reviewers A-w5 / B-1 on embedded relevance).
    "mobilenetv3_l":   "mobilenetv3_large_100",   # canonical mobile CNN
    "effnet_lite0":    "tf_efficientnet_lite0",   # TFLite-targeted edge CNN
    "mobileone_s1":    "mobileone_s1",            # Apple 2023, sub-1ms mobile
    "fastvit_sa12":    "fastvit_sa12",            # Apple 2023, mobile hybrid-ViT
    # Embedded-SOTA additions (latest, deployed on phones/Jetson/edge accelerators).
    "mobilenetv4_m":   "mobilenetv4_conv_medium",  # Google 2024, newest MobileNet
    "efficientvit_b1": "efficientvit_b1",           # MIT 2023, hardware-efficient ViT
    "edgenext_small":  "edgenext_small",            # 2022, edge-designed ConvNeXt-lite
    # ResNet family (paper baseline + depth variants).
    "resnet18":        "resnet18",
    "resnet34":        "resnet34",
    "resnet50":        "resnet50",
    "resnet101":       "resnet101",
    "resnet152":       "resnet152",
    # DenseNet family (paper's strongest baseline + variants).
    "densenet121":     "densenet121",
    "densenet161":     "densenet161",
    "densenet169":     "densenet169",
    "densenet201":     "densenet201",
}

# The 8 datasets used as the OOD universe (unchanged from the original run).
# For a given ID dataset the OOD pool is this set minus the ID itself and any
# near-duplicate (see NEAR_DUPLICATES).
OOD_UNIVERSE: dict[str, dict] = dict(OOD_DATASETS)

# Datasets that overlap too much to be each other's OOD (e.g. CIFAR-10 draws
# from the same image source as CIFAR-100). Disabled by default so a fresh
# CIFAR-100 run reproduces the original OOD pool exactly (which included
# cifar10). Populate, e.g. [{"cifar10", "cifar100"}], to exclude near-dups.
NEAR_DUPLICATES: list[set[str]] = []

# Full spec for every dataset that can serve as an ID dataset. `tv_cls` is the
# torchvision class; `split_kwargs` selects train vs test; `gray` triggers the
# 3-channel broadcast; `num_classes` sets the classifier head. EuroSAT has no
# native split → we carve a deterministic train/test split (see data_utils).
DATASET_SPECS: dict[str, dict] = {
    "cifar100": dict(num_classes=100, tv_cls="CIFAR100", train_kw={"train": True},
                     test_kw={"train": False}, gray=False, hflip=True),
    "cifar10":  dict(num_classes=10,  tv_cls="CIFAR10",  train_kw={"train": True},
                     test_kw={"train": False}, gray=False, hflip=True),
    "svhn":     dict(num_classes=10,  tv_cls="SVHN",     train_kw={"split": "train"},
                     test_kw={"split": "test"}, gray=False, hflip=False),
    "gtsrb":    dict(num_classes=43,  tv_cls="GTSRB",    train_kw={"split": "train"},
                     test_kw={"split": "test"}, gray=False, hflip=False),
    "mnist":    dict(num_classes=10,  tv_cls="MNIST",    train_kw={"train": True},
                     test_kw={"train": False}, gray=True,  hflip=False),
    "fashionmnist": dict(num_classes=10, tv_cls="FashionMNIST", train_kw={"train": True},
                     test_kw={"train": False}, gray=True,  hflip=True),
    "eurosat":  dict(num_classes=10,  tv_cls="EuroSAT",  train_kw={"_split": "train"},
                     test_kw={"_split": "test"}, gray=False, hflip=True, needs_split=True),
    "pets":     dict(num_classes=37,  tv_cls="OxfordIIITPet", train_kw={"split": "trainval"},
                     test_kw={"split": "test"}, gray=False, hflip=True),
    "food101":  dict(num_classes=101, tv_cls="Food101",  train_kw={"split": "train"},
                     test_kw={"split": "test"}, gray=False, hflip=True),
    "flowers102": dict(num_classes=102, tv_cls="Flowers102", train_kw={"split": "train"},
                     test_kw={"split": "test"}, gray=False, hflip=True),
}

# Per-dataset finetuning recipe, tuned to reach near-SOTA for an ImageNet-
# pretrained backbone finetuned at 224×224. Digit/sign datasets converge fast
# and must NOT use horizontal flip or mixup. Fine-grained natural datasets get
# more epochs + mixup. lr is the backbone lr; the head uses 10× (see finetune).
# ONE simple, consistent finetune recipe for every (model, dataset):
#   AdamW, single LR, cosine schedule + 1-epoch warmup, label smoothing,
#   light standard augmentation (RandomResizedCrop + dataset-appropriate flip),
#   AMP, grad-clip, fixed seed. Deliberately NO mixup / cutmix / RandAugment /
#   EMA — those need long (100s-epoch) schedules and caused the ResNet underfit
#   at 20 epochs; removing them makes the recipe converge fast, be reproducible
#   (few stochastic parts), and identical across architectures. Only the epoch
#   count is tuned per dataset by difficulty. lr=5e-4 converges in ~30 epochs.
FINETUNE_DEFAULTS = dict(
    lr=5e-4, head_lr_mult=1.0, weight_decay=0.05, warmup_epochs=1,
    label_smoothing=0.1, mixup=0.0, randaugment=False, ema=False,
    amp=True, grad_clip=1.0, epochs=30,
)
# Only epochs (and trivially-easy MNIST's lr) differ per dataset — everything
# else is shared, for consistency.
FINETUNE_CFG: dict[str, dict] = {
    "mnist":    dict(epochs=8, lr=3e-4),
    "svhn":     dict(epochs=15),
    "gtsrb":    dict(epochs=20),
    # cifar100, cifar10, eurosat, pets, food101, flowers102 → 30-epoch default
}


def finetune_cfg(dataset: str) -> dict:
    """Merged finetune hyper-parameters for `dataset` (defaults + overrides)."""
    cfg = dict(FINETUNE_DEFAULTS)
    cfg.update(FINETUNE_CFG.get(dataset, {}))
    cfg["hflip"] = DATASET_SPECS[dataset]["hflip"]
    return cfg


def ood_pool(dataset: str) -> list[str]:
    """OOD dataset names for a given ID dataset: the OOD universe minus the ID
    itself and any near-duplicate."""
    dups = set()
    for grp in NEAR_DUPLICATES:
        if dataset in grp:
            dups |= grp
    return [d for d in OOD_UNIVERSE if d != dataset and d not in dups]


# Legacy CIFAR-100 weight files (HuggingFace download names) — kept so the
# original run's artifacts resolve unchanged.
_CIFAR100_WEIGHTS: dict[str, Path] = {
    "convnextv2_base": WEIGHTS_DIR / "convnextv2_base_cifar100.pth",
    "swin_tiny":       WEIGHTS_DIR / "swin_tiny_patch4_window7_224_cifar100.pth",
    "efficientnetv2_l": WEIGHTS_DIR / "tf_efficientnetv2_l_cifar100.pth",
    "vit_base":        WEIGHTS_DIR / "vit_b_cifar100.pth",
}

# The currently-selected ID dataset (mutated by set_dataset). Steps that load
# the ID split read this so they don't need the name threaded everywhere.
CURRENT_DATASET = "cifar100"


def weight_path(dataset: str, model: str) -> Path:
    """Checkpoint path for (dataset, model). CIFAR-100 uses the legacy flat
    filenames; every other dataset uses weights/<dataset>/<model>.pth."""
    if dataset == "cifar100":
        # Original 4 keep their legacy flat names; newer models (e.g. edge
        # backbones) namespace under weights/cifar100/<model>.pth.
        if model in _CIFAR100_WEIGHTS:
            return _CIFAR100_WEIGHTS[model]
        return WEIGHTS_BASE / "cifar100" / f"{model}.pth"
    return WEIGHTS_BASE / dataset / f"{model}.pth"


def dataset_dirs(dataset: str) -> dict[str, Path]:
    """Return {weights, adv, features, analysis, plots} dirs for a dataset.
    CIFAR-100 → legacy flat layout; others → namespaced under <dataset>/."""
    if dataset == "cifar100":
        return dict(weights=WEIGHTS_BASE, adv=ADV_DIR, features=FEATURES_DIR,
                    analysis=ANALYSIS_DIR, plots=PLOTS_DIR)
    return dict(
        weights=WEIGHTS_BASE / dataset,
        adv=DATA_DIR / "adversarial" / dataset,
        features=RESULTS_DIR / dataset / "features",
        analysis=RESULTS_DIR / dataset / "analysis",
        plots=RESULTS_DIR / dataset / "plots",
    )


def models_for(dataset: str) -> dict[str, tuple[str, Path]]:
    """{logical_name: (timm_arch, weight_path)} for a dataset — same shape as
    the legacy MODELS dict."""
    return {m: (MODEL_ARCHS[m], weight_path(dataset, m)) for m in MODEL_ARCHS}


def set_dataset(dataset: str) -> None:
    """Point the dataset-varying module globals at `dataset`.

    Pipeline scripts call this at the top of main() and then rebind their own
    module-level names (via `global`) so their per-model helper functions —
    which reference these names — operate on the selected dataset. Creates the
    output directories. CIFAR-100 reproduces the original flat layout exactly.
    """
    global CURRENT_DATASET, NUM_CLASSES, MODELS, OOD_DATASETS
    global WEIGHTS_DIR, ADV_DIR, FEATURES_DIR, ANALYSIS_DIR, PLOTS_DIR
    if dataset not in DATASET_SPECS:
        raise ValueError(f"Unknown dataset '{dataset}'. Known: {list(DATASET_SPECS)}")
    CURRENT_DATASET = dataset
    NUM_CLASSES = DATASET_SPECS[dataset]["num_classes"]
    MODELS = models_for(dataset)
    OOD_DATASETS = {n: OOD_UNIVERSE[n] for n in ood_pool(dataset)}
    d = dataset_dirs(dataset)
    WEIGHTS_DIR, ADV_DIR, FEATURES_DIR = d["weights"], d["adv"], d["features"]
    ANALYSIS_DIR, PLOTS_DIR = d["analysis"], d["plots"]
    for p in (WEIGHTS_DIR, ADV_DIR, FEATURES_DIR, ANALYSIS_DIR, PLOTS_DIR):
        p.mkdir(parents=True, exist_ok=True)


# --- Near-SOTA acceptance gate --------------------------------------------
# A model is INCLUDED in the detection experiments + rebuttal reporting only if
# its finetuned clean top-1 reaches the per-dataset floor below. Under-trained
# models are dropped: they would confound the OOD/ADV separation results and
# weaken paper credibility. Floors are conservative near-SOTA values for an
# ImageNet-pretrained backbone finetuned at 224×224.
ACC_FLOOR: dict[str, float] = {
    "cifar100": 78.0,
    "cifar10":  94.0,
    "svhn":     93.0,
    "gtsrb":    96.0,
    "mnist":    99.0,
}


def accepted_models(dataset: str, models: list[str] | None = None) -> tuple[list[str], list[tuple[str, float]]]:
    """Return (accepted, dropped) where accepted = models with clean top-1 ≥
    ACC_FLOOR[dataset] (read from <analysis>/clean_accuracy.json), and dropped =
    [(model, acc)] below the floor or with unknown accuracy (acc=-1)."""
    import json
    models = models or list(MODEL_ARCHS)
    floor = ACC_FLOOR.get(dataset, 0.0)
    accs = {}
    p = dataset_dirs(dataset)["analysis"] / "clean_accuracy.json"
    if p.exists():
        accs = json.loads(p.read_text())
    accepted, dropped = [], []
    for m in models:
        a = accs.get(m, {}).get("top1")
        if a is not None and a >= floor:
            accepted.append(m)
        else:
            dropped.append((m, a if a is not None else -1.0))
    return accepted, dropped
