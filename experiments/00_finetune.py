"""Step 0 – Finetune ImageNet-pretrained timm backbones on an ID dataset.

Produces the per-dataset checkpoints the rest of the pipeline consumes:
    weights/<dataset>/<model>.pth          (CIFAR-100 keeps its legacy names)

Each checkpoint is saved in the HuggingFace amanyagami/Cifar100_Finetuned
format ({"model_state", "val_acc", "epoch", "model_name"}) so model_utils
loads it unchanged.

Recipe (tuned for near-SOTA, per config.FINETUNE_CFG):
  - timm backbone, pretrained=True, head reset to the dataset's num_classes
  - trained INSIDE a NormalizedModel wrapper on [0,1] pixels (identical input
    convention to attack generation + feature extraction)
  - AdamW, discriminative LR (head = head_lr_mult × backbone lr), cosine
    schedule with linear warmup, weight decay, AMP (fp16)
  - label smoothing; optional Mixup/CutMix for natural datasets (off for digits)
  - optional EMA of weights; best-by-test-accuracy checkpoint is kept
  - horizontal flip / RandAugment disabled for digit/sign datasets

Usage:
    CUDA_VISIBLE_DEVICES=0 python experiments/00_finetune.py --dataset cifar10
    CUDA_VISIBLE_DEVICES=1 python experiments/00_finetune.py --dataset gtsrb --models vit_base swin_tiny
    python experiments/00_finetune.py --dataset svhn --epochs 20 --force

Skips a (dataset, model) whose checkpoint already exists unless --force.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time

import config
import timm
import torch
import torch.nn as nn
from config import DEVICE, MODEL_ARCHS, finetune_cfg
from data_utils import get_id_eval_loader, get_id_train_loader
from model_utils import NormalizedModel

# Per-model finetune batch size (224×224, fwd+bwd, AMP). Override with --batch.
FT_BATCH: dict[str, int] = {
    "convnextv2_base": 128,
    "swin_tiny":       256,
    "vit_base":        128,
    "efficientnetv2_l": 64,
    # Edge backbones — tiny, train with large batch for throughput.
    "mobilenetv3_l":   512,
    "effnet_lite0":    512,
    "mobileone_s1":    512,
    "fastvit_sa12":    256,
    # Edge backbones added for multi-dataset parity (were finetuned via 00b for
    # cifar100; these recipes let run_matrix's 00_finetune chain build them too).
    "mobilenetv4_m":   384,
    "efficientvit_b1": 256,
    "edgenext_small":  384,
    # ResNet / DenseNet families.
    "resnet18":        256,
    "resnet34":        256,
    "resnet50":        192,
    "resnet101":       128,
    "resnet152":       96,
    "densenet121":     128,
    "densenet161":     96,
    "densenet169":     96,
    "densenet201":     64,
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Finetune timm backbones on an ID dataset")
    p.add_argument("--dataset", required=True, choices=list(config.DATASET_SPECS),
                   help="ID dataset to finetune on")
    p.add_argument("--models", nargs="+", default=list(MODEL_ARCHS),
                   choices=list(MODEL_ARCHS), help="Which backbones (default: all)")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override the per-dataset epoch count")
    p.add_argument("--batch", type=int, default=None,
                   help="Override the per-model batch size")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0, help="RNG seed (reproducibility)")
    p.add_argument("--force", action="store_true", help="Re-finetune even if the checkpoint exists")
    return p.parse_args()


def _seed_everything(seed: int) -> None:
    """Seed all RNGs for a reproducible finetune (logged in the checkpoint).
    AMP + multi-worker augmentation leave minor residual nondeterminism."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_param_groups(backbone: nn.Module, lr: float, head_mult: float, wd: float):
    """Discriminative LR: classifier head gets head_mult × the backbone lr."""
    head = backbone.get_classifier()
    head_ids = {id(p) for p in head.parameters()}
    base = [p for p in backbone.parameters() if id(p) not in head_ids and p.requires_grad]
    headp = [p for p in backbone.parameters() if id(p) in head_ids and p.requires_grad]
    return [
        {"params": base, "lr": lr, "weight_decay": wd},
        {"params": headp, "lr": lr * head_mult, "weight_decay": wd},
    ]


def _make_scheduler(opt, total_steps: int, warmup_steps: int):
    """Linear warmup → cosine decay to ~1% of base, as a per-step LambdaLR."""
    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))
    return torch.optim.lr_scheduler.LambdaLR(opt, fn)


class _EMA:
    """Minimal exponential moving average of a module's params + buffers."""

    def __init__(self, model: nn.Module, decay: float = 0.9998) -> None:
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for s, m in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(d).add_(m, alpha=1 - d)
        for s, m in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(m)


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: str) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return 100.0 * correct / max(1, total)


def finetune_one(dataset: str, model_name: str, args: argparse.Namespace) -> dict:
    cfg = finetune_cfg(dataset)
    epochs = args.epochs if args.epochs is not None else cfg["epochs"]
    batch = args.batch if args.batch is not None else FT_BATCH[model_name]
    arch = MODEL_ARCHS[model_name]
    num_classes = config.DATASET_SPECS[dataset]["num_classes"]
    out_path = config.weight_path(dataset, model_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not args.force:
        print(f"  [skip] {out_path} exists (use --force to retrain)")
        return {"skipped": True, "path": str(out_path)}

    print(f"\n  === finetune {model_name} ({arch}) on {dataset} "
          f"[{num_classes} cls, {epochs} ep, batch {batch}] ===")
    backbone = timm.create_model(arch, pretrained=True, num_classes=num_classes)
    model = NormalizedModel(backbone).to(DEVICE).train()

    train_loader = get_id_train_loader(dataset, batch_size=batch, num_workers=args.workers,
                                       aug=cfg.get("randaugment", False))
    test_loader = get_id_eval_loader(dataset, batch_size=max(256, batch), num_workers=args.workers)

    opt = torch.optim.AdamW(
        _build_param_groups(backbone, cfg["lr"], cfg["head_lr_mult"], cfg["weight_decay"]))
    steps_per_epoch = len(train_loader)
    sched = _make_scheduler(opt, epochs * steps_per_epoch, cfg["warmup_epochs"] * steps_per_epoch)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["amp"])

    # Loss + optional mixup (graceful fallback if timm.data.Mixup is unavailable).
    mixup_fn = None
    if cfg["mixup"] > 0:
        try:
            from timm.data import Mixup
            from timm.loss import SoftTargetCrossEntropy
            mixup_fn = Mixup(mixup_alpha=cfg["mixup"], cutmix_alpha=1.0, prob=1.0,
                             switch_prob=0.5, label_smoothing=cfg["label_smoothing"],
                             num_classes=num_classes)
            criterion = SoftTargetCrossEntropy()
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] mixup unavailable ({e}); using label-smoothing CE")
            criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])

    # Step-aware EMA decay: ~2-epoch half-life (fixed 0.9998 lags badly on short
    # finetunes → underreports accuracy and was selecting a near-init shadow).
    ema_decay = max(0.99, min(0.9999, 1.0 - math.log(2) / (2.0 * max(1, steps_per_epoch))))
    ema = _EMA(model, decay=ema_decay) if cfg["ema"] else None
    best_acc, best_state, best_epoch = -1.0, None, -1
    history = []

    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            if mixup_fn is not None:
                x, y = mixup_fn(x, y)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=cfg["amp"]):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            if cfg["grad_clip"]:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(opt)
            scaler.update()
            sched.step()
            if ema is not None:
                ema.update(model)
            running += loss.item()

        # Evaluate BOTH the raw model and the EMA, keep whichever is better.
        # (A high EMA decay lags badly on short finetunes — the raw model is
        # often better early; selecting the max avoids the EMA-lag underfit.)
        acc, src = evaluate(model, test_loader, DEVICE), model
        if ema is not None:
            acc_ema = evaluate(ema.shadow, test_loader, DEVICE)
            if acc_ema > acc:
                acc, src = acc_ema, ema.shadow
        history.append({"epoch": epoch, "loss": running / max(1, steps_per_epoch),
                        "test_acc": acc, "lr": sched.get_last_lr()[0]})
        print(f"    epoch {epoch + 1}/{epochs}  loss={running / steps_per_epoch:.3f}  "
              f"test_acc={acc:.2f}%  ({time.time() - t0:.0f}s)")
        if acc > best_acc:
            best_acc, best_epoch = acc, epoch
            # Save the BACKBONE state dict (model_utils loads into the bare timm model).
            best_state = {k: v.detach().cpu().clone() for k, v in src.model.state_dict().items()}

    torch.save({"model_state": best_state, "val_acc": best_acc,
                "epoch": best_epoch, "model_name": arch, "seed": args.seed,
                "norm": "imagenet"}, out_path)
    print(f"  ✓ saved {out_path}  best_test_acc={best_acc:.2f}% @epoch {best_epoch + 1}")
    return {"skipped": False, "best_acc": best_acc, "epoch": best_epoch,
            "epochs": epochs, "path": str(out_path), "history": history}


def main() -> None:
    args = _parse_args()
    _seed_everything(args.seed)
    config.set_dataset(args.dataset)  # creates weights/<dataset>/ etc.
    print(f"=== Step 0: finetune on {args.dataset} → {args.models}  (seed={args.seed}) ===")

    log_path = config.ANALYSIS_DIR / "finetune_log.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {}
    for model_name in args.models:
        res = finetune_one(args.dataset, model_name, args)
        log[model_name] = res
        log_path.write_text(json.dumps(log, indent=2))

    print(f"\n  finetune log → {log_path}")
    for m, r in log.items():
        if not r.get("skipped"):
            print(f"    {m}: best_test_acc={r.get('best_acc', '?')}")


if __name__ == "__main__":
    main()
