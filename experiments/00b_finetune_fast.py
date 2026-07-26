"""Step 0b – FAST GPU-resident CIFAR-100 finetune (no CPU dataloader).

Why this exists: 00_finetune.py is CPU-dataloader-bound (32->224 upscale ×
RandAugment × mixup across many workers) and its mixup(prob=1.0) recipe needs
~100 epochs — on 20 epochs the edge models reached only 9-16% acc. This script:

  * loads the whole CIFAR-100 train/test set ONCE, upscales 32->224 to a uint8
    tensor held RESIDENT ON THE GPU (~7.5 GB train) — zero per-epoch CPU work,
    so training is GPU-bound and big batches actually help (maximal-VRAM design);
  * augments on-GPU (random crop + hflip), no dataloader, no workers;
  * uses a plain label-smoothing CE recipe (NO mixup) → converges in ~25 epochs;
  * trains inside NormalizedModel on [0,1] and saves the legacy checkpoint format
    so model_utils / steps 03-08 load it unchanged.

Usage:
    CUDA_VISIBLE_DEVICES=1 python experiments/00b_finetune_fast.py --models mobilenetv3_l effnet_lite0
"""

from __future__ import annotations

import argparse
import math
import time

import config
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from config import DEVICE
from model_utils import NormalizedModel

CIFAR_ROOT = "/mnt/data1/asing725/viyog/data/cifar"
IMG = 224
FT_BATCH = {"mobilenetv3_l": 1024, "effnet_lite0": 1024, "mobileone_s1": 1024, "fastvit_sa12": 512,
            "mobilenetv4_m": 1024, "efficientvit_b1": 1024, "edgenext_small": 1024}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fast GPU-resident CIFAR-100 finetune")
    p.add_argument("--models", nargs="+", default=["mobilenetv3_l", "effnet_lite0",
                   "mobileone_s1", "fastvit_sa12"], choices=list(config.MODEL_ARCHS))
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


@torch.no_grad()
def load_gpu_dataset(train: bool, dev: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (uint8 [N,3,224,224] on GPU, int64 labels on GPU)."""
    ds = tv.datasets.CIFAR100(CIFAR_ROOT, train=train, download=True)
    x = torch.from_numpy(ds.data).permute(0, 3, 1, 2).contiguous()  # N,3,32,32 uint8
    y = torch.tensor(ds.targets, dtype=torch.long)
    out = torch.empty((x.shape[0], 3, IMG, IMG), dtype=torch.uint8, device=dev)
    for i in range(0, x.shape[0], 4096):  # chunked resize to bound transient VRAM
        c = x[i:i + 4096].to(dev, non_blocking=True).float()
        c = F.interpolate(c, size=IMG, mode="bilinear", align_corners=False)
        out[i:i + 4096] = c.round_().clamp_(0, 255).to(torch.uint8)
    return out, y.to(dev)


def _augment(batch_u8: torch.Tensor) -> torch.Tensor:
    """On-GPU: uint8->[0,1] float, random hflip + 4px-equiv reflect-pad crop."""
    x = batch_u8.float().div_(255.0)
    if torch.rand(1, device=x.device) < 0.5:
        x = torch.flip(x, dims=[3])
    pad = 28  # ~4px at 32 scaled to 224
    x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    i = int(torch.randint(0, 2 * pad + 1, (1,)).item())
    j = int(torch.randint(0, 2 * pad + 1, (1,)).item())
    return x[:, :, i:i + IMG, j:j + IMG]


@torch.no_grad()
def evaluate(model: nn.Module, xte: torch.Tensor, yte: torch.Tensor, bs: int) -> float:
    model.eval()
    correct = 0
    for i in range(0, xte.shape[0], bs):
        x = xte[i:i + bs].float().div_(255.0)
        with torch.autocast("cuda", dtype=torch.float16):
            pred = model(x).argmax(1)
        correct += (pred == yte[i:i + bs]).sum().item()
    return 100.0 * correct / xte.shape[0]


def finetune_one(model_name: str, args: argparse.Namespace,
                 xtr, ytr, xte, yte) -> dict:
    arch = config.MODEL_ARCHS[model_name]
    out_path = config.weight_path("cifar100", model_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not args.force:
        print(f"  [skip] {out_path}")
        return {"skipped": True}
    bs = args.batch or FT_BATCH[model_name]
    print(f"\n  === {model_name} ({arch}) | {args.epochs} ep | batch {bs} ===", flush=True)

    backbone = timm.create_model(arch, pretrained=True, num_classes=100)
    model = NormalizedModel(backbone).to(DEVICE).train()
    head = backbone.get_classifier()
    hid = {id(p) for p in head.parameters()}
    opt = torch.optim.AdamW([
        {"params": [p for p in backbone.parameters() if id(p) not in hid], "lr": 2e-4},
        {"params": [p for p in head.parameters()], "lr": 2e-3},
    ], weight_decay=0.05)
    n = xtr.shape[0]
    steps = args.epochs * math.ceil(n / bs)
    warm = math.ceil(n / bs) * 2
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm
                  else 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    scaler = torch.amp.GradScaler("cuda")
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    best = -1.0
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); run = 0.0
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            x = _augment(xtr[idx]); y = ytr[idx]
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = crit(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            run += loss.item()
        acc = evaluate(model, xte, yte, max(512, bs))
        print(f"    ep {ep+1}/{args.epochs} loss={run/math.ceil(n/bs):.3f} acc={acc:.2f}% ({time.time()-t0:.0f}s)", flush=True)
        if acc > best:
            best = acc
            state = {k: v.detach().cpu().clone() for k, v in model.model.state_dict().items()}
    torch.save({"model_state": state, "val_acc": best, "epoch": args.epochs, "model_name": arch}, out_path)
    print(f"  ✓ saved {out_path} best_acc={best:.2f}%", flush=True)
    return {"best_acc": best}


def main() -> None:
    args = _parse_args()
    config.set_dataset("cifar100")
    print(f"=== Step 0b: fast GPU-resident finetune → {args.models} ===", flush=True)
    print("  loading CIFAR-100 onto GPU (224px uint8) ...", flush=True)
    xtr, ytr = load_gpu_dataset(True, DEVICE)
    xte, yte = load_gpu_dataset(False, DEVICE)
    print(f"  train {tuple(xtr.shape)} {xtr.element_size()*xtr.nelement()/1e9:.1f}GB | test {tuple(xte.shape)} resident on {DEVICE}", flush=True)
    for m in args.models:
        finetune_one(m, args, xtr, ytr, xte, yte)


if __name__ == "__main__":
    main()
