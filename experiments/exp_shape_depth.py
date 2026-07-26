"""Shape-vs-raw separation by layer depth (closes D-w2 / D2 adaptive-layer).

Reviewer D asked whether a *deeper* layer would beat the deployed first conv, and
whether an adaptive layer selector is needed. This measures, at several depths,
BOTH the raw L-inf statistic AND the deployed dormant-band roughness (shape)
statistic V, so we can show that:

  (a) the first-conv SHAPE read is at/near its best across depths (the deployed
      choice is justified beyond cost), while the raw L-inf read is the *weakest*
      at the first conv and only recovers mid-depth; and
  (b) a non-learned cross-attack layer selector on the raw norm reaches mid-depth
      but still sits below the first-conv shape read.

Per-channel dorm band is selected from an ID *train* split (no leakage); V is then
read on the held-out ID test / OOD / adv splits.

    CUDA_VISIBLE_DEVICES=4 python experiments/exp_shape_depth.py --dataset cifar100 \
        --models resnet50 densenet121 mobilenetv3_l convnextv2_base swin_tiny vit_base \
        --attacks pgd apgd_ce --n 2000 --n-layers 7
"""
from __future__ import annotations

import argparse

import config
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from config import DEVICE
from data_utils import adv_loader_from_h5, get_id_loader, get_ood_loader
from layer_ablation import MultiHook, auroc_dl, pick_layers
from model_utils import load_normalized_model

EPS = 1e-6
KAPPA = 1e-4
DORM_P = 0.10  # bottom 10% of alive channels


@torch.no_grad()
def _stats(a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (mean_abs, shape, linf) for a (B,C,H,W) activation.

    shape_c = TV(a_c) / (mean|a_c| + eps), TV = 1/2(mean|d_h| + mean|d_w|).
    """
    if a.dim() == 3:  # (B, N, D) token layout -> treat as (B, D, N, 1)
        a = a.transpose(1, 2).unsqueeze(-1)
    mean_abs = a.abs().mean(dim=(2, 3))  # (B,C)
    dh = (a[:, :, 1:, :] - a[:, :, :-1, :]).abs().mean(dim=(2, 3)) if a.shape[2] > 1 else torch.zeros_like(mean_abs)
    dw = (a[:, :, :, 1:] - a[:, :, :, :-1]).abs().mean(dim=(2, 3)) if a.shape[3] > 1 else torch.zeros_like(mean_abs)
    tv = 0.5 * (dh + dw)
    shape = tv / (mean_abs + EPS)
    linf = a.abs().flatten(1).amax(1)  # (B,)
    return mean_abs, shape, linf


@torch.no_grad()
def profile(model, hook, loader, names, n):
    """Per-channel E_ID[mean|a_c|] per layer (for dorm-band selection)."""
    acc = {nm: 0.0 for nm in names}
    cnt = {nm: 0 for nm in names}
    seen = 0
    for batch in loader:
        x = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(DEVICE)
        model(x)
        for nm in names:
            ma, _, _ = _stats(hook.feats[nm])
            acc[nm] = acc[nm] + ma.sum(0).cpu().numpy()
            cnt[nm] += ma.shape[0]
        seen += x.shape[0]
        if seen >= n:
            break
    return {nm: acc[nm] / max(cnt[nm], 1) for nm in names}


def dorm_idx(prof: np.ndarray) -> np.ndarray:
    """Bottom-DORM_P of *alive* channels by ID mean activation."""
    alive = np.where(prof > KAPPA)[0]
    if len(alive) == 0:
        alive = np.arange(len(prof))
    k = max(1, int(round(DORM_P * len(alive))))
    order = alive[np.argsort(prof[alive])]
    return order[:k]


@torch.no_grad()
def read(model, hook, loader, names, dorm, n):
    """Per-sample V (dorm-band shape) and L-inf for each layer."""
    accV = {nm: [] for nm in names}
    accL = {nm: [] for nm in names}
    seen = 0
    for batch in loader:
        x = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(DEVICE)
        model(x)
        for nm in names:
            _, shape, linf = _stats(hook.feats[nm])
            idx = torch.as_tensor(dorm[nm], device=shape.device)
            accV[nm].append(shape.index_select(1, idx).mean(1).cpu().numpy())
            accL[nm].append(linf.cpu().numpy())
        seen += x.shape[0]
        if seen >= n:
            break
    return ({nm: np.concatenate(v)[:n] for nm, v in accV.items()},
            {nm: np.concatenate(v)[:n] for nm, v in accL.items()})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar100")
    ap.add_argument("--models", nargs="+",
                    default=["resnet50", "densenet121", "mobilenetv3_l",
                             "convnextv2_base", "swin_tiny", "vit_base"])
    ap.add_argument("--attacks", nargs="+", default=["pgd", "apgd_ce"])
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--n-layers", type=int, default=7)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    config.set_dataset(args.dataset)

    _kind = {k: (v.get("kind") if isinstance(v, dict) else v) for k, v in config.OOD_UNIVERSE.items()}
    far = [o for o in config.OOD_DATASETS if _kind.get(o) == "far_ood"][:2]
    near = [o for o in config.OOD_DATASETS if _kind.get(o) == "near_ood"][:2]
    oods = far + near

    rows = []
    for model in args.models:
        arch = config.MODEL_ARCHS[model]
        wp = config.weight_path(args.dataset, model)
        if not wp.exists():
            print(f"[skip] {model}: no weights"); continue
        nm = load_normalized_model(arch, wp, num_classes=config.NUM_CLASSES, device=DEVICE).eval()
        layers = pick_layers(nm.model, args.n_layers)
        names = [n for n, _ in layers]
        hook = MultiHook(layers)
        print(f"\n=== {model} ({arch}) | {len(names)} conv layers ===", flush=True)

        prof = profile(nm, hook, get_id_loader(args.dataset, batch_size=128, num_workers=4, train=True), names, 1000)
        dorm = {nm_: dorm_idx(prof[nm_]) for nm_ in names}

        idl = get_id_loader(args.dataset, batch_size=128, num_workers=4, train=False)
        idV, idL = read(nm, hook, idl, names, dorm, args.n)

        oV = {nm_: [] for nm_ in names}; oL = {nm_: [] for nm_ in names}
        for o in oods:
            try:
                ol = get_ood_loader(o, batch_size=128, num_workers=4, max_samples=args.n)
            except Exception:  # noqa: BLE001
                continue
            rv, rl = read(nm, hook, ol, names, dorm, args.n)
            for nm_ in names:
                oV[nm_].append(rv[nm_]); oL[nm_].append(rl[nm_])
        oV = {nm_: np.concatenate(v) for nm_, v in oV.items() if v}
        oL = {nm_: np.concatenate(v) for nm_, v in oL.items() if v}

        for atk in args.attacks:
            h5 = config.ADV_DIR / f"{model}_{atk}.h5"
            if not h5.exists():
                print(f"   [skip adv] {h5.name}"); continue
            al = adv_loader_from_h5(h5, batch_size=128, num_workers=4)
            aV, aL = read(nm, hook, al, names, dorm, args.n)
            al.dataset.close()
            for d, nm_ in enumerate(names):
                row = {"model": model, "attack": atk, "depth": d, "layer": nm_,
                       "T2_shape": round(auroc_dl(idV[nm_], aV[nm_]), 3),
                       "T2_linf": round(auroc_dl(idL[nm_], aL[nm_]), 3),
                       "T3_shape": round(auroc_dl(oV[nm_], aV[nm_]), 3) if nm_ in oV else np.nan,
                       "T3_linf": round(auroc_dl(oL[nm_], aL[nm_]), 3) if nm_ in oL else np.nan}
                rows.append(row)
                print(f"   [{atk}] d{d:2} {nm_:30} T2 shape={row['T2_shape']:.3f} linf={row['T2_linf']:.3f}", flush=True)
        hook.close(); del nm; torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    out = args.csv or str(config.ANALYSIS_DIR / f"shape_depth_{args.dataset}.csv")
    df.to_csv(out, index=False)
    if len(df):
        print("\n=== mean T2 by depth (shape should peak at/near depth 0; linf weakest there) ===")
        print(df.groupby("depth")[["T2_shape", "T2_linf"]].mean().round(3).to_string())
    print(f"\n  saved -> {out}")


if __name__ == "__main__":
    main()
