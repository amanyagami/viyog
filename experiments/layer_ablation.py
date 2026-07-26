"""Layer-depth ablation for the L∞ separation claim (closes A-w3 / A-d4).

Reviewer A: "the authors justify using the FIRST layer theoretically but give no
ablation." This measures the ADV-vs-OOD (T3) and ID-vs-ADV (T2) AUROC of the
L∞-activation statistic at several depths — first conv, then evenly-spaced
intermediate convs, through to a late layer — to show empirically that the
separation is strongest at (or near) the first layer, as the suppression theory
predicts, and decays with depth.

Reuses the existing adversarial h5 (pgd / apgd_ce by default) plus the ID and OOD
loaders; runs one forward pass per batch with hooks on all chosen layers.

    CUDA_VISIBLE_DEVICES=1 python experiments/layer_ablation.py --dataset cifar100 \
        --models resnet50 densenet121 --attacks pgd apgd_ce --n 2000 --n-layers 6
"""
from __future__ import annotations

import argparse

import config
import numpy as np
import torch
import torch.nn as nn
from config import DEVICE
from data_utils import adv_loader_from_h5, get_id_loader, get_ood_loader
from model_utils import load_normalized_model
from sklearn.metrics import roc_auc_score


def pick_layers(backbone: nn.Module, n_layers: int) -> list[tuple[str, nn.Module]]:
    """Evenly-spaced Conv2d layers through the network, always including the first."""
    convs = [(n, m) for n, m in backbone.named_modules() if isinstance(m, nn.Conv2d)]
    if len(convs) <= n_layers:
        return convs
    idx = np.unique(np.linspace(0, len(convs) - 1, n_layers).round().astype(int))
    return [convs[i] for i in idx]


class MultiHook:
    """Capture detached outputs of several layers in one forward pass."""
    def __init__(self, layers: list[tuple[str, nn.Module]]) -> None:
        self.feats: dict[str, torch.Tensor] = {}
        self.handles = []
        for name, layer in layers:
            self.handles.append(layer.register_forward_hook(
                lambda m, i, o, n=name: self.feats.__setitem__(n, o.detach())))

    def close(self) -> None:
        for h in self.handles:
            h.remove()


def auroc_dl(neg: np.ndarray, pos: np.ndarray) -> float:
    y = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
    s = np.r_[neg, pos]
    if len(np.unique(s)) < 2:
        return 0.5
    a = roc_auc_score(y, s)
    return float(max(a, 1 - a))


@torch.no_grad()
def linf_per_layer(model, hook, loader, names, n):
    """Return {layer_name: (N,) per-sample L∞} over up to n samples."""
    acc = {nm: [] for nm in names}
    seen = 0
    for batch in loader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(DEVICE)
        model(x)
        for nm in names:
            f = hook.feats[nm]
            acc[nm].append(f.abs().flatten(1).amax(1).cpu().numpy())
        seen += x.shape[0]
        if seen >= n:
            break
    return {nm: np.concatenate(v)[:n] for nm, v in acc.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar100", choices=list(config.DATASET_SPECS))
    ap.add_argument("--models", nargs="+", default=["resnet50", "densenet121"])
    ap.add_argument("--attacks", nargs="+", default=["pgd", "apgd_ce"])
    ap.add_argument("--ood", nargs="+", default=None, help="OOD sets (default first 3)")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    config.set_dataset(args.dataset)
    # balanced far+near OOD set (NOT the near-biased first-3, which confounded the
    # depth comparison — L∞-style separation only exists for far-OOD)
    _kind = {k: (v.get("kind") if isinstance(v, dict) else v) for k, v in config.OOD_UNIVERSE.items()}
    _far = [o for o in config.OOD_DATASETS if _kind.get(o) == "far_ood"][:2]
    _near = [o for o in config.OOD_DATASETS if _kind.get(o) == "near_ood"][:2]
    oods = args.ood or (_far + _near)

    import pandas as pd
    rows = []
    for model in args.models:
        arch = config.MODEL_ARCHS[model]
        wp = config.weight_path(args.dataset, model)
        if not wp.exists():
            print(f"[skip] {model}: no weights")
            continue
        nm = load_normalized_model(arch, wp, num_classes=config.NUM_CLASSES, device=DEVICE).eval()
        layers = pick_layers(nm.model, args.n_layers)
        names = [n for n, _ in layers]
        hook = MultiHook(layers)
        print(f"\n=== {model} ({arch}) | {len(names)} layers ===")
        for d, nmn in enumerate(names):
            print(f"   depth {d}: {nmn}")

        idl = get_id_loader(args.dataset, batch_size=128, num_workers=4, train=False)
        id_l = linf_per_layer(nm, hook, idl, names, args.n)

        ood_l = {nm_: [] for nm_ in names}
        for o in oods:
            try:
                ol = get_ood_loader(o, batch_size=128, num_workers=4, max_samples=args.n)
            except Exception:  # noqa: BLE001
                continue
            r = linf_per_layer(nm, hook, ol, names, args.n)
            for nm_ in names:
                ood_l[nm_].append(r[nm_])
        ood_l = {nm_: np.concatenate(v) for nm_, v in ood_l.items() if v}

        for atk in args.attacks:
            h5 = config.ADV_DIR / f"{model}_{atk}.h5"
            if not h5.exists():
                print(f"   [skip adv] {h5.name} not found")
                continue
            al = adv_loader_from_h5(h5, batch_size=128, num_workers=4)
            adv_l = linf_per_layer(nm, hook, al, names, args.n)
            al.dataset.close()
            for d, nm_ in enumerate(names):
                t2 = auroc_dl(id_l[nm_], adv_l[nm_])
                t3 = auroc_dl(ood_l[nm_], adv_l[nm_]) if nm_ in ood_l else np.nan
                rows.append({"model": model, "attack": atk, "depth": d, "layer": nm_,
                             "T2_ID_ADV": round(t2, 3), "T3_OOD_ADV": round(t3, 3)})
                print(f"   [{atk}] depth {d:2} {nm_:32} T2={t2:.3f} T3={t3:.3f}")
        hook.close()
        del nm
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    if len(df):
        print("\n=== mean T3 (OOD-vs-ADV) by depth (first layer should win) ===")
        print(df.groupby("depth")[["T2_ID_ADV", "T3_OOD_ADV"]].mean().round(3).to_string())
    out = args.csv or str(config.ANALYSIS_DIR / f"layer_ablation_{args.dataset}.csv")
    df.to_csv(out, index=False)
    print(f"\n  saved → {out}")


if __name__ == "__main__":
    main()
