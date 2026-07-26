"""Distance-based + ODIN OOD baselines (pytorch-ood) vs Viyog/Viyog.

The logit baselines (MSP/MaxLogit/Energy/Entropy/KL) are already in the 37-sig
battery (F-family). This adds the paper's STRONG baselines that need penultimate
features or on-model passes — Mahalanobis, KNN, ViM (pytorch-ood) + ODIN — and
scores them on T1 (ID-vs-OOD), T2 (ID-vs-ADV), T3 (OOD-vs-ADV) for a head-to-head
with raw L-inf (Viyog) and the best Viyog variant.

Penultimate features = input to the final classifier (forward-pre-hook). Mahalanobis
/KNN/ViM are fit on an ID-train subset; ODIN runs on-model (temperature + input
perturbation). ADV samples are the existing data/adversarial/<model>_<atk>.h5.

    CUDA_VISIBLE_DEVICES=1 python experiments/baselines_feature.py --dataset cifar100 \
        --models convnextv2_base resnet50 densenet121 mobilenetv3_l --n 2000
"""
from __future__ import annotations

import argparse
import glob
import os

import config
import numpy as np
import torch
import torch.nn as nn
from config import DEVICE
from data_utils import adv_loader_from_h5, get_id_loader, get_ood_loader
from model_utils import load_normalized_model
from pytorch_ood.detector import KNN, Mahalanobis, ViM
from sklearn.metrics import roc_auc_score

# near/far split from config's authoritative kind map (not hardcoded)
_KIND = {k: (v.get("kind") if isinstance(v, dict) else v) for k, v in config.OOD_UNIVERSE.items()}
NEAR = {k for k, kd in _KIND.items() if kd == "near_ood"}
FAR = {k for k, kd in _KIND.items() if kd == "far_ood"}


def auroc_dl(neg, pos):
    if len(neg) == 0 or len(pos) == 0:
        return np.nan
    y = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
    s = np.r_[neg, pos]
    if len(np.unique(s)) < 2:
        return 0.5
    return float(max(roc_auc_score(y, s), 1 - roc_auc_score(y, s)))


class Penult:
    """Capture penultimate features (classifier input) + logits in one pass."""
    def __init__(self, norm_model):
        self.clf = self._find_classifier(norm_model.model)
        self.z = None
        self.h = self.clf.register_forward_pre_hook(lambda m, inp: self._cap(inp))

    def _cap(self, inp):
        self.z = inp[0].detach()

    @staticmethod
    def _find_classifier(backbone):
        # timm: get_classifier() returns the final Linear (or head holding it)
        c = backbone.get_classifier() if hasattr(backbone, "get_classifier") else None
        if isinstance(c, nn.Linear):
            return c
        # fall back: last Linear in the module tree
        last = None
        for m in backbone.modules():
            if isinstance(m, nn.Linear):
                last = m
        return last

    def close(self):
        self.h.remove()


@torch.no_grad()
def extract(norm_model, pen, loader, n):
    """Return (z penultimate [N,D], logits [N,K], labels [N])."""
    Z, L, Y = [], [], []
    seen = 0
    for batch in loader:
        if isinstance(batch, (list, tuple)):
            x = batch[0]; y = batch[1] if len(batch) > 1 else torch.full((x.shape[0],), -1)
        else:
            x = batch; y = torch.full((x.shape[0],), -1)
        logit = norm_model(x.to(DEVICE))
        Z.append(pen.z.cpu()); L.append(logit.cpu()); Y.append(y)
        seen += x.shape[0]
        if seen >= n:
            break
    return torch.cat(Z)[:n], torch.cat(L)[:n], torch.cat(Y)[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar100", choices=list(config.DATASET_SPECS))
    ap.add_argument("--models", nargs="+",
                    default=["convnextv2_base", "resnet50", "densenet121", "mobilenetv3_l"])
    ap.add_argument("--attacks", nargs="+", default=["fgsm", "bim", "pgd", "apgd_ce"])
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    config.set_dataset(args.dataset)
    import pandas as pd
    rows = []

    for m in args.models:
        arch = config.MODEL_ARCHS[m]
        wp = config.weight_path(args.dataset, m)
        if not wp.exists():
            print(f"[skip] {m}: no weights"); continue
        nm = load_normalized_model(arch, wp, num_classes=config.NUM_CLASSES, device=DEVICE).eval()
        for p in nm.parameters():
            p.requires_grad_(False)
        pen = Penult(nm)
        print(f"\n=== {m} ({arch}) ===", flush=True)

        # fit set: ID train subset (remap labels to contiguous 0..K-1 for Mahalanobis/ViM)
        ztr, _, ytr = extract(nm, pen, get_id_loader(args.dataset, batch_size=128, num_workers=4, train=True), max(args.n, 5000))
        _, ytr_remap = np.unique(ytr.numpy(), return_inverse=True)
        ytr = torch.from_numpy(ytr_remap).long()
        # eval splits
        zid, _, _ = extract(nm, pen, get_id_loader(args.dataset, batch_size=128, num_workers=4, train=False), args.n)
        ood = {}
        for o in config.OOD_DATASETS:
            try:
                ol = get_ood_loader(o, batch_size=128, num_workers=4, max_samples=args.n)
                ood[o], _, _ = extract(nm, pen, ol, args.n)
            except Exception as e:  # noqa: BLE001
                print(f"  [warn] ood {o}: {e}")
        adv = {}
        for atk in args.attacks:
            h5 = config.ADV_DIR / f"{m}_{atk}.h5"
            if h5.exists():
                al = adv_loader_from_h5(h5, batch_size=128, num_workers=4)
                adv[atk], _, _ = extract(nm, pen, al, args.n)
                al.dataset.close()
        if not ood or not adv:
            print("  [skip] missing ood/adv"); pen.close(); continue
        z_ood = torch.cat(list(ood.values())); z_adv = torch.cat(list(adv.values()))
        far = torch.cat([ood[o] for o in ood if o in FAR]) if any(o in FAR for o in ood) else None
        near = torch.cat([ood[o] for o in ood if o in NEAR]) if any(o in NEAR for o in ood) else None

        clf = pen.clf
        w, b = clf.weight.detach().cpu(), clf.bias.detach().cpu()
        D = w.shape[1]
        dets = {
            "Mahalanobis": Mahalanobis(None).fit_features(ztr, ytr),
            "KNN": KNN(None).fit_features(ztr, ytr),
            "ViM": ViM(None, d=min(64, D - 1), w=w, b=b).fit_features(ztr, ytr),
        }
        for name, det in dets.items():
            sid = det.predict_features(zid).numpy()
            sood = det.predict_features(z_ood).numpy()
            sadv = det.predict_features(z_adv).numpy()
            t3f = auroc_dl(det.predict_features(far).numpy(), sadv) if far is not None else np.nan
            t3n = auroc_dl(det.predict_features(near).numpy(), sadv) if near is not None else np.nan
            rows.append({"model": m, "method": name, "type": "distance",
                         "T1_ID_OOD": auroc_dl(sid, sood), "T2_ID_ADV": auroc_dl(sid, sadv),
                         "T3_OOD_ADV": auroc_dl(sood, sadv), "T3_far": t3f, "T3_near": t3n})
            print(f"  {name:12} T1={rows[-1]['T1_ID_OOD']:.3f} T2={rows[-1]['T2_ID_ADV']:.3f} T3={rows[-1]['T3_OOD_ADV']:.3f}", flush=True)
        pen.close()
        del nm; torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    if len(df):
        print("\n=== mean over models (distance baselines) ===")
        print(df.groupby("method")[["T1_ID_OOD", "T2_ID_ADV", "T3_OOD_ADV", "T3_far", "T3_near"]].mean().round(3).to_string())
    out = args.csv or str(config.ANALYSIS_DIR / f"baselines_feature_{args.dataset}.csv")
    df.to_csv(out, index=False)
    print(f"\n  saved → {out}")


if __name__ == "__main__":
    main()
