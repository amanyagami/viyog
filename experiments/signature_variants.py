"""Empirical variant search for the top-4 first-layer signatures.

Top-4 (by T3 OOD-vs-ADV, CIFAR-100): G_tv_dorm, G_hf_low_large, B_low_frac, G_hf_dorm
— all dormant-band statistics. This sweeps the design axes that define them, ALL
computable from the stored per-filter arrays in featfull_*.h5 (no GPU, no re-extract):

  * per-filter QUANTITY in the dormant band: mean|a| (mass), tv, hf, std, l2, max
  * BAND percentile: bottom {5,10,15,20,25}% of filters by ID-mean activation
  * BAND rule: `fixed` (bottom-p of all filters) vs `adaptive` (drop near-dead
    filters with ID-mean ≈ 0 FIRST, then bottom-p of the live ones — the candidate
    fix for the DenseNet degeneracy where the fixed band lands entirely in dead filters)
  * NORMALIZATION: raw band-mean · ratio (dorm/large) · mass-fraction (Σdorm/Σall)

For every variant × model it reports directionless AUROC for T1 (ID-vs-OOD),
T2 (ID-vs-ADV), T3 (OOD-vs-ADV), plus the FAR/NEAR T3 split, and flags degeneracy
(T3≈0.5). Output: per-model CSV + cross-model summary ranking variants.

    python experiments/signature_variants.py --dataset cifar100 [--csv out.csv]
"""
from __future__ import annotations

import argparse
import glob
import os

import config
import h5py
import numpy as np
from sklearn.metrics import roc_auc_score

EPS = 1e-8
# OOD near/far split derived from config's authoritative kind map (NOT hardcoded):
# near_ood = {cifar10, stl10, flowers102, food101}; far_ood = {svhn, mnist,
# fashionmnist, eurosat, gtsrb}; texture_ood = {dtd} (excluded from the binary split).
_KIND = {k: (v.get("kind") if isinstance(v, dict) else v) for k, v in config.OOD_UNIVERSE.items()}
NEAR = {k for k, kd in _KIND.items() if kd == "near_ood"}
FAR = {k for k, kd in _KIND.items() if kd == "far_ood"}
PERCENTILES = [5, 10, 15, 20, 25]
QUANTS = ["mean", "tv", "hf", "std", "l2", "max"]          # per-filter quantity
KEY = {"mean": "filter_means", "tv": "filter_tv", "hf": "filter_hf",
       "std": "filter_std", "l2": "filter_l2", "max": "filter_maxs"}
NORMS = ["bandmean", "ratio_dorm_large", "massfrac"]       # aggregation/normalization
CORRUPT = {"mobileone_s1"}


def auroc_dl(neg, pos):
    if len(neg) == 0 or len(pos) == 0:
        return np.nan
    y = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
    s = np.r_[neg, pos]
    if len(np.unique(s)) < 2:
        return 0.5
    return float(max(roc_auc_score(y, s), 1 - roc_auc_score(y, s)))


def load(p):
    with h5py.File(p, "r") as f:
        return {k: f[k][:].astype(np.float64) for k in
                ("filter_means", "filter_tv", "filter_hf", "filter_std", "filter_l2", "filter_maxs")}


def bands(id_mean_profile, p, adaptive):
    """Return (dorm_idx, large_idx) for percentile p%. adaptive drops dead filters."""
    C = len(id_mean_profile)
    live = np.arange(C)
    if adaptive:
        live = np.where(id_mean_profile > 1e-4 * id_mean_profile.max())[0]
        if len(live) < 4:
            live = np.arange(C)
    order = live[np.argsort(id_mean_profile[live])]
    k = max(1, int(round(p / 100.0 * len(order))))
    return order[:k], order[-k:]


def variant_stat(d, quant, dorm, large, norm):
    """Per-sample scalar for (quantity, band, normalization)."""
    x = d[KEY[quant]]                                        # (N, C)
    if norm == "massfrac":                                  # share of mass in dorm band
        m = d["filter_means"]
        return m[:, dorm].sum(1) / (m.sum(1) + EPS)
    if norm == "ratio_dorm_large":
        return x[:, dorm].mean(1) / (x[:, large].mean(1) + EPS)
    return x[:, dorm].mean(1)                                # bandmean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar100", choices=list(config.DATASET_SPECS))
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    config.set_dataset(args.dataset)
    FD = config.FEATURES_DIR
    import pandas as pd

    models = sorted({os.path.basename(p).split("featfull_")[1].split("_id.h5")[0]
                     for p in glob.glob(str(FD / "featfull_*_id.h5"))})
    models = [m for m in models if m not in CORRUPT]
    print(f"=== variant search [{args.dataset}] over {len(models)} clean models ===")

    rows = []
    for m in models:
        idp = FD / f"featfull_{m}_id.h5"
        if not idp.exists():
            continue
        idd = load(idp)
        if idd["filter_means"].max() == 0:
            print(f"  [skip corrupt] {m}"); continue
        id_prof = idd["filter_means"].mean(0)
        oodp = sorted(glob.glob(str(FD / f"featfull_{m}_ood_*.h5")))
        advp = sorted(glob.glob(str(FD / f"featfull_{m}_adv_*.h5")))
        ood = [(os.path.basename(p).split("_ood_")[1][:-3], load(p)) for p in oodp]
        ood = [(n, d) for n, d in ood if d["filter_means"].max() > 0]
        adv = [load(p) for p in advp if load(p)["filter_means"].max() > 0]
        if not ood or not adv:
            continue

        for quant in QUANTS:
            for p in PERCENTILES:
                for adaptive in (False, True):
                    dorm, large = bands(id_prof, p, adaptive)
                    for norm in NORMS:
                        if norm == "massfrac" and quant != "mean":
                            continue                       # mass-fraction only meaningful on mean
                        name = f"{quant}|p{p}|{'adapt' if adaptive else 'fixed'}|{norm}"
                        i = variant_stat(idd, quant, dorm, large, norm)
                        a = np.concatenate([variant_stat(d, quant, dorm, large, norm) for d in adv])
                        o_all, far, near = [], [], []
                        for nm, d in ood:
                            v = variant_stat(d, quant, dorm, large, norm)
                            o_all.append(v)
                            if nm in NEAR:
                                near.append(v)
                            elif nm in FAR:
                                far.append(v)
                            # texture_ood (dtd) excluded from the binary far/near split
                        o_all = np.concatenate(o_all)
                        t3_far = auroc_dl(np.concatenate(far) if far else np.array([]), a)
                        t3_near = auroc_dl(np.concatenate(near) if near else np.array([]), a)
                        rows.append({
                            "model": m, "variant": name, "quant": quant, "pct": p,
                            "band": "adapt" if adaptive else "fixed", "norm": norm,
                            "T1_ID_OOD": auroc_dl(i, o_all),
                            "T2_ID_ADV": auroc_dl(i, a),
                            "T3_OOD_ADV": auroc_dl(o_all, a),
                            "T3_far": t3_far, "T3_near": t3_near,
                        })
        print(f"  {m} done")

    df = pd.DataFrame(rows)
    out = args.csv or str(config.ANALYSIS_DIR / f"signature_variants_{args.dataset}.csv")
    df.to_csv(out, index=False)
    # cross-model summary
    g = df.groupby("variant").agg(
        T2=("T2_ID_ADV", "mean"), T3=("T3_OOD_ADV", "mean"),
        T3_far=("T3_far", "mean"), T3_near=("T3_near", "mean"),
        T3_min=("T3_OOD_ADV", "min"),
        n_degen=("T3_OOD_ADV", lambda s: int((s < 0.55).sum())),
    ).round(3)
    print("\n=== TOP 15 variants by mean T3 (OOD-vs-ADV) ===")
    print(g.sort_values("T3", ascending=False).head(15).to_string())
    print("\n=== TOP 10 by FAR-OOD T3 ===")
    print(g.sort_values("T3_far", ascending=False).head(10)[["T2", "T3", "T3_far", "T3_near", "n_degen"]].to_string())
    print("\n=== fewest degeneracies (robustness) among T3>0.7 variants ===")
    print(g[g.T3 > 0.7].sort_values("n_degen").head(10).to_string())
    print(f"\n  saved → {out}")


if __name__ == "__main__":
    main()
