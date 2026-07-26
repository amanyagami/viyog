"""Near- / far- / texture-OOD breakdown for the top first-conv signatures (D-w2).

Reviewer D-w2: near-distribution and texture-biased OOD inputs may *also* suppress
early activations, so they would look ADV-like and erode the OOD-vs-ADV (T3) separation.
This tests that directly on the existing CIFAR-100 first-conv features (no GPU, no
re-extract): for each top signature it reports the directionless T3 (OOD-vs-ADV) AUROC
split by OOD kind — far / near / texture — per model. If near/texture T3 collapses
toward 0.5 while far stays high, D-w2's hypothesis holds and the detector is weakest
exactly in the ambiguous regime.

    python experiments/exp_near_ood.py --dataset cifar100
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
CORRUPT = {"mobileone_s1"}
_KIND = {k: (v.get("kind") if isinstance(v, dict) else v) for k, v in config.OOD_UNIVERSE.items()}


def auroc_dl(neg: np.ndarray, pos: np.ndarray) -> float:
    if len(neg) == 0 or len(pos) == 0:
        return np.nan
    y = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
    s = np.r_[neg, pos]
    if len(np.unique(s)) < 2:
        return 0.5
    a = roc_auc_score(y, s)
    return float(max(a, 1 - a))


def load(p: str) -> dict:
    with h5py.File(p, "r") as f:
        return {k: f[k][:].astype(np.float64) for k in
                ("filter_means", "filter_tv", "filter_hf", "filter_std", "filter_l2", "filter_maxs")}


def adaptive_band(id_prof: np.ndarray, p: float = 5.0) -> np.ndarray:
    """Bottom-p% quietest LIVE first-conv filters (drop near-dead first)."""
    C = len(id_prof)
    live = np.where(id_prof > 1e-4 * id_prof.max())[0]
    if len(live) < 4:
        live = np.arange(C)
    order = live[np.argsort(id_prof[live])]
    k = max(1, int(round(p / 100.0 * len(order))))
    return order[:k]


def sigs(d: dict, dorm: np.ndarray) -> dict[str, np.ndarray]:
    """Per-sample scalar for each headline signature."""
    return {
        "Viyog_D*(tv|p5|adapt)": d["filter_tv"][:, dorm].mean(1),
        "G_tv_mean": d["filter_tv"].mean(1),
        "G_hf_mean": d["filter_hf"].mean(1),
        "A_inf_norm": d["filter_maxs"].max(1),
    }


def main() -> None:
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
    print(f"=== near/far/texture-OOD T3 breakdown [{args.dataset}] {len(models)} models ===\n")
    rows = []
    for m in models:
        idp = FD / f"featfull_{m}_id.h5"
        if not idp.exists():
            continue
        idd = load(str(idp))
        if idd["filter_means"].max() == 0:
            continue
        dorm = adaptive_band(idd["filter_means"].mean(0))
        advp = sorted(glob.glob(str(FD / f"featfull_{m}_adv_*.h5")))
        adv = [load(p) for p in advp if load(p)["filter_means"].max() > 0]
        if not adv:
            continue
        adv_sig = {k: np.concatenate([sigs(d, dorm)[k] for d in adv]) for k in sigs(idd, dorm)}

        # group OOD by kind
        kinds = {"far": [], "near": [], "texture": []}
        for p in sorted(glob.glob(str(FD / f"featfull_{m}_ood_*.h5"))):
            nm = os.path.basename(p).split("_ood_")[1][:-3]
            d = load(p)
            if d["filter_means"].max() == 0:
                continue
            kd = _KIND.get(nm, "")
            bucket = "far" if kd == "far_ood" else "near" if kd == "near_ood" else "texture" if kd == "texture_ood" else None
            if bucket:
                kinds[bucket].append((nm, d))

        for sname in sigs(idd, dorm):
            row = {"model": m, "signature": sname}
            for kind, items in kinds.items():
                if not items:
                    row[f"T3_{kind}"] = np.nan
                    continue
                ood_s = np.concatenate([sigs(d, dorm)[sname] for _, d in items])
                row[f"T3_{kind}"] = round(auroc_dl(ood_s, adv_sig[sname]), 3)
            rows.append(row)
        print(f"  {m} done", flush=True)

    df = pd.DataFrame(rows)
    out = args.csv or str(config.ANALYSIS_DIR / f"near_ood_breakdown_{args.dataset}.csv")
    df.to_csv(out, index=False)
    print("\n=== mean T3 by signature × OOD-kind (does near/texture collapse vs far?) ===")
    g = df.groupby("signature")[["T3_far", "T3_near", "T3_texture"]].mean().round(3)
    print(g.to_string())
    print("\nInterpretation: T3_near/T3_texture << T3_far ⇒ D-w2 holds (detector weakest on ambiguous OOD).")
    print(f"\n  saved → {out}")


if __name__ == "__main__":
    main()
