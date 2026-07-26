"""Complementary TV+HF routing to rescue texture-OOD (D-w2 / D-d2b), training-free.

near_ood_breakdown showed Viyog* (first-conv TV on the dorm band) is strong on
far-OOD (~0.97 T3) but weak on texture-OOD (~0.6), while the high-frequency stat
G_hf is the mirror image (strong on texture, weaker on far). D-d2b asked whether a
detector *augmented with complementary early-layer statistics* fixes the ambiguous
regime. This tests a strictly PARAMETER-FREE, TRAINING-FREE combine (no learned
weights, so Viyog stays post-hoc): per signature, z-score against the ID distribution
(the only thing a deployed detector sees), then take the per-sample MAX z-anomaly over
{TV_dorm, HF}. OOD of any kind is high on at least one stat; ADV is suppressed on both.

Reports T3 (OOD-vs-ADV, directionless) by OOD kind for the singles and the combines,
so we can see whether zmax lifts texture WITHOUT sacrificing far/near.

    python experiments/exp_texture_routing.py --dataset cifar100
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
        return {k: f[k][:].astype(np.float64) for k in ("filter_means", "filter_tv", "filter_hf")}


def adaptive_band(prof, p=5.0):
    live = np.where(prof > 1e-4 * prof.max())[0]
    if len(live) < 4:
        live = np.arange(len(prof))
    order = live[np.argsort(prof[live])]
    return order[: max(1, int(round(p / 100.0 * len(order))))]


def raw_sigs(d, dorm):
    """Raw (un-normalised) per-sample signature values; higher = more OOD-like."""
    return {
        "TV_dorm": d["filter_tv"][:, dorm].mean(1),
        "HF_full": d["filter_hf"].mean(1),
        "HF_dorm": d["filter_hf"][:, dorm].mean(1),
    }


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
    print(f"=== TV+HF complementary routing (training-free) [{args.dataset}] {len(models)} models ===\n")
    rows = []
    for m in models:
        idp = FD / f"featfull_{m}_id.h5"
        if not idp.exists():
            continue
        idd = load(str(idp))
        if idd["filter_means"].max() == 0:
            continue
        dorm = adaptive_band(idd["filter_means"].mean(0))
        # ID-only calibration: mean/std per raw signature (post-hoc, label-free)
        id_raw = raw_sigs(idd, dorm)
        mu = {k: id_raw[k].mean() for k in id_raw}
        sd = {k: id_raw[k].std() + EPS for k in id_raw}

        def z_and_combos(d):
            r = raw_sigs(d, dorm)
            z = {k: (r[k] - mu[k]) / sd[k] for k in r}
            out = dict(z)  # TV_dorm, HF_full, HF_dorm as z-scores
            out["zmax_TV_HFfull"] = np.maximum(z["TV_dorm"], z["HF_full"])
            out["zmax_TV_HFdorm"] = np.maximum(z["TV_dorm"], z["HF_dorm"])
            out["zmean_TV_HFfull"] = 0.5 * (z["TV_dorm"] + z["HF_full"])
            return out

        advp = [p for p in sorted(glob.glob(str(FD / f"featfull_{m}_adv_*.h5"))) if load(p)["filter_means"].max() > 0]
        if not advp:
            continue
        adv = {k: np.concatenate([z_and_combos(load(p))[k] for p in advp]) for k in z_and_combos(idd)}

        kinds = {"far": [], "near": [], "texture": []}
        for p in sorted(glob.glob(str(FD / f"featfull_{m}_ood_*.h5"))):
            nm = os.path.basename(p).split("_ood_")[1][:-3]
            d = load(p)
            if d["filter_means"].max() == 0:
                continue
            kd = _KIND.get(nm, "")
            bucket = "far" if kd == "far_ood" else "near" if kd == "near_ood" else "texture" if kd == "texture_ood" else None
            if bucket:
                kinds[bucket].append(z_and_combos(d))

        for sname in z_and_combos(idd):
            row = {"model": m, "signature": sname}
            allood = []
            for kind, items in kinds.items():
                if not items:
                    row[f"T3_{kind}"] = np.nan
                    continue
                ood_s = np.concatenate([it[sname] for it in items])
                allood.append(ood_s)
                row[f"T3_{kind}"] = round(auroc_dl(ood_s, adv[sname]), 3)
            row["T3_all"] = round(auroc_dl(np.concatenate(allood), adv[sname]), 3) if allood else np.nan
            rows.append(row)
        print(f"  {m} done", flush=True)

    df = pd.DataFrame(rows)
    out = args.csv or str(config.ANALYSIS_DIR / f"texture_routing_{args.dataset}.csv")
    df.to_csv(out, index=False)
    g = df.groupby("signature")[["T3_far", "T3_near", "T3_texture", "T3_all"]].mean().round(3)
    g = g.reindex(["TV_dorm", "HF_full", "HF_dorm", "zmax_TV_HFfull", "zmax_TV_HFdorm", "zmean_TV_HFfull"])
    print("\n=== mean T3 by signature × OOD-kind (does zmax rescue texture w/o hurting far?) ===")
    print(g.to_string())
    base = g.loc["TV_dorm"]
    best = g.loc["zmax_TV_HFfull"]
    print(f"\nTexture: TV_dorm {base['T3_texture']:.3f} → zmax {best['T3_texture']:.3f} "
          f"({best['T3_texture']-base['T3_texture']:+.3f}); "
          f"Far cost {base['T3_far']:.3f} → {best['T3_far']:.3f} ({best['T3_far']-base['T3_far']:+.3f}); "
          f"All {base['T3_all']:.3f} → {best['T3_all']:.3f} ({best['T3_all']-base['T3_all']:+.3f})")
    print(f"\n  saved → {out}")


if __name__ == "__main__":
    main()
