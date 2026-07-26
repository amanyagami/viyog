"""Aggregate Viyog* detection metrics across independently-finetuned seeds (B-3/C-w2).

Each seed was finetuned/attacked/extracted into its own VIYOG_RESULTS dir (namespaced
via the env override). This reads the per-seed featfull, computes Viyog* T2/T3/recall
per (seed, model), and reports mean +/- std ACROSS SEEDS — the training-seed CI the
reviewers asked for (init/shuffle variance of independently-trained models).

    python experiments/exp_multiseed_agg.py \
        --feat-dirs results/features results/multiseed/seed1/results/features ... \
        --models resnet50 densenet121 convnextv2_base vit_base swin_tiny mobilenetv3_l
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

EPS = 1e-8


def auroc_dl(neg, pos):
    if len(neg) == 0 or len(pos) == 0:
        return np.nan
    y = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
    s = np.r_[neg, pos]
    if len(np.unique(s)) < 2:
        return 0.5
    a = roc_auc_score(y, s)
    return float(max(a, 1 - a))


def recall_at_fpr(neg, pos, fpr=0.05):
    if len(neg) == 0 or len(pos) == 0:
        return np.nan
    best = 0.0
    for sign in (1.0, -1.0):
        tau = np.percentile(sign * neg, 100 * (1 - fpr))
        best = max(best, float(np.mean(sign * pos > tau)))
    return best


def adaptive_band(prof, p=5.0):
    live = np.where(prof > 1e-4 * prof.max())[0]
    if len(live) < 4:
        live = np.arange(len(prof))
    order = live[np.argsort(prof[live])]
    return order[: max(1, int(round(p / 100.0 * len(order))))]


def load(p):
    import h5py
    with h5py.File(p, "r") as f:
        return {k: f[k][:].astype(np.float64) for k in ("filter_means", "filter_tv")}


def viyogd(d, dorm):
    return d["filter_tv"][:, dorm].mean(1)


def metrics_for(fdir: Path, model: str):
    idp = fdir / f"featfull_{model}_id.h5"
    if not idp.exists():
        return None
    idd = load(str(idp))
    if idd["filter_means"].max() == 0:
        return None
    dorm = adaptive_band(idd["filter_means"].mean(0))
    advp = [p for p in sorted(glob.glob(str(fdir / f"featfull_{model}_adv_*.h5"))) if load(p)["filter_means"].max() > 0]
    oodp = [p for p in sorted(glob.glob(str(fdir / f"featfull_{model}_ood_*.h5"))) if load(p)["filter_means"].max() > 0]
    if not advp or not oodp:
        return None
    i_s = viyogd(idd, dorm)
    a_s = np.concatenate([viyogd(load(p), dorm) for p in advp])
    o_s = np.concatenate([viyogd(load(p), dorm) for p in oodp])
    return {"T2": auroc_dl(i_s, a_s), "T3": auroc_dl(o_s, a_s), "recall": recall_at_fpr(o_s, a_s)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-dirs", nargs="+", required=True, help="one featfull dir per seed")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--csv", default="results/analysis/multiseed_viyogd.csv")
    args = ap.parse_args()
    import pandas as pd

    rows = []
    for si, fd in enumerate(args.feat_dirs):
        for m in args.models:
            r = metrics_for(Path(fd), m)
            if r:
                rows.append({"seed_idx": si, "feat_dir": fd, "model": m, **{k: round(v, 4) for k, v in r.items()}})
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    df.to_csv(args.csv, index=False)
    print(f"=== Viyog* across {df.seed_idx.nunique()} seeds x {df.model.nunique()} models ===\n")
    # per-model mean +/- std across seeds
    g = df.groupby("model").agg(
        T2=("T2", "mean"), T2_sd=("T2", "std"),
        T3=("T3", "mean"), T3_sd=("T3", "std"),
        recall=("recall", "mean"), recall_sd=("recall", "std"),
        n=("seed_idx", "nunique")).round(3)
    print("per-model (mean +/- std across seeds):")
    print(g.to_string())
    print("\nOVERALL training-seed CI (mean +/- std over models x seeds):")
    for k in ["T2", "T3", "recall"]:
        print(f"  {k}: {df[k].mean():.3f} +/- {df[k].std():.3f}  "
              f"(per-model-mean std {g[k].std():.3f})")
    print(f"\n  saved -> {args.csv}")


if __name__ == "__main__":
    main()
