"""Viyog score variants + stage-2 evaluation — one self-contained, fast file.

Keeps the ORIGINAL Viyog design (center a per-sample first-layer statistic by its
ID mean, then double-exponential squash) and swaps the statistic:

  * viyog_linf  (ORIGINAL): per-sample L∞ of first-layer activations  -> inf_norms
  * viyog_dorm  (NEW):      dormant-band activation fraction (a B_low_frac variant),
                            i.e. the share of first-layer mass in the filters that
                            are quietest on clean ID data, MINUS its ID mean.

Both are fit-free up to a single ID pass (Viyog needs the ID mean; the dormant
variant needs the ID filter ranking + the ID mean of the fraction). The squash
is monotonic, so a score's AUROC equals the AUROC of its centered statistic.

Runs on the existing feature files (feat_*.h5 or featfull_*.h5 — both store
filter_means + inf_norms). Reports, per model: AUROC for ID-vs-OOD / ID-vs-ADV /
OOD-vs-ADV, recall@5%FPR for stage-2 (ADV vs OOD), and old-vs-new deltas.

    python experiments/viyog_scores.py --dataset cifar100 [--prefix featfull] [--csv out.csv]
"""
from __future__ import annotations

import argparse
import glob

import config
import h5py
import numpy as np
from sklearn.metrics import roc_auc_score

EPS = 1e-8


def squash(centered: np.ndarray, T: float = 1000.0) -> np.ndarray:
    """Original Viyog double-exponential squash (monotonic; AUROC-preserving)."""
    x = centered / T
    s = np.sign(x)
    e = np.exp(np.abs(x))
    return s / (1.0 + np.exp(-e))


def linf_stat(d: dict) -> np.ndarray:
    """Original Viyog statistic: per-sample L∞ of first-layer activations."""
    return d["inf_norms"].astype(np.float64)


def dorm_stat(d: dict, dorm_idx: np.ndarray) -> np.ndarray:
    """NEW statistic: dormant-band activation fraction (B_low_frac variant)."""
    m = d["filter_means"].astype(np.float64)
    return m[:, dorm_idx].sum(1) / (m.sum(1) + EPS)


def auroc_dl(neg: np.ndarray, pos: np.ndarray) -> float:
    y = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
    s = np.r_[neg, pos]
    if len(np.unique(s)) < 2:
        return 0.5
    a = roc_auc_score(y, s)
    return max(a, 1 - a)


def recall_at_fpr(ood: np.ndarray, adv: np.ndarray, fpr: float = 0.05) -> float:
    """Stage-2: ADV=positive, OOD=negative. Threshold at `fpr` on OOD; TPR on ADV.
    Orientation auto-picked (ADV may be high- or low-tail)."""
    best = 0.0
    for sign in (+1, -1):
        o, a = sign * ood, sign * adv
        thr = np.quantile(o, 1 - fpr)        # top `fpr` of OOD above thr
        best = max(best, float((a > thr).mean()))
    return best


def load(prefix: str, model: str, split: str):
    fdir = config.FEATURES_DIR
    p = fdir / f"{prefix}_{model}_{split}.h5"
    if not p.exists():
        return None
    with h5py.File(p, "r") as f:
        return {k: f[k][:] for k in ("filter_means", "inf_norms") if k in f}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar100", choices=list(config.DATASET_SPECS))
    ap.add_argument("--prefix", default="feat", help="feat (legacy) or featfull (rich)")
    ap.add_argument("--low-pct", type=float, default=0.10)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--no-gate", action="store_true",
                    help="Include all models, even below the near-SOTA accuracy floor")
    args = ap.parse_args()
    config.set_dataset(args.dataset)

    if args.no_gate:
        models = list(config.MODEL_ARCHS)
    else:
        models, dropped = config.accepted_models(args.dataset)
        if dropped:
            print(f"[gate] dropped (clean acc < {config.ACC_FLOOR.get(args.dataset)}%): "
                  + ", ".join(f"{m}={a:.1f}" for m, a in dropped))
        print(f"[gate] accepted {len(models)} near-SOTA models: {models}")
    rows = []
    for model in models:
        idd = load(args.prefix, model, "id")
        if idd is None:
            print(f"[skip] {model}: no {args.prefix}_{model}_id.h5")
            continue
        m = idd["filter_means"].astype(np.float64)
        C = m.shape[1]
        order = np.argsort(m.mean(0))           # ascending: dormant first
        k = max(1, int(args.low_pct * C))
        dorm = order[:k]

        ood = sorted(glob.glob(str(config.FEATURES_DIR / f"{args.prefix}_{model}_ood_*.h5")))
        adv = sorted(glob.glob(str(config.FEATURES_DIR / f"{args.prefix}_{model}_adv_*.h5")))
        def pool(paths, fn):
            out = []
            for p in paths:
                with h5py.File(p, "r") as f:
                    d = {kk: f[kk][:] for kk in ("filter_means", "inf_norms") if kk in f}
                out.append(fn(d))
            return np.concatenate(out) if out else np.array([])

        for name, fn in [("viyog_linf", lambda d: linf_stat(d)),
                         ("viyog_dorm", lambda d: dorm_stat(d, dorm))]:
            i = fn(idd); o = pool(ood, fn); a = pool(adv, fn)
            rows.append({
                "model": model, "score": name,
                "T1_ID_OOD": auroc_dl(i, o) if len(o) else np.nan,
                "T2_ID_ADV": auroc_dl(i, a) if len(a) else np.nan,
                "T3_OOD_ADV": auroc_dl(o, a) if len(o) and len(a) else np.nan,
                "recall@5%FPR_stage2": recall_at_fpr(o, a) if len(o) and len(a) else np.nan,
            })

    # report
    import pandas as pd
    df = pd.DataFrame(rows)
    print(f"\n=== Viyog score comparison [{args.dataset}, prefix={args.prefix}] ===")
    piv = df.pivot_table(index="model", columns="score",
                         values=["T2_ID_ADV", "T3_OOD_ADV", "recall@5%FPR_stage2"])
    print(piv.round(3).to_string())
    # deltas (new − old), mean over models
    g = df.pivot_table(index="model", columns="score",
                       values=["T2_ID_ADV", "T3_OOD_ADV", "recall@5%FPR_stage2"])
    print("\n=== mean over models: NEW (dorm) − OLD (linf) ===")
    for met in ["T2_ID_ADV", "T3_OOD_ADV", "recall@5%FPR_stage2"]:
        old = g[(met, "viyog_linf")].mean()
        new = g[(met, "viyog_dorm")].mean()
        print(f"  {met:22}  old={old:.3f}  new={new:.3f}  Δ={new-old:+.3f}")
    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\n  saved → {args.csv}")


if __name__ == "__main__":
    main()
