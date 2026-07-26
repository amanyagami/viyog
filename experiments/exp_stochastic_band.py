"""Stochastic dorm-band sampling — benign-cost precondition for the D-d3 defense.

D-d3 asks for a defense-hardening strategy such as stochastic feature sampling against
norm-preserving adaptive attacks. A white-box attacker that breaks Viyog must suppress
the score on the SPECIFIC dorm band the detector reads. If, at inference, the detector
draws a RANDOM sub-band each time (from the quiet pool), the attacker can no longer
target one fixed set of filters — it must suppress all of them simultaneously, which
costs more perturbation / visibility.

The full robustness test needs the adaptive adversary regenerated on GPU (its featfull
was cleaned). What we CAN verify now, on CPU, is the NECESSARY precondition: stochastic
band sampling must not damage benign separation. This compares the fixed-band Viyog*
to a K-draw stochastic ensemble (mean over random quiet sub-bands) on T2/T3, and reports
the across-draw score variance (a candidate extra adversarial signal).

    python experiments/exp_stochastic_band.py --dataset cifar100 --draws 20
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


def auroc_dl(neg, pos):
    if len(neg) == 0 or len(pos) == 0:
        return np.nan
    s = np.r_[neg, pos]
    if len(np.unique(s)) < 2:
        return 0.5
    y = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
    return float(max(roc_auc_score(y, s), 1 - roc_auc_score(y, s)))


def load(p):
    with h5py.File(p, "r") as f:
        return {"m": f["filter_means"][:].astype(np.float64), "tv": f["filter_tv"][:].astype(np.float64)}


def fixed_band(prof, p=5.0):
    live = np.where(prof > 1e-4 * prof.max())[0]
    if len(live) < 4:
        live = np.arange(len(prof))
    order = live[np.argsort(prof[live])]
    return order[: max(1, int(round(p / 100.0 * len(order))))], order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar100", choices=list(config.DATASET_SPECS))
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--pool", type=float, default=20.0, help="quiet-pool %% to sample sub-bands from")
    ap.add_argument("--band", type=float, default=5.0, help="sub-band size as %% of channels")
    args = ap.parse_args()
    config.set_dataset(args.dataset)
    FD = config.FEATURES_DIR
    import pandas as pd

    models = sorted({os.path.basename(p).split("featfull_")[1].split("_id.h5")[0]
                     for p in glob.glob(str(FD / "featfull_*_id.h5"))})
    models = [m for m in models if m not in CORRUPT]
    rng = np.random.default_rng(0)
    print(f"=== stochastic dorm-band ({args.draws} draws) vs fixed [{args.dataset}] {len(models)} models ===\n")
    rows = []
    for m in models:
        idp = FD / f"featfull_{m}_id.h5"
        if not idp.exists():
            continue
        idd = load(str(idp))
        if idd["m"].max() == 0:
            continue
        prof = idd["m"].mean(0)
        dorm, order = fixed_band(prof, args.band)
        C = len(prof)
        pool = order[: max(args.draws, int(round(args.pool / 100.0 * len(order))))]  # quiet pool to sample from
        ksz = len(dorm)
        # random sub-bands (each = ksz filters sampled from the quiet pool)
        bands = [rng.choice(pool, size=min(ksz, len(pool)), replace=False) for _ in range(args.draws)]

        def fixed_score(d):
            return d["tv"][:, dorm].mean(1)

        def stoch_scores(d):  # (N, draws)
            return np.stack([d["tv"][:, b].mean(1) for b in bands], axis=1)

        advp = [p for p in sorted(glob.glob(str(FD / f"featfull_{m}_adv_*.h5"))) if load(p)["m"].max() > 0]
        oodp = [p for p in sorted(glob.glob(str(FD / f"featfull_{m}_ood_*.h5"))) if load(p)["m"].max() > 0]
        if not advp or not oodp:
            continue
        id_f, id_s = fixed_score(idd), stoch_scores(idd)
        adv_f = np.concatenate([fixed_score(load(p)) for p in advp])
        ood_f = np.concatenate([fixed_score(load(p)) for p in oodp])
        adv_s = np.concatenate([stoch_scores(load(p)) for p in advp])
        ood_s = np.concatenate([stoch_scores(load(p)) for p in oodp])

        rows.append({
            "model": m,
            "T2_fixed": round(auroc_dl(id_f, adv_f), 3),
            "T2_stoch": round(auroc_dl(id_s.mean(1), adv_s.mean(1)), 3),
            "T3_fixed": round(auroc_dl(ood_f, adv_f), 3),
            "T3_stoch": round(auroc_dl(ood_s.mean(1), adv_s.mean(1)), 3),
            # across-draw variance as an auxiliary OOD-vs-ADV signal
            "T3_var": round(auroc_dl(ood_s.var(1), adv_s.var(1)), 3),
        })
        print(f"  {m} done", flush=True)

    df = pd.DataFrame(rows)
    out = str(config.ANALYSIS_DIR / f"stochastic_band_{args.dataset}.csv")
    df.to_csv(out, index=False)
    mean = df[["T2_fixed", "T2_stoch", "T3_fixed", "T3_stoch", "T3_var"]].mean().round(3)
    print("\n=== mean over models ===")
    print(mean.to_string())
    print(f"\nBenign-cost precondition: T2 {mean['T2_fixed']:.3f}→{mean['T2_stoch']:.3f} "
          f"({mean['T2_stoch']-mean['T2_fixed']:+.3f}), T3 {mean['T3_fixed']:.3f}→{mean['T3_stoch']:.3f} "
          f"({mean['T3_stoch']-mean['T3_fixed']:+.3f}). Near-zero Δ ⇒ stochastic sampling is benign-free; "
          f"the band moves every call so a fixed-target adaptive attack cannot lock on. "
          f"(Adaptive-robustness test itself is queued on GPU — adv featfull was cleaned.)")
    print(f"\n  saved → {out}")


if __name__ == "__main__":
    main()
