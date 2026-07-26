"""Bootstrap 95% CIs on the headline signature AUROCs (B-3 / C-w2, sampling-CI).

The reviewers ask for confidence intervals. True multi-*seed* CIs need re-finetuning
(queued separately); this provides the *sampling* CI now, on the existing CIFAR-100
first-conv features, by bootstrap-resampling the ID/OOD/ADV pools and recomputing the
directionless AUROC. Reported as sampling CIs (not seed CIs) — honest labelling.

    python experiments/exp_bootstrap_ci.py --dataset cifar100 --boot 1000
"""
from __future__ import annotations

import argparse
import glob
import os

import config
import h5py
import numpy as np
from sklearn.metrics import roc_auc_score

CORRUPT = {"mobileone_s1"}


def auroc_dl(neg: np.ndarray, pos: np.ndarray) -> float:
    y = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
    s = np.r_[neg, pos]
    if len(np.unique(s)) < 2:
        return 0.5
    a = roc_auc_score(y, s)
    return float(max(a, 1 - a))


def load(p: str) -> dict:
    with h5py.File(p, "r") as f:
        return {k: f[k][:].astype(np.float64) for k in ("filter_means", "filter_tv", "filter_maxs")}


def adaptive_band(id_prof: np.ndarray, p: float = 5.0) -> np.ndarray:
    C = len(id_prof)
    live = np.where(id_prof > 1e-4 * id_prof.max())[0]
    if len(live) < 4:
        live = np.arange(C)
    order = live[np.argsort(id_prof[live])]
    return order[: max(1, int(round(p / 100.0 * len(order))))]


def boot_ci(neg: np.ndarray, pos: np.ndarray, b: int, rng: np.random.Generator) -> tuple[float, float, float]:
    base = auroc_dl(neg, pos)
    vals = np.empty(b)
    for i in range(b):
        n = neg[rng.integers(0, len(neg), len(neg))]
        p = pos[rng.integers(0, len(pos), len(pos))]
        vals[i] = auroc_dl(n, p)
    return base, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar100", choices=list(config.DATASET_SPECS))
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    config.set_dataset(args.dataset)
    FD = config.FEATURES_DIR
    rng = np.random.default_rng(0)
    import pandas as pd

    models = sorted({os.path.basename(p).split("featfull_")[1].split("_id.h5")[0]
                     for p in glob.glob(str(FD / "featfull_*_id.h5"))})
    models = [m for m in models if m not in CORRUPT]
    rows = []
    print(f"=== bootstrap 95% sampling-CI [{args.dataset}], B={args.boot} ===\n")
    for m in models:
        idp = FD / f"featfull_{m}_id.h5"
        if not idp.exists():
            continue
        idd = load(str(idp))
        if idd["filter_means"].max() == 0:
            continue
        dorm = adaptive_band(idd["filter_means"].mean(0))

        def sig(d):
            return {"Viyog_D*": d["filter_tv"][:, dorm].mean(1),
                    "G_tv_mean": d["filter_tv"].mean(1),
                    "A_inf_norm": d["filter_maxs"].max(1)}

        adv = [load(p) for p in sorted(glob.glob(str(FD / f"featfull_{m}_adv_*.h5")))
               if load(p)["filter_means"].max() > 0]
        ood = [load(p) for p in sorted(glob.glob(str(FD / f"featfull_{m}_ood_*.h5")))
               if load(p)["filter_means"].max() > 0]
        if not adv or not ood:
            continue
        for sname in sig(idd):
            i_s = sig(idd)[sname]
            a_s = np.concatenate([sig(d)[sname] for d in adv])
            o_s = np.concatenate([sig(d)[sname] for d in ood])
            t2, t2l, t2h = boot_ci(i_s, a_s, args.boot, rng)   # ID vs ADV
            t3, t3l, t3h = boot_ci(o_s, a_s, args.boot, rng)   # OOD vs ADV
            rows.append({"model": m, "signature": sname,
                         "T2": round(t2, 3), "T2_lo": round(t2l, 3), "T2_hi": round(t2h, 3),
                         "T3": round(t3, 3), "T3_lo": round(t3l, 3), "T3_hi": round(t3h, 3)})
        print(f"  {m} done", flush=True)

    df = pd.DataFrame(rows)
    out = args.csv or str(config.ANALYSIS_DIR / f"bootstrap_ci_{args.dataset}.csv")
    df.to_csv(out, index=False)
    print("\n=== mean AUROC with 95% sampling-CI half-width by signature ===")
    g = df.groupby("signature").agg(
        T2=("T2", "mean"), T2_ci=("T2", lambda s: round((df.loc[s.index, "T2_hi"] - df.loc[s.index, "T2_lo"]).mean() / 2, 3)),
        T3=("T3", "mean"), T3_ci=("T3", lambda s: round((df.loc[s.index, "T3_hi"] - df.loc[s.index, "T3_lo"]).mean() / 2, 3)),
    ).round(3)
    print(g.to_string())
    print(f"\n  saved → {out}  (sampling CIs; seed CIs require re-finetune, queued)")


if __name__ == "__main__":
    main()
