"""Full-pipeline (stage-1 + stage-2) evaluation across seeds, Viyog vs logit baselines.

Runs the complete cascade (stage-1 Energy gate on logits -> stage-2 OOD-vs-ADV router)
end-to-end, for several stage-2 detectors — Viyog* (first-conv TV/dorm), Viyog-L-inf
(original), and the pytorch-ood-style logit baselines (Energy / MSP / MaxLogit) used as
the stage-2 router — and repeats it over S bootstrap seeds to get mean +/- std on every
metric. All from featfull_*.h5 (logits + first-conv stats), CPU only.

Metrics per detector: T2 (ID-vs-ADV AUROC), T3 (OOD-vs-ADV AUROC), recall@5%FPR (stage-2),
and the end-to-end pipeline OOD recall, ADV recall, and ID-FP->ADV escalation.

Note: these are *sampling*-seed CIs (bootstrap of the ID/OOD/ADV pools); training-seed CIs
require re-finetuning (queued on GPU). Honest labelling.

    python experiments/exp_pipeline_seeds.py --dataset cifar100 --seeds 10
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


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / (e.sum(1, keepdims=True) + EPS)


def energy(logits):
    z = logits.astype(np.float64)
    m = z.max(1)
    return -(m + np.log(np.exp(z - m[:, None]).sum(1) + EPS))


def msp(logits):
    return -softmax(logits.astype(np.float64)).max(1)


def maxlogit(logits):
    return -logits.astype(np.float64).max(1)


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


def best_balanced_threshold(neg, pos):
    best_acc, best_thr, best_sign = -1.0, 0.0, 1
    for sign in (1, -1):
        n, p = sign * neg, sign * pos
        for t in np.quantile(np.r_[n, p], np.linspace(0.01, 0.99, 99)):
            ba = 0.5 * ((p > t).mean() + (n <= t).mean())
            if ba > best_acc:
                best_acc, best_thr, best_sign = ba, float(t), sign
    return best_thr, best_sign


def adaptive_band(id_prof, p=5.0):
    C = len(id_prof)
    live = np.where(id_prof > 1e-4 * id_prof.max())[0]
    if len(live) < 4:
        live = np.arange(C)
    order = live[np.argsort(id_prof[live])]
    return order[: max(1, int(round(p / 100.0 * len(order))))]


def loadf(p):
    with h5py.File(p, "r") as f:
        return {k: f[k][:].astype(np.float64) for k in ("filter_means", "filter_tv", "inf_norms", "logits")}


def s2_scores(d, dorm5, dorm10):
    """All stage-2 detector scores for a split."""
    m = d["filter_means"]
    return {
        "Viyog_D*": d["filter_tv"][:, dorm5].mean(1),
        "Viyog_Linf": d["inf_norms"],
        "viyog_dorm": m[:, dorm10].sum(1) / (m.sum(1) + EPS),
        "s2_Energy": energy(d["logits"]),
        "s2_MSP": msp(d["logits"]),
        "s2_MaxLogit": maxlogit(d["logits"]),
    }


def resample(rng, n):
    return rng.integers(0, n, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar100", choices=list(config.DATASET_SPECS))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--id-tpr", type=float, default=0.95)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    config.set_dataset(args.dataset)
    FD = config.FEATURES_DIR
    import pandas as pd

    models = sorted({os.path.basename(p).split("featfull_")[1].split("_id.h5")[0]
                     for p in glob.glob(str(FD / "featfull_*_id.h5"))})
    models = [m for m in models if m not in CORRUPT]
    print(f"=== full-pipeline x {args.seeds} seeds [{args.dataset}] {len(models)} models ===\n")
    rows = []
    for m in models:
        idp = FD / f"featfull_{m}_id.h5"
        if not idp.exists():
            continue
        idd = loadf(str(idp))
        if idd["filter_means"].max() == 0:
            continue
        prof = idd["filter_means"].mean(0)
        dorm5 = adaptive_band(prof, 5.0)
        dorm10 = np.argsort(prof)[: max(1, int(0.10 * len(prof)))]
        oodp = [p for p in sorted(glob.glob(str(FD / f"featfull_{m}_ood_*.h5"))) if loadf(p)["filter_means"].max() > 0]
        advp = [p for p in sorted(glob.glob(str(FD / f"featfull_{m}_adv_*.h5"))) if loadf(p)["filter_means"].max() > 0]
        if not oodp or not advp:
            continue
        ood = {k: np.concatenate([s2_scores(loadf(p), dorm5, dorm10)[k] for p in oodp]) for k in s2_scores(idd, dorm5, dorm10)}
        adv = {k: np.concatenate([s2_scores(loadf(p), dorm5, dorm10)[k] for p in advp]) for k in s2_scores(idd, dorm5, dorm10)}
        ids = s2_scores(idd, dorm5, dorm10)
        # stage-1 energy from logits
        e_id, e_ood, e_adv = energy(idd["logits"]), np.concatenate([energy(loadf(p)["logits"]) for p in oodp]), np.concatenate([energy(loadf(p)["logits"]) for p in advp])

        rng = np.random.default_rng(0)
        for seed in range(args.seeds):
            ri = resample(rng, len(e_id)); ro = resample(rng, len(e_ood)); ra = resample(rng, len(e_adv))
            tau = np.quantile(e_id[ri], args.id_tpr)
            f_id, f_ood, f_adv = e_id[ri] > tau, e_ood[ro] > tau, e_adv[ra] > tau
            for det in ids:
                vi, vo, va = ids[det][ri], ood[det][ro], adv[det][ra]
                t2 = auroc_dl(vi, va)
                t3 = auroc_dl(vo, va)
                r5 = recall_at_fpr(vo, va)
                # full cascade
                fo, fa = vo[f_ood], va[f_adv]
                if len(fo) < 5 or len(fa) < 5:
                    e2e_ood = e2e_adv = idfp = np.nan
                else:
                    thr, sign = best_balanced_threshold(fo, fa)
                    def route(v):
                        return (sign * v > thr).astype(int)
                    def predict(flag, v):
                        pr = np.zeros(len(v), dtype=int)
                        pr[flag] = np.where(route(v[flag]) == 1, 2, 1)
                        return pr
                    p_id, p_ood, p_adv = predict(f_id, vi), predict(f_ood, vo), predict(f_adv, va)
                    e2e_ood = float((p_ood == 1).mean())
                    e2e_adv = float((p_adv == 2).mean())
                    idfp = float((p_id[f_id] == 2).mean()) if f_id.any() else 0.0
                rows.append({"model": m, "detector": det, "seed": seed,
                             "T2": t2, "T3": t3, "recall@5FPR": r5,
                             "e2e_OOD_recall": e2e_ood, "e2e_ADV_recall": e2e_adv,
                             "ID_FP_to_ADV": idfp})
        print(f"  {m} done", flush=True)

    df = pd.DataFrame(rows)
    out = args.csv or str(config.ANALYSIS_DIR / f"pipeline_seeds_{args.dataset}.csv")
    df.to_csv(out, index=False)
    # aggregate: mean +/- std over (models x seeds) per detector
    agg = df.groupby("detector").agg(
        T2=("T2", "mean"), T2_sd=("T2", "std"),
        T3=("T3", "mean"), T3_sd=("T3", "std"),
        recall=("recall@5FPR", "mean"), recall_sd=("recall@5FPR", "std"),
        e2e_OOD=("e2e_OOD_recall", "mean"), e2e_OOD_sd=("e2e_OOD_recall", "std"),
        e2e_ADV=("e2e_ADV_recall", "mean"), e2e_ADV_sd=("e2e_ADV_recall", "std"),
        idfp=("ID_FP_to_ADV", "mean"), idfp_sd=("ID_FP_to_ADV", "std"),
    ).round(3).sort_values("T3", ascending=False)
    print("\n=== full-pipeline metrics (mean +/- std over models x seeds), by stage-2 detector ===")
    print(agg.to_string())
    print(f"\n  saved → {out}")


if __name__ == "__main__":
    main()
