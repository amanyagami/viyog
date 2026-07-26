"""Training-free combinations of first-conv signatures, evaluated end-to-end (D-d2).

S2 (near_ood_breakdown) showed the signatures are *complementary*: TV/dorm separate
far-OOD best, HF separates texture-OOD best, Viyog* is strongest on near-OOD. A
single statistic is therefore never best across all OOD kinds. This builds simple,
training-free ensembles — each signature z-scored on ID, then combined — and evaluates
them on T2 (ID-vs-ADV), T3 (OOD-vs-ADV), the far/near/texture T3 split, and the
operating-point recall@5%FPR, per model on the existing CIFAR-100 first-conv features.
The winner is the candidate for an upgraded, robust stage-2 detector.

    python experiments/exp_combinations.py --dataset cifar100
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
_KIND = {k: (v.get("kind") if isinstance(v, dict) else v) for k, v in config.OOD_UNIVERSE.items()}
KEYS = ("filter_means", "filter_tv", "filter_hf", "filter_std", "filter_maxs")


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
    """Directionless recall of `pos` (ADV) at a given FPR on `neg` (OOD)."""
    if len(neg) == 0 or len(pos) == 0:
        return np.nan
    best = 0.0
    for sign in (1.0, -1.0):
        n, p = sign * neg, sign * pos
        tau = np.percentile(n, 100 * (1 - fpr))
        best = max(best, float(np.mean(p > tau)))
    return best


def load(p):
    with h5py.File(p, "r") as f:
        return {k: f[k][:].astype(np.float64) for k in KEYS}


def adaptive_band(id_prof, p=5.0):
    C = len(id_prof)
    live = np.where(id_prof > 1e-4 * id_prof.max())[0]
    if len(live) < 4:
        live = np.arange(C)
    order = live[np.argsort(id_prof[live])]
    return order[: max(1, int(round(p / 100.0 * len(order))))]


def base_sigs(d, dorm):
    """Raw per-sample scalars for each base first-conv signature."""
    return {
        "ViyogD": d["filter_tv"][:, dorm].mean(1),
        "tv": d["filter_tv"].mean(1),
        "hf": d["filter_hf"].mean(1),
        "std": d["filter_std"].mean(1),
        "inf": d["filter_maxs"].max(1),
    }


# combinations are (name -> list of base-signature keys to z-average)
COMBOS = {
    "C_tv+hf": ["tv", "hf"],
    "C_dorm+hf": ["ViyogD", "hf"],
    "C_dorm+tv+hf": ["ViyogD", "tv", "hf"],
    "C_dorm+tv+hf+std": ["ViyogD", "tv", "hf", "std"],
    "C_all5": ["ViyogD", "tv", "hf", "std", "inf"],
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
    print(f"=== signature combinations [{args.dataset}] {len(models)} models ===\n")
    rows = []
    for m in models:
        idp = FD / f"featfull_{m}_id.h5"
        if not idp.exists():
            continue
        idd = load(str(idp))
        if idd["filter_means"].max() == 0:
            continue
        dorm = adaptive_band(idd["filter_means"].mean(0))
        advp = [p for p in sorted(glob.glob(str(FD / f"featfull_{m}_adv_*.h5"))) if load(p)["filter_means"].max() > 0]
        oodp = [p for p in sorted(glob.glob(str(FD / f"featfull_{m}_ood_*.h5"))) if load(p)["filter_means"].max() > 0]
        if not advp or not oodp:
            continue

        # z-score parameters from ID
        idb = base_sigs(idd, dorm)
        mu = {k: idb[k].mean() for k in idb}
        sd = {k: idb[k].std() + 1e-8 for k in idb}

        def z(d):  # z-scored base sigs for a split
            b = base_sigs(d, dorm)
            return {k: (b[k] - mu[k]) / sd[k] for k in b}

        zid = z(idd)
        zadv = [z(load(p)) for p in advp]
        ood_by_kind = {"far": [], "near": [], "texture": []}
        zood_all = []
        for p in oodp:
            nm = os.path.basename(p).split("_ood_")[1][:-3]
            zz = z(load(p))
            zood_all.append(zz)
            kd = _KIND.get(nm, "")
            b = "far" if kd == "far_ood" else "near" if kd == "near_ood" else "texture" if kd == "texture_ood" else None
            if b:
                ood_by_kind[b].append(zz)

        OPS = {"mean": lambda M: np.mean(M, 0), "maxabs": lambda M: M[np.argmax(np.abs(M), 0), np.arange(M.shape[1])],
               "min": lambda M: np.min(M, 0)}

        def combine(zsplit, keys, op="mean"):
            M = np.stack([zsplit[k] for k in keys], 0)
            return OPS[op](M)

        # baselines (single signatures) + mean combos + operator variants on the 4-sig set
        cand = {f"single_{k}": ([k], "mean") for k in ["ViyogD", "tv", "hf"]}
        cand.update({c: (keys, "mean") for c, keys in COMBOS.items()})
        cand["Cmaxabs_4"] = (["ViyogD", "tv", "hf", "std"], "maxabs")
        cand["Cmin_4"] = (["ViyogD", "tv", "hf", "std"], "min")
        for cname, (keys, op) in cand.items():
            i_s = combine(zid, keys, op)
            a_s = np.concatenate([combine(zz, keys, op) for zz in zadv])
            o_s = np.concatenate([combine(zz, keys, op) for zz in zood_all])
            t3_split = {}
            for kind, lst in ood_by_kind.items():
                ok = np.concatenate([combine(zz, keys, op) for zz in lst]) if lst else np.array([])
                t3_split[kind] = auroc_dl(ok, a_s)
            rows.append({
                "model": m, "combo": cname,
                "T2": round(auroc_dl(i_s, a_s), 3),
                "T3": round(auroc_dl(o_s, a_s), 3),
                "T3_far": round(t3_split["far"], 3),
                "T3_near": round(t3_split["near"], 3),
                "T3_texture": round(t3_split["texture"], 3),
                "recall@5FPR": round(recall_at_fpr(o_s, a_s), 3),
            })
        # ORACLE kind-router: per OOD kind, the best single base signature (upper bound)
        singles = ["ViyogD", "tv", "hf", "std", "inf"]
        a_by = {k: np.concatenate([base_sigs(load(p), dorm)[k] for p in advp]) for k in singles}
        i_by = {k: base_sigs(idd, dorm)[k] for k in singles}
        orow = {"model": m, "combo": "ORACLE_kind_router", "T2": round(max(auroc_dl(i_by[k], a_by[k]) for k in singles), 3)}
        tot = 0.0
        for kind, lst in ood_by_kind.items():
            if not lst:
                orow[f"T3_{kind}"] = np.nan; continue
            best = max(auroc_dl(np.concatenate([base_sigs(load(p2), dorm)[k] for p2 in oodp
                                if _KIND.get(os.path.basename(p2).split('_ood_')[1][:-3], '') == (kind + '_ood')]), a_by[k]) for k in singles)
            orow[f"T3_{kind}"] = round(best, 3)
        orow["T3"] = round(np.nanmean([orow.get(f"T3_{x}", np.nan) for x in ["far", "near", "texture"]]), 3)
        orow["recall@5FPR"] = np.nan
        rows.append(orow)
        print(f"  {m} done", flush=True)

    df = pd.DataFrame(rows)
    out = args.csv or str(config.ANALYSIS_DIR / f"combinations_{args.dataset}.csv")
    df.to_csv(out, index=False)
    g = df.groupby("combo").agg(
        T2=("T2", "mean"), T3=("T3", "mean"), T3_far=("T3_far", "mean"),
        T3_near=("T3_near", "mean"), T3_texture=("T3_texture", "mean"),
        recall=("recall@5FPR", "mean"),
        T3_min=("T3", "min"), n_degen=("T3", lambda s: int((s < 0.55).sum())),
    ).round(3).sort_values("T3", ascending=False)
    print("\n=== combos vs singles (mean over models), ranked by T3 ===")
    print(g.to_string())
    print("\n  worst-OOD-kind (the regime that matters for D-w2):")
    g["worst_kind_T3"] = g[["T3_far", "T3_near", "T3_texture"]].min(axis=1)
    print(g[["T3", "T3_near", "worst_kind_T3", "recall", "n_degen"]].sort_values("worst_kind_T3", ascending=False).to_string())
    print(f"\n  saved → {out}")


if __name__ == "__main__":
    main()
