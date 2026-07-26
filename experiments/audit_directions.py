"""Direction-consistency audit for T1/T2/T3 separation.

For every (model, statistic, task) we report the *directionless* AUROC
max(a, 1-a) PLUS the sign of the separation and whether the model is
degenerate (|AUROC-0.5| < tol, statistic ~constant for a class). A
directionless score is only honest if the sign is CONSISTENT across the
non-degenerate models — otherwise max(a,1-a) is per-model cheating (no
single deployed threshold could realise it).

Tasks:  T1 = ID vs OOD,  T2 = ID vs ADV,  T3 = OOD vs ADV (deployment).
Also dissects the L-inf ID-vs-OOD headline by far/near OOD split.

    python experiments/audit_directions.py --dataset cifar100
Outputs:
    results/analysis/dir_audit_<ds>.csv        (per model/stat/task)
    results/analysis/dir_audit_<ds>_summary.csv (per stat/task consistency)
    results/analysis/dir_audit_<ds>_ood.csv    (L-inf T1 per OOD split)
CPU only, pure post-hoc.
"""
from __future__ import annotations
import argparse, glob, os
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
import h5py, numpy as np
import config

CORE6 = ["resnet50", "densenet121", "convnextv2_base", "vit_base", "swin_tiny", "mobilenetv3_l"]
NEAR_OOD = {"cifar10", "stl10"}          # natural-image, near-distribution (for cifar100 ID)
DEGEN_TOL = 0.02                          # |AUROC-0.5| below this => direction undefined


def auroc(pos, neg):
    """AUROC with `pos` as the positive class (rank-sum / Mann-Whitney)."""
    pos = np.asarray(pos, np.float64); neg = np.asarray(neg, np.float64)
    n1, n0 = len(pos), len(neg)
    if n1 == 0 or n0 == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv), np.float64)
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt); start = csum - cnt
    avg = (start + csum + 1) / 2.0
    ranks = avg[inv]
    r1 = ranks[:n1].sum()
    return float((r1 - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def dirless(pos, neg):
    """(directionless auroc, raw auroc with pos=positive, sign: +1 if pos>neg)."""
    a = auroc(pos, neg)
    if np.isnan(a):
        return float("nan"), float("nan"), 0
    return max(a, 1 - a), a, (1 if a >= 0.5 else -1)


def load(p, keys):
    with h5py.File(p, "r") as f:
        return {k: f[k][:].astype(np.float64) for k in keys if k in f}


def stat_vectors(d, dorm):
    """Return the per-sample scalar for each statistic from a loaded dict."""
    out = {"Linf": d["inf_norms"], "gram": d["gram_offdiag"]}
    out["tv_dorm"] = d["filter_tv"][:, dorm].mean(1)
    out["hf_dorm"] = d["filter_hf"][:, dorm].mean(1)
    out["std_dorm"] = d["filter_std"][:, dorm].mean(1)
    out["mean_dorm"] = d["filter_means"][:, dorm].mean(1)
    return out


KEYS = ("inf_norms", "gram_offdiag", "filter_tv", "filter_hf", "filter_std",
        "filter_means", "filter_maxs", "filter_l2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar100")
    ap.add_argument("--low-pct", type=float, default=0.10)
    ap.add_argument("--models", nargs="+", default=CORE6)
    args = ap.parse_args()
    config.set_dataset(args.dataset)
    FD = config.FEATURES_DIR
    models = [m for m in args.models
              if (FD / f"featfull_{m}_id.h5").exists()
              and glob.glob(str(FD / f"featfull_{m}_adv_*.h5"))
              and glob.glob(str(FD / f"featfull_{m}_ood_*.h5"))]
    if not models:
        print(f"[{args.dataset}] no complete Core-6 features yet — skip"); return
    import pandas as pd

    rows, ood_rows = [], []
    for m in models:
        idd = load(str(FD / f"featfull_{m}_id.h5"), KEYS)
        fm = idd["filter_means"]; C = fm.shape[1]
        # Dormant band among ALIVE channels only. Some first-conv channels are
        # permanently dead (mean==0, TV==0, e.g. 25/64 in densenet121); the naive
        # "lowest 10%" picks those dead channels -> all-zero statistic -> AUROC=0.5
        # (spurious degeneracy). Select the low-activation band among alive channels.
        ch_mean = fm.mean(0)
        alive = np.where(ch_mean > 1e-4)[0]
        if len(alive) == 0:
            alive = np.arange(C)
        k = max(1, int(args.low_pct * len(alive)))
        dorm = alive[np.argsort(ch_mean[alive])[:k]]
        ood_files = sorted(glob.glob(str(FD / f"featfull_{m}_ood_*.h5")))
        adv_files = sorted(glob.glob(str(FD / f"featfull_{m}_adv_*.h5")))
        ood = {os.path.basename(p).split("_ood_")[1][:-3]: load(p, KEYS) for p in ood_files}
        adv = {os.path.basename(p).split("_adv_")[1][:-3]: load(p, KEYS) for p in adv_files}

        id_s = stat_vectors(idd, dorm)
        # concat all OOD / ADV
        oa = {name: np.concatenate([stat_vectors(d, dorm)[name] for d in ood.values()])
              for name in id_s}
        aa = {name: np.concatenate([stat_vectors(d, dorm)[name] for d in adv.values()])
              for name in id_s}
        far_splits = [s for s in ood if s not in NEAR_OOD]
        near_splits = [s for s in ood if s in NEAR_OOD]
        far = {name: np.concatenate([stat_vectors(ood[s], dorm)[name] for s in far_splits])
               for name in id_s} if far_splits else {}
        near = {name: np.concatenate([stat_vectors(ood[s], dorm)[name] for s in near_splits])
                for name in id_s} if near_splits else {}

        for name in id_s:
            # T1 ID-vs-OOD (pos=OOD): paper claims OOD>ID for Linf
            t1_dl, t1_raw, t1_sgn = dirless(oa[name], id_s[name])
            # T2 ID-vs-ADV (pos=ADV)
            t2_dl, t2_raw, t2_sgn = dirless(aa[name], id_s[name])
            # T3 OOD-vs-ADV (pos=ADV): the deployment task
            t3_dl, t3_raw, t3_sgn = dirless(aa[name], oa[name])
            t1f_dl, t1f_raw, _ = dirless(far[name], id_s[name]) if far else (float("nan"),) * 3
            t1n_dl, t1n_raw, _ = dirless(near[name], id_s[name]) if near else (float("nan"),) * 3
            rows.append(dict(
                model=m, stat=name,
                T1_dl=round(t1_dl, 3), T1_raw=round(t1_raw, 3), T1_sign=t1_sgn,
                T1far_dl=round(t1f_dl, 3), T1far_raw=round(t1f_raw, 3),
                T1near_dl=round(t1n_dl, 3), T1near_raw=round(t1n_raw, 3),
                T2_dl=round(t2_dl, 3), T2_raw=round(t2_raw, 3), T2_sign=t2_sgn,
                T3_dl=round(t3_dl, 3), T3_raw=round(t3_raw, 3), T3_sign=t3_sgn,
                T3_degen=int(abs(t3_raw - 0.5) < DEGEN_TOL),
            ))
        # L-inf per-OOD-split T1 (for the 92.38 headline dissection)
        for s, d in ood.items():
            v = stat_vectors(d, dorm)["Linf"]
            dl, raw, sgn = dirless(v, id_s["Linf"])
            ood_rows.append(dict(model=m, ood=s, kind="near" if s in NEAR_OOD else "far",
                                 T1_dl=round(dl, 3), T1_raw=round(raw, 3),
                                 OOD_gt_ID=int(sgn > 0)))

    df = pd.DataFrame(rows)
    ad = config.ANALYSIS_DIR; ad.mkdir(parents=True, exist_ok=True)
    df.to_csv(ad / f"dir_audit_{args.dataset}.csv", index=False)

    # ---- consistency summary per stat/task ----
    def summarize(task):
        g = df.groupby("stat")
        out = []
        for stat, sub in g:
            nd = sub[sub["T3_degen"] == 0] if task == "T3" else sub
            signs = nd[f"{task}_sign"]
            pos = int((signs > 0).sum()); neg = int((signs < 0).sum())
            consistent = (pos == 0 or neg == 0)
            out.append(dict(
                stat=stat, task=task,
                mean_raw=round(sub[f"{task}_raw"].mean(), 3),
                mean_dl=round(sub[f"{task}_dl"].mean(), 3),
                n_pos=pos, n_neg=neg, n_degen=int(sub.get("T3_degen", 0).sum()) if task == "T3" else 0,
                direction_consistent=consistent,
                # honest deployable score: dirless only if consistent, else raw (can't flip)
                deployable_dl=round(sub[f"{task}_dl"].mean() if consistent else sub[f"{task}_raw"].mean(), 3),
            ))
        return out

    summ = []
    for t in ("T1", "T2", "T3"):
        summ += summarize(t)
    sdf = pd.DataFrame(summ)
    sdf.to_csv(ad / f"dir_audit_{args.dataset}_summary.csv", index=False)

    odf = pd.DataFrame(ood_rows)
    odf.to_csv(ad / f"dir_audit_{args.dataset}_ood.csv", index=False)

    # ---- console report ----
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
    print(f"\n################  DIRECTION AUDIT — {args.dataset}  ({len(models)} models)  ################")
    print("\n--- per-model T3 (OOD vs ADV, the deployment task) ---")
    print(df[df.stat.isin(["Linf", "tv_dorm", "hf_dorm", "gram"])]
          [["model", "stat", "T3_raw", "T3_dl", "T3_sign", "T3_degen"]].to_string(index=False))
    print("\n--- consistency summary (deployable_dl = dirless ONLY if direction consistent) ---")
    print(sdf.to_string(index=False))
    print("\n--- L-inf ID-vs-OOD headline, by OOD split (far drives 92.x; near collapses) ---")
    pv = odf.pivot_table(index=["kind", "ood"], values=["T1_dl", "T1_raw", "OOD_gt_ID"], aggfunc="mean").round(3)
    print(pv.to_string())
    print("\n  L-inf T1 by far/near (mean over models & splits):")
    print(odf.groupby("kind")[["T1_dl", "T1_raw", "OOD_gt_ID"]].mean().round(3).to_string())
    print(f"\n  saved -> {ad}/dir_audit_{args.dataset}*.csv")


if __name__ == "__main__":
    main()
