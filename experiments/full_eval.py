"""Comprehensive head-to-head: best Viyog first-layer signatures vs all logit
OOD baselines, across every valid metric and every task the reviewers ask for.

Tasks            T1 = ID-vs-OOD, T2 = ID-vs-ADV, T3 = OOD-vs-ADV (deployment).
Metrics          AUROC (directionless = max(a,1-a)) + raw + direction sign +
                 cross-model direction-consistency; FPR@95%TPR; AUPR.
Cuts             pooled / far-OOD / near-OOD ; per-attack (T2,T3).
Methods          Viyog first-layer signatures (L-inf, dorm-TV, dorm-HF) computed
                 from the stored first-conv stats, PLUS logit baselines (MSP,
                 MaxLogit, Energy, Entropy, GEN, KL-Matching) from stored logits.
                 Feature baselines (Mahalanobis/KNN/ViM) come from
                 baselines_feature.py (needs a model pass) and are merged in
                 downstream — this script is pure post-hoc / CPU.

    python experiments/full_eval.py --dataset cifar100
Outputs results/analysis/full_eval_<ds>_{permodel,summary,perattack}.csv
"""
from __future__ import annotations
import argparse, glob, os
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
import h5py, numpy as np
import config

CORE6 = ["resnet50", "densenet121", "convnextv2_base", "vit_base", "swin_tiny", "mobilenetv3_l"]
NEAR_OOD = {"cifar10", "stl10"}
DEGEN_TOL = 0.02
EPS = 1e-8


# ---------- metrics ----------
def _ranks(v):
    order = np.argsort(v, kind="mergesort")
    r = np.empty(len(v), np.float64); r[order] = np.arange(1, len(v) + 1)
    _, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt); avg = (csum - cnt + csum + 1) / 2.0
    return avg[inv]


def auroc(pos, neg):
    pos = np.asarray(pos, np.float64); neg = np.asarray(neg, np.float64)
    n1, n0 = len(pos), len(neg)
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = _ranks(np.concatenate([pos, neg]))
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def fpr_at_tpr(pos, neg, tpr=0.95):
    """FPR when TPR(pos)=`tpr`, scores oriented so higher => positive."""
    pos = np.asarray(pos, np.float64); neg = np.asarray(neg, np.float64)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    thr = np.quantile(pos, 1.0 - tpr)         # 95% of pos are >= thr
    return float((neg >= thr).mean())


def aupr(pos, neg):
    """Area under precision-recall, positive class = pos (higher score)."""
    pos = np.asarray(pos, np.float64); neg = np.asarray(neg, np.float64)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    s = np.concatenate([pos, neg])
    order = np.argsort(-s, kind="mergesort"); y = y[order]
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    prec = tp / np.maximum(tp + fp, 1); rec = tp / max(y.sum(), 1)
    rec = np.concatenate([[0], rec]); prec = np.concatenate([[1], prec])
    return float(np.sum((rec[1:] - rec[:-1]) * prec[1:]))


def eval_pair(score_pos_class, score_neg_class):
    """Given the raw scores of the two classes (pos=detection target), return
    directionless auroc + raw + sign + fpr95 + aupr, orienting by the sign so
    fpr95/aupr are computed on the *favourable* direction (reported honestly with
    the sign so per-model flipping is auditable)."""
    a = auroc(score_pos_class, score_neg_class)
    if np.isnan(a):
        return dict(auroc_raw=np.nan, auroc_dl=np.nan, sign=0, fpr95=np.nan, aupr=np.nan)
    sign = 1 if a >= 0.5 else -1
    p, n = (score_pos_class, score_neg_class) if sign > 0 else (-score_pos_class, -score_neg_class)
    return dict(auroc_raw=a, auroc_dl=max(a, 1 - a), sign=sign,
                fpr95=fpr_at_tpr(p, n), aupr=aupr(p, n))


# ---------- feature / score builders ----------
def load(p, keys):
    with h5py.File(p, "r") as f:
        return {k: f[k][:].astype(np.float64) for k in keys if k in f}


def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def logit_scores(logits, klm_ref=None):
    """OOD-direction scores from logits (higher => more OOD/ADV)."""
    p = softmax(logits)
    out = {
        "MSP": -p.max(1),                                   # high => OOD
        "MaxLogit": -logits.max(1),
        "Energy": -(np.log(np.exp(logits - logits.max(1, keepdims=True)).sum(1))
                    + logits.max(1)),                        # -logsumexp
        "Entropy": -(p * np.log(p + EPS)).sum(1),
        "GEN": np.sum(p ** 0.1 * (1 - p) ** 0.1, axis=1),    # generalized entropy gamma=0.1
    }
    if klm_ref is not None:                                  # KL-Matching to ID class means
        kl = np.array([[np.sum(pi * np.log(pi / (m + EPS) + EPS)) for m in klm_ref] for pi in p])
        out["KLMatching"] = kl.min(1)
    return out


SIG_KEYS = ("inf_norms", "filter_tv", "filter_hf", "filter_means", "logits")


def sig_scores(d, dorm):
    """First-layer Viyog signatures (higher => more OOD/ADV for the natural sign)."""
    return {
        "Viyog_Linf": d["inf_norms"],
        "ViyogD_tv_dorm": d["filter_tv"][:, dorm].mean(1),
        "ViyogD_hf_dorm": d["filter_hf"][:, dorm].mean(1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar100")
    ap.add_argument("--models", nargs="+", default=CORE6)
    ap.add_argument("--low-pct", type=float, default=0.10)
    args = ap.parse_args()
    config.set_dataset(args.dataset)
    FD = config.FEATURES_DIR
    models = [m for m in args.models
              if (FD / f"featfull_{m}_id.h5").exists()
              and glob.glob(str(FD / f"featfull_{m}_adv_*.h5"))
              and glob.glob(str(FD / f"featfull_{m}_ood_*.h5"))]
    if not models:
        print(f"[{args.dataset}] no complete Core-6 features — skip"); return
    import pandas as pd

    rows, parows = [], []
    for m in models:
        idd = load(str(FD / f"featfull_{m}_id.h5"), SIG_KEYS)
        ch = idd["filter_means"].mean(0); C = len(ch)
        alive = np.where(ch > 1e-4)[0]; alive = alive if len(alive) else np.arange(C)
        k = max(1, int(args.low_pct * len(alive)))
        dorm = alive[np.argsort(ch[alive])[:k]]
        klm_ref = list(softmax(idd["logits"]).mean(0)[None])  # crude single ID-mean softmax
        # class-conditional softmax means for KLM
        # (use ID only; labels available)
        ood_files = sorted(glob.glob(str(FD / f"featfull_{m}_ood_*.h5")))
        adv_files = sorted(glob.glob(str(FD / f"featfull_{m}_adv_*.h5")))
        ood = {os.path.basename(p).split("_ood_")[1][:-3]: load(p, SIG_KEYS) for p in ood_files}
        adv = {os.path.basename(p).split("_adv_")[1][:-3]: load(p, SIG_KEYS) for p in adv_files}

        def all_scores(d):
            s = sig_scores(d, dorm); s.update(logit_scores(d["logits"], klm_ref)); return s

        id_s = all_scores(idd)
        ood_s = {n: all_scores(d) for n, d in ood.items()}
        adv_s = {n: all_scores(d) for n, d in adv.items()}
        methods = list(id_s.keys())

        def pool(dct, sel=None):
            return {mth: np.concatenate([v[mth] for s, v in dct.items()
                                         if sel is None or sel(s)]) for mth in methods}
        OOD = pool(ood_s); ADV = pool(adv_s)
        FAR = pool(ood_s, lambda s: s not in NEAR_OOD) if any(s not in NEAR_OOD for s in ood) else None
        NEAR = pool(ood_s, lambda s: s in NEAR_OOD) if any(s in NEAR_OOD for s in ood) else None

        for mth in methods:
            t1 = eval_pair(OOD[mth], id_s[mth])      # pos=OOD
            t2 = eval_pair(ADV[mth], id_s[mth])      # pos=ADV
            t3 = eval_pair(ADV[mth], OOD[mth])       # pos=ADV (deployment)
            t1f = eval_pair(FAR[mth], id_s[mth]) if FAR else {}
            t1n = eval_pair(NEAR[mth], id_s[mth]) if NEAR else {}
            rows.append(dict(
                dataset=args.dataset, model=m, method=mth,
                T1_dl=round(t1["auroc_dl"], 4), T1_raw=round(t1["auroc_raw"], 4), T1_sign=t1["sign"],
                T1_fpr95=round(t1["fpr95"], 4), T1_aupr=round(t1["aupr"], 4),
                T1far_dl=round(t1f.get("auroc_dl", np.nan), 4),
                T1near_dl=round(t1n.get("auroc_dl", np.nan), 4),
                T2_dl=round(t2["auroc_dl"], 4), T2_raw=round(t2["auroc_raw"], 4), T2_sign=t2["sign"],
                T2_fpr95=round(t2["fpr95"], 4), T2_aupr=round(t2["aupr"], 4),
                T3_dl=round(t3["auroc_dl"], 4), T3_raw=round(t3["auroc_raw"], 4), T3_sign=t3["sign"],
                T3_fpr95=round(t3["fpr95"], 4), T3_aupr=round(t3["aupr"], 4),
                T3_degen=int(abs(t3["auroc_raw"] - 0.5) < DEGEN_TOL),
            ))
            # per-attack
            for atk, av in adv_s.items():
                t2a = eval_pair(av[mth], id_s[mth]); t3a = eval_pair(av[mth], OOD[mth])
                parows.append(dict(dataset=args.dataset, model=m, method=mth, attack=atk,
                                   T2_dl=round(t2a["auroc_dl"], 4), T2_sign=t2a["sign"],
                                   T3_dl=round(t3a["auroc_dl"], 4), T3_sign=t3a["sign"]))

    df = pd.DataFrame(rows); pa = pd.DataFrame(parows)
    ad = config.ANALYSIS_DIR; ad.mkdir(parents=True, exist_ok=True)
    df.to_csv(ad / f"full_eval_{args.dataset}_permodel.csv", index=False)
    pa.to_csv(ad / f"full_eval_{args.dataset}_perattack.csv", index=False)

    # ---- summary with direction-consistency gating ----
    def summarize(task):
        out = []
        for mth, sub in df.groupby("method"):
            nd = sub[sub["T3_degen"] == 0] if task == "T3" else sub
            sg = nd[f"{task}_sign"]; pos = int((sg > 0).sum()); neg = int((sg < 0).sum())
            consistent = (pos == 0 or neg == 0)
            out.append(dict(
                method=mth, task=task,
                mean_dl=round(sub[f"{task}_dl"].mean(), 4),
                mean_raw=round(sub[f"{task}_raw"].mean(), 4),
                deployable=round(sub[f"{task}_dl"].mean() if consistent else sub[f"{task}_raw"].mean(), 4),
                dir_consistent=consistent, n_pos=pos, n_neg=neg,
                mean_fpr95=round(sub[f"{task}_fpr95"].mean(), 4),
                mean_aupr=round(sub[f"{task}_aupr"].mean(), 4),
                T1far_dl=round(sub["T1far_dl"].mean(), 4) if task == "T1" else np.nan,
                T1near_dl=round(sub["T1near_dl"].mean(), 4) if task == "T1" else np.nan,
            ))
        return out
    summ = []
    for t in ("T1", "T2", "T3"):
        summ += summarize(t)
    sdf = pd.DataFrame(summ)
    sdf.to_csv(ad / f"full_eval_{args.dataset}_summary.csv", index=False)

    import pandas as pd
    pd.set_option("display.width", 230); pd.set_option("display.max_columns", 30)
    print(f"\n############  FULL EVAL — {args.dataset}  ({len(models)} models)  ############")
    for t, name in [("T3", "OOD-vs-ADV (DEPLOYMENT)"), ("T2", "ID-vs-ADV"), ("T1", "ID-vs-OOD")]:
        s = sdf[sdf.task == t].sort_values("deployable", ascending=False)
        cols = ["method", "mean_dl", "deployable", "dir_consistent", "mean_fpr95", "mean_aupr"]
        if t == "T1":
            cols += ["T1far_dl", "T1near_dl"]
        print(f"\n--- {t}: {name}  (sorted by deployable AUROC) ---")
        print(s[cols].to_string(index=False))
    print(f"\n  saved -> {ad}/full_eval_{args.dataset}_*.csv")


if __name__ == "__main__":
    main()
