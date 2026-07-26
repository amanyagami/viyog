"""Deep-dive: which first-layer signature is the best drop-in for Viyog's L-inf,
which separates ID/OOD/ADV best as a single axis, end-to-end recall, and how
performance scales with the NUMBER of signatures combined.

(A) DROP-IN test: Viyog squashes sign(stat-mu_ID) -> routes ADV(-)/OOD(+). A
    statistic is a clean drop-in only if ADV and OOD straddle mu_ID (opposite
    signs). We measure the straddle + the directionless AUROCs.
(B) SINGLE-AXIS 3-way: balanced 3-way accuracy from a 2-threshold rule on one
    statistic (only meaningful when it straddles).
(C) END-TO-END recall via the existing cascade idea, per signature.
(D) PANEL: LDA on the top-k signatures (k=1..K), 5-fold CV, 3-way balanced acc
    and per-class recall vs k.

    python experiments/signature_3way_analysis.py --dataset cifar100
"""
from __future__ import annotations
import argparse, glob, os
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
import h5py, numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
import config

EPS = 1e-8


def auroc_dl(a, b):
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]; s = np.r_[a, b]
    if len(np.unique(s)) < 2:
        return 0.5
    v = roc_auc_score(y, s); return float(max(v, 1 - v))


def gini(x):                                   # per-sample Gini over filters
    x = np.sort(np.abs(x), 1); n = x.shape[1]
    idx = np.arange(1, n + 1)
    return (2 * (idx * x).sum(1)) / (n * x.sum(1) + EPS) - (n + 1) / n


def entropy(x):
    p = x / (x.sum(1, keepdims=True) + EPS)
    return -(p * np.log(p + EPS)).sum(1)


def sigs_from(d, dorm, large, idref):
    """Per-sample dict of signatures from one featfull split."""
    fm = d["filter_means"]; tv = d["filter_tv"]; hf = d["filter_hf"]
    std = d["filter_std"]; Mx = d["filter_maxs"]; l2 = d["filter_l2"]
    z = d["logits"]
    mu, prec = idref
    xs = fm - mu
    maha = np.einsum("bi,ij,bj->b", xs, prec, xs)
    zc = z - z.max(1, keepdims=True); e = np.exp(zc)
    return {
        "A_inf_norm":      d["inf_norms"],                       # paper's Viyog
        "B_low_frac":      fm[:, dorm].sum(1) / (fm.sum(1) + EPS),
        "G_tv_dorm":       tv[:, dorm].mean(1),
        "G_tv_mean":       tv.mean(1),
        "G_hf_dorm":       hf[:, dorm].mean(1),
        "G_hf_low_large":  hf[:, dorm].mean(1) / (hf[:, large].mean(1) + EPS),
        "G_std_dorm":      std[:, dorm].mean(1),
        "H_dorm_entropy":  entropy(fm[:, dorm]),
        "D_crest_mean":    (Mx / (l2 + EPS)).mean(1),
        "C_gini":          gini(fm),
        "E_mahalanobis":   maha,
        "F_energy":        -(zc.max(1) + np.log(e.sum(1) + EPS)),  # -logsumexp
        "F_softmax_ent":   entropy(e / (e.sum(1, keepdims=True) + EPS)),
    }


def load(p):
    with h5py.File(p, "r") as f:
        return {k: f[k][:].astype(np.float64) for k in
                ("filter_means", "filter_tv", "filter_hf", "filter_std",
                 "filter_maxs", "filter_l2", "inf_norms", "logits") if k in f}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar100", choices=list(config.DATASET_SPECS))
    ap.add_argument("--low-pct", type=float, default=0.10)
    ap.add_argument("--models", nargs="+", default=None)
    args = ap.parse_args()
    config.set_dataset(args.dataset)
    FD = config.FEATURES_DIR
    models = args.models or [m for m in config.MODEL_ARCHS
                             if (FD / f"featfull_{m}_id.h5").exists()
                             and glob.glob(str(FD / f"featfull_{m}_adv_*.h5"))]

    # ---- accumulate per-model results ----
    from collections import defaultdict
    straddle = defaultdict(list); aurT = defaultdict(lambda: defaultdict(list))
    threeway = defaultdict(list)
    panel_acc = defaultdict(list); panel_rec = defaultdict(lambda: defaultdict(list))
    names = None

    for model in models:
        idp = FD / f"featfull_{model}_id.h5"
        idd = load(str(idp))
        fm = idd["filter_means"]; C = fm.shape[1]; mu = fm.mean(0)
        # Dormant band among ALIVE channels only: some first-conv channels are
        # permanently dead (mean==0, e.g. 25/64 in densenet121); selecting the
        # global lowest-k picks those dead channels -> all-zero signature ->
        # spurious AUROC=0.5. Restrict the low band to active channels.
        alive = np.where(mu > 1e-4)[0]
        if len(alive) == 0:
            alive = np.arange(C)
        k = max(1, int(args.low_pct * len(alive)))
        order = alive[np.argsort(mu[alive])]; dorm = order[:k]; large = order[-k:]
        prec = np.linalg.pinv(np.cov(fm.T) + 1e-3 * np.eye(C))
        idref = (mu, prec)

        idS = sigs_from(idd, dorm, large, idref)
        oodS = [sigs_from(load(p), dorm, large, idref)
                for p in glob.glob(str(FD / f"featfull_{model}_ood_*.h5"))]
        advS = [sigs_from(load(p), dorm, large, idref)
                for p in glob.glob(str(FD / f"featfull_{model}_adv_*.h5"))]
        if not oodS or not advS:
            continue
        names = list(idS.keys())
        ood = {n: np.concatenate([o[n] for o in oodS]) for n in names}
        adv = {n: np.concatenate([a[n] for a in advS]) for n in names}

        for n in names:
            i, o, a = idS[n], ood[n], adv[n]
            mi, mo, ma = np.median(i), np.median(o), np.median(a)
            # straddle: ADV and OOD on opposite sides of ID median
            straddle[n].append(int(np.sign(ma - mi) != np.sign(mo - mi) and (ma != mi) and (mo != mi)))
            aurT[n]["T1"].append(auroc_dl(i, o))
            aurT[n]["T2"].append(auroc_dl(i, a))
            aurT[n]["T3"].append(auroc_dl(o, a))
            # single-axis 3-way: 2 thresholds (ID band) via quantiles of ID, route tails
            lo, hi = np.quantile(i, 0.05), np.quantile(i, 0.95)
            def route(v):
                p = np.where(v < lo, "ADV", np.where(v > hi, "OOD", "ID"))
                return p
            # orient so ADV is the low tail (flip if ADV median > OOD median)
            flip = ma > mo
            def route_or(v):
                vv = -v if flip else v
                llo, hhi = (-hi, -lo) if flip else (lo, hi)
                return np.where(vv < llo, "ADV", np.where(vv > hhi, "OOD", "ID"))
            acc = (np.mean(route_or(i) == "ID") + np.mean(route_or(o) == "OOD")
                   + np.mean(route_or(a) == "ADV")) / 3
            threeway[n].append(acc)

        # ---- (D) PANEL: top-k by T3 over this model, LDA 5-fold ----
        rank = sorted(names, key=lambda n: -np.median([auroc_dl(ood[n], adv[n])]))
        X_parts, y_parts = [], []
        for lab, src in [(0, idS), (1, ood), (2, adv)]:
            n0 = len(next(iter(src.values())))
            X_parts.append(np.column_stack([src[n] for n in rank]))
            y_parts.append(np.full(n0, lab))
        X = np.vstack(X_parts); y = np.concatenate(y_parts)
        X = (X - X.mean(0)) / (X.std(0) + EPS)
        for kk in range(1, len(rank) + 1):
            accs, recs = [], {0: [], 1: [], 2: []}
            skf = StratifiedKFold(5, shuffle=True, random_state=0)
            for tr, te in skf.split(X[:, :kk], y):
                clf = LinearDiscriminantAnalysis().fit(X[tr, :kk], y[tr])
                pr = clf.predict(X[te, :kk])
                accs.append(np.mean([np.mean(pr[y[te] == c] == c) for c in (0, 1, 2)]))
                for c in (0, 1, 2):
                    recs[c].append(np.mean(pr[y[te] == c] == c))
            panel_acc[kk].append(np.mean(accs))
            for c in (0, 1, 2):
                panel_rec[kk][c].append(np.mean(recs[c]))

    # ---- report ----
    print(f"\n=== Signature deep-dive [{args.dataset}] · {len(models)} models · pooled 10-OOD ===\n")
    print("(A) DROP-IN suitability (straddle = ADV & OOD on opposite sides of ID) + AUROCs")
    print(f"{'signature':16}{'straddle%':>10}{'T1_OOD':>8}{'T2_ADV':>8}{'T3_O-A':>8}{'1ax-3way':>10}")
    rows = []
    for n in names:
        st = 100 * np.mean(straddle[n])
        t1, t2, t3 = (np.mean(aurT[n][t]) for t in ("T1", "T2", "T3"))
        tw = np.mean(threeway[n])
        rows.append((n, st, t1, t2, t3, tw))
    for n, st, t1, t2, t3, tw in sorted(rows, key=lambda r: -r[4]):
        star = " <= clean drop-in" if (st >= 80 and t3 >= 0.70) else ""
        print(f"{n:16}{st:>9.0f}%{t1:>8.3f}{t2:>8.3f}{t3:>8.3f}{tw:>10.3f}{star}")

    print("\n(D) PANEL — LDA on top-k signatures (by T3), 5-fold CV, mean over models")
    print(f"{'k':>3}{'3way_bal_acc':>14}{'recall_ID':>11}{'recall_OOD':>12}{'recall_ADV':>12}")
    for kk in sorted(panel_acc):
        a = np.mean(panel_acc[kk])
        r = [np.mean(panel_rec[kk][c]) for c in (0, 1, 2)]
        print(f"{kk:>3}{a:>14.3f}{r[0]:>11.3f}{r[1]:>12.3f}{r[2]:>12.3f}")


if __name__ == "__main__":
    main()
