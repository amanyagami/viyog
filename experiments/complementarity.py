"""How logit-OOD and first-layer Viyog signatures COMPLEMENT for overall
3-way ID / OOD / ADV separation.

Each family is strong on a different boundary:
  * logit scores (Energy/MSP/MaxLogit) separate ID|OOD but are BLIND to ADV
    (adversarials are optimised in logit space) -> T2 near/under chance.
  * first-layer Viyog (dorm-band TV/HF) separates ID|ADV and OOD|ADV but is
    mediocre on ID|OOD.
Neither alone gives good 3-way accuracy; together they do. We quantify this with
(a) an LDA over feature subsets (marginal value of adding Viyog to a logit
score) and (b) a transparent 2-stage cascade (Energy gates OOD, then Viyog
gates ADV). Balanced 3-way accuracy + per-class recall, 5-fold CV, pooled OOD/ADV.

    python experiments/complementarity.py --dataset cifar100
Outputs results/analysis/complementarity_<ds>.csv + plot fig_complementarity.png
CPU, post-hoc on stored logits + first-conv stats.
"""
from __future__ import annotations
import argparse, glob, os
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
import h5py, numpy as np
import config

CORE6 = ["resnet50", "densenet121", "convnextv2_base", "vit_base", "swin_tiny", "mobilenetv3_l"]
EPS = 1e-8
KEYS = ("inf_norms", "filter_tv", "filter_hf", "filter_means", "logits")


def load(p):
    with h5py.File(p, "r") as f:
        return {k: f[k][:].astype(np.float64) for k in KEYS if k in f}


def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def feats(d, dorm):
    """Per-sample feature dict (oriented; sign irrelevant to LDA)."""
    p = softmax(d["logits"]); lse = np.log(np.exp(d["logits"] - d["logits"].max(1, keepdims=True)).sum(1)) + d["logits"].max(1)
    return {
        "Energy": -lse, "MSP": -p.max(1), "MaxLogit": -d["logits"].max(1),
        "tv_dorm": d["filter_tv"][:, dorm].mean(1), "hf_dorm": d["filter_hf"][:, dorm].mean(1),
        "Linf": d["inf_norms"],
    }


FEATURE_SETS = {
    "Energy only (logit)":        ["Energy"],
    "Viyog L∞ (paper)":           ["Linf"],
    "Viyog only":               ["tv_dorm", "hf_dorm"],
    "Energy + Viyog":           ["Energy", "tv_dorm", "hf_dorm"],
    "Full panel":                 ["Energy", "MSP", "MaxLogit", "tv_dorm", "hf_dorm", "Linf"],
}


def lda_cv(X, y, folds=5, seed=0):
    """Balanced 3-way accuracy + per-class recall via stratified k-fold LDA."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    rng = np.random.RandomState(seed)
    idx = {c: rng.permutation(np.where(y == c)[0]) for c in np.unique(y)}
    parts = {c: np.array_split(idx[c], folds) for c in idx}
    recs = []
    for f in range(folds):
        te = np.concatenate([parts[c][f] for c in idx])
        tr = np.concatenate([np.concatenate([parts[c][g] for g in range(folds) if g != f]) for c in idx])
        Xs = (X - X[tr].mean(0)) / (X[tr].std(0) + EPS)
        clf = LDA().fit(Xs[tr], y[tr]); pred = clf.predict(Xs[te])
        rec = [np.mean(pred[y[te] == c] == c) for c in np.unique(y)]
        recs.append(rec)
    rec = np.mean(recs, 0)
    return float(rec.mean()), rec  # balanced acc, per-class recall (ID,OOD,ADV)


def cascade(fE, fV, yID, yOOD, yADV, tpr=0.95):
    """2-stage rule: Energy gates OOD (thr@95% ID-TPR), then Viyog gates ADV.
    Returns balanced 3-way accuracy + per-class recall on a balanced pool."""
    # thresholds from ID
    e_thr = np.quantile(fE["ID"], tpr)                  # OOD if Energy-score > e_thr
    v_thr = np.quantile(fV["ID"], tpr)                  # ADV if tv_dorm > v_thr
    def classify(e, v):
        out = np.zeros(len(e), int)                     # 0=ID
        out[e > e_thr] = 1                              # OOD
        rest = e <= e_thr
        out[rest & (v > v_thr)] = 2                     # ADV (only among non-OOD)
        return out
    # balance classes
    n = min(len(fE["ID"]), len(fE["OOD"]), len(fE["ADV"]))
    rng = np.random.RandomState(0)
    sel = {k: rng.choice(len(fE[k]), n, replace=False) for k in ("ID", "OOD", "ADV")}
    recs = []
    for ci, k in enumerate(("ID", "OOD", "ADV")):
        pred = classify(fE[k][sel[k]], fV[k][sel[k]])
        recs.append(np.mean(pred == ci))
    return float(np.mean(recs)), recs


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", default="cifar100")
    ap.add_argument("--models", nargs="+", default=CORE6); ap.add_argument("--low-pct", type=float, default=0.10)
    ap.add_argument("--per-class", type=int, default=4000)
    args = ap.parse_args()
    config.set_dataset(args.dataset); FD = config.FEATURES_DIR
    models = [m for m in args.models if (FD / f"featfull_{m}_id.h5").exists()
              and glob.glob(str(FD / f"featfull_{m}_adv_*.h5")) and glob.glob(str(FD / f"featfull_{m}_ood_*.h5"))]
    if not models:
        print(f"[{args.dataset}] no features"); return
    import pandas as pd

    rows, cas_rows = [], []
    for m in models:
        idd = load(str(FD / f"featfull_{m}_id.h5"))
        ch = idd["filter_means"].mean(0); C = len(ch)
        alive = np.where(ch > 1e-4)[0]; alive = alive if len(alive) else np.arange(C)
        k = max(1, int(args.low_pct * len(alive))); dorm = alive[np.argsort(ch[alive])[:k]]
        ood = np.concatenate([_stack(load(p), dorm) for p in glob.glob(str(FD / f"featfull_{m}_ood_*.h5"))])
        adv = np.concatenate([_stack(load(p), dorm) for p in glob.glob(str(FD / f"featfull_{m}_adv_*.h5"))])
        idf = _stack(idd, dorm)
        names = list(feats(idd, dorm).keys())
        # balanced 3-way pool
        n = min(args.per_class, len(idf), len(ood), len(adv))
        rng = np.random.RandomState(0)
        Xid, Xood, Xadv = idf[rng.choice(len(idf), n, False)], ood[rng.choice(len(ood), n, False)], adv[rng.choice(len(adv), n, False)]
        X = np.vstack([Xid, Xood, Xadv]); y = np.array([0] * n + [1] * n + [2] * n)
        for setname, cols in FEATURE_SETS.items():
            ci = [names.index(c) for c in cols]
            ba, rec = lda_cv(X[:, ci], y)
            rows.append(dict(dataset=args.dataset, model=m, feature_set=setname,
                             bal_acc=round(ba, 4), recall_ID=round(rec[0], 4),
                             recall_OOD=round(rec[1], 4), recall_ADV=round(rec[2], 4)))
        # transparent cascade
        col = {c: names.index(c) for c in names}
        fE = {"ID": idf[:, col["Energy"]], "OOD": ood[:, col["Energy"]], "ADV": adv[:, col["Energy"]]}
        fV = {"ID": idf[:, col["tv_dorm"]], "OOD": ood[:, col["tv_dorm"]], "ADV": adv[:, col["tv_dorm"]]}
        cba, crec = cascade(fE, fV, None, None, None)
        cas_rows.append(dict(dataset=args.dataset, model=m, feature_set="Cascade (E→OOD, V→ADV)",
                             bal_acc=round(cba, 4), recall_ID=round(crec[0], 4),
                             recall_OOD=round(crec[1], 4), recall_ADV=round(crec[2], 4)))

    df = pd.concat([pd.DataFrame(rows), pd.DataFrame(cas_rows)], ignore_index=True)
    ad = config.ANALYSIS_DIR; ad.mkdir(parents=True, exist_ok=True)
    df.to_csv(ad / f"complementarity_{args.dataset}.csv", index=False)
    agg = df.groupby("feature_set")[["bal_acc", "recall_ID", "recall_OOD", "recall_ADV"]].mean().round(3)
    order = list(FEATURE_SETS) + ["Cascade (E→OOD, V→ADV)"]
    agg = agg.reindex([o for o in order if o in agg.index])
    import pandas as pd
    pd.set_option("display.width", 160)
    print(f"\n######## COMPLEMENTARITY — {args.dataset} ({len(models)} models), balanced 3-way ID/OOD/ADV ########\n")
    print(agg.to_string())
    print(f"\n  saved -> {ad}/complementarity_{args.dataset}.csv")

    # plot
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 5))
        x = np.arange(len(agg)); w = 0.2
        for i, (c, lab, col_) in enumerate([("recall_ID", "ID recall", "#3182bd"),
                                            ("recall_OOD", "OOD recall", "#31a354"),
                                            ("recall_ADV", "ADV recall", "#e6550d"),
                                            ("bal_acc", "3-way bal-acc", "#000000")]):
            ax.bar(x + (i - 1.5) * w, agg[c].values, w, label=lab, color=col_,
                   alpha=0.95 if c == "bal_acc" else 0.85)
        ax.set_xticks(x); ax.set_xticklabels(agg.index, rotation=20, ha="right")
        ax.axhline(1 / 3, ls="--", c="gray", lw=0.8, label="chance (3-way)")
        ax.set_ylabel("recall / balanced accuracy"); ax.set_ylim(0, 1.0)
        ax.set_title(f"[{args.dataset}] Complementarity — Energy alone is blind to ADV; "
                     "Viyog alone weak on OOD; together they separate all three")
        ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.18)); ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        pp = config.PLOTS_DIR / "rebuttal" / "fig_complementarity.png"; pp.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(pp, dpi=150, bbox_inches="tight"); plt.close()
        print(f"  plot -> {pp}")
    except Exception as e:
        print(f"  [plot skipped] {e}")


def _stack(d, dorm):
    f = feats(d, dorm)
    return np.column_stack([f[k] for k in f])


if __name__ == "__main__":
    main()
