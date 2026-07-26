"""Comprehensive END-TO-END recall over every available model (per dataset).

Stage-1 non-ID gate (Energy @ 5% ID-FPR) -> stage-2 OOD/ADV router. Reports, per
model, the end-to-end per-class recall (ID / OOD / ADV) and overall balanced
accuracy for THREE stage-2 statistics:
  * linf     : the paper's L-inf Viyog statistic
  * tv_dorm  : total-variation in the dormant band  (the best drop-in)
  * panel    : LDA over a 6-signature panel (logit + dormant TV/HF + L-inf + dorm-frac)

Iterates over whatever datasets have featfull features. Pure post-hoc, no GPU.

    python experiments/cascade_recall_full.py --datasets cifar100 [cifar10 ...]
"""
from __future__ import annotations
import argparse, glob, os
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
import h5py, numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
import config

EPS = 1e-8


def energy(z):
    zc = z - z.max(1, keepdims=True)
    return -(z.max(1) + np.log(np.exp(zc).sum(1) + EPS))


def softent(z):
    zc = z - z.max(1, keepdims=True); e = np.exp(zc); p = e / (e.sum(1, keepdims=True) + EPS)
    return -(p * np.log(p + EPS)).sum(1)


def load(p):
    with h5py.File(p, "r") as f:
        return {k: f[k][:].astype(np.float64) for k in
                ("filter_means", "filter_tv", "filter_hf", "inf_norms", "logits") if k in f}


def panel_feats(d, dorm, large):
    fm, tv, hf = d["filter_means"], d["filter_tv"], d["filter_hf"]
    return np.column_stack([
        energy(d["logits"]),                                  # logit confidence
        d["inf_norms"],                                       # L-inf
        tv[:, dorm].mean(1),                                  # TV dormant
        hf[:, dorm].mean(1),                                  # HF dormant
        hf[:, dorm].mean(1) / (hf[:, large].mean(1) + EPS),   # HF low/large
        fm[:, dorm].sum(1) / (fm.sum(1) + EPS),               # dorm fraction
    ])


def youden(neg, pos):
    best, thr, sgn = -1, 0.0, 1
    for s in (+1, -1):
        n, p = s * neg, s * pos
        for t in np.quantile(np.r_[n, p], np.linspace(.02, .98, 49)):
            j = (p > t).mean() - (n > t).mean()
            if j > best:
                best, thr, sgn = j, float(t), s
    return thr, sgn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["cifar100"])
    ap.add_argument("--id-tpr", type=float, default=0.95)
    ap.add_argument("--low-pct", type=float, default=0.10)
    args = ap.parse_args()

    import pandas as pd
    for ds in args.datasets:
        config.set_dataset(ds)
        FD = config.FEATURES_DIR
        models = [m for m in config.MODEL_ARCHS
                  if (FD / f"featfull_{m}_id.h5").exists()
                  and glob.glob(str(FD / f"featfull_{m}_adv_*.h5"))]
        if not models:
            print(f"[{ds}] no features — skip (needs Wave-2 finetune+extract)")
            continue
        print(f"\n################  END-TO-END RECALL — {ds}  ({len(models)} models)  ################")
        print(f"{'model':16}{'stage2':9}{'rec_ID':>8}{'rec_OOD':>9}{'rec_ADV':>9}{'bal_acc':>9}"
              f"{'s1_OODrec':>10}{'s1_ADVrec':>10}")
        rows = []
        for model in models:
            idd = load(str(FD / f"featfull_{model}_id.h5"))
            ood = [load(p) for p in glob.glob(str(FD / f"featfull_{model}_ood_*.h5"))]
            adv = [load(p) for p in glob.glob(str(FD / f"featfull_{model}_adv_*.h5"))]
            # Dormant band among ALIVE channels only (skip permanently-dead
            # first-conv channels, e.g. 25/64 in densenet121, which otherwise make
            # the dormant signature all-zero -> spurious AUROC=0.5).
            fm = idd["filter_means"]; C = fm.shape[1]; ch = fm.mean(0)
            alive = np.where(ch > 1e-4)[0]
            if len(alive) == 0:
                alive = np.arange(C)
            k = max(1, int(args.low_pct * len(alive)))
            order = alive[np.argsort(ch[alive])]; dorm, large = order[:k], order[-k:]

            def cat(splits, fn): return np.concatenate([fn(s) for s in splits])
            # stage-1 energy gate
            e_id = energy(idd["logits"])
            e_o = cat(ood, lambda s: energy(s["logits"]))
            e_a = cat(adv, lambda s: energy(s["logits"]))
            tau = np.quantile(e_id, args.id_tpr)
            f_id, f_o, f_a = e_id > tau, e_o > tau, e_a > tau
            s1_orec, s1_arec = f_o.mean(), f_a.mean()

            stat = {
                "linf":    (idd["inf_norms"], cat(ood, lambda s: s["inf_norms"]),
                            cat(adv, lambda s: s["inf_norms"])),
                "tv_dorm": (idd["filter_tv"][:, dorm].mean(1),
                            cat(ood, lambda s: s["filter_tv"][:, dorm].mean(1)),
                            cat(adv, lambda s: s["filter_tv"][:, dorm].mean(1))),
            }
            for s2, (vi, vo, va) in stat.items():
                thr, sgn = youden(vo[f_o], va[f_a]) if (f_o.sum() > 5 and f_a.sum() > 5) else (0, 1)
                def predict(flag, v):
                    p = np.zeros(len(v), int)
                    p[flag] = np.where(sgn * v[flag] > thr, 2, 1)
                    return p
                rec = [np.mean(predict(fl, v) == c) for c, (fl, v) in
                       zip((0, 1, 2), [(f_id, vi), (f_o, vo), (f_a, va)])]
                ba = np.mean(rec)
                rows.append(dict(model=model, stage2=s2, rec_ID=rec[0], rec_OOD=rec[1],
                                 rec_ADV=rec[2], bal_acc=ba, s1_OODrec=s1_orec, s1_ADVrec=s1_arec))
                print(f"{model:16}{s2:9}{rec[0]:>8.3f}{rec[1]:>9.3f}{rec[2]:>9.3f}{ba:>9.3f}"
                      f"{s1_orec:>10.3f}{s1_arec:>10.3f}")

            # panel: single-stage LDA 3-way (no gate), 5-fold CV
            Xs, ys = [], []
            for lab, src in [(0, [idd]), (1, ood), (2, adv)]:
                F = np.vstack([panel_feats(s, dorm, large) for s in src]); Xs.append(F)
                ys.append(np.full(len(F), lab))
            X = np.vstack(Xs); y = np.concatenate(ys); X = (X - X.mean(0)) / (X.std(0) + EPS)
            rec = {0: [], 1: [], 2: []}
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
                pr = LinearDiscriminantAnalysis().fit(X[tr], y[tr]).predict(X[te])
                for c in (0, 1, 2):
                    rec[c].append(np.mean(pr[y[te] == c] == c))
            r = [np.mean(rec[c]) for c in (0, 1, 2)]
            rows.append(dict(model=model, stage2="panel6", rec_ID=r[0], rec_OOD=r[1],
                             rec_ADV=r[2], bal_acc=np.mean(r), s1_OODrec=np.nan, s1_ADVrec=np.nan))
            print(f"{model:16}{'panel6':9}{r[0]:>8.3f}{r[1]:>9.3f}{r[2]:>9.3f}{np.mean(r):>9.3f}"
                  f"{'(no gate)':>20}")

        df = pd.DataFrame(rows)
        print(f"\n  === MEAN over {len(models)} models [{ds}] ===")
        g = df.groupby("stage2")[["rec_ID", "rec_OOD", "rec_ADV", "bal_acc"]].mean().round(3)
        print(g.to_string())
        out = config.ANALYSIS_DIR / f"cascade_recall_full_{ds}.csv"
        df.to_csv(out, index=False)
        print(f"  saved -> {out}")


if __name__ == "__main__":
    main()
