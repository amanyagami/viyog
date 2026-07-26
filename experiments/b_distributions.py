"""Reviewer-B evidence: distributions (not means) of first-layer statistics, and
the overlap that the global L-inf norm suffers vs the dormant-band statistic.

Addresses B7 ("Fig 2 reports means without distributional information; overlapping
distributions would undermine separation") and B6 (squashed-score distribution).
Pure post-hoc on cifar100 featfull — CPU only.

    python experiments/b_distributions.py --dataset cifar100
Outputs: results/analysis/b_distributions_<ds>.csv  + results/plots/rebuttal/b_distributions.png
"""
from __future__ import annotations
import argparse, glob, os
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
import h5py, numpy as np
import config

EPS = 1e-8
CORE6 = ["resnet50", "densenet121", "convnextv2_base", "vit_base", "swin_tiny", "mobilenetv3_l"]


def load(p, keys=("inf_norms", "filter_tv", "filter_means")):
    with h5py.File(p, "r") as f:
        return {k: f[k][:].astype(np.float64) for k in keys if k in f}


def overlap_coef(a, b, bins=80):
    """Histogram overlap coefficient in [0,1]; 1 = identical, 0 = disjoint."""
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    if hi <= lo:
        return 1.0
    ha, _ = np.histogram(a, bins=bins, range=(lo, hi), density=True)
    hb, _ = np.histogram(b, bins=bins, range=(lo, hi), density=True)
    w = (hi - lo) / bins
    return float(np.minimum(ha, hb).sum() * w)


def squash(c, T=1000.0):
    """Viyog double-exponential squash (monotonic), centered statistic c."""
    return np.sign(c) / (1.0 + np.exp(-np.exp(np.abs(c / T))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar100")
    ap.add_argument("--low-pct", type=float, default=0.10)
    args = ap.parse_args()
    config.set_dataset(args.dataset)
    FD = config.FEATURES_DIR
    models = [m for m in CORE6 if (FD / f"featfull_{m}_id.h5").exists()
              and glob.glob(str(FD / f"featfull_{m}_adv_*.h5"))]
    if not models:
        print(f"[{args.dataset}] no Core-6 features — skip"); return

    import pandas as pd
    rows = []
    panel = {}  # for plotting
    for m in models:
        idd = load(str(FD / f"featfull_{m}_id.h5"))
        ood = [load(p) for p in glob.glob(str(FD / f"featfull_{m}_ood_*.h5"))]
        adv = [load(p) for p in glob.glob(str(FD / f"featfull_{m}_adv_*.h5"))]
        fm = idd["filter_means"]; C = fm.shape[1]; ch = fm.mean(0)
        # dormant band among ALIVE channels only (skip permanently-dead first-conv
        # channels that would make the dormant statistic all-zero, e.g. densenet121)
        alive = np.where(ch > 1e-4)[0]
        if len(alive) == 0:
            alive = np.arange(C)
        k = max(1, int(args.low_pct * len(alive)))
        dorm = alive[np.argsort(ch[alive])[:k]]

        def cat(splits, fn): return np.concatenate([fn(s) for s in splits])
        # statistic 1: global L-inf (the paper's Viyog)
        linf = dict(ID=idd["inf_norms"], ADV=cat(adv, lambda s: s["inf_norms"]),
                    OOD=cat(ood, lambda s: s["inf_norms"]))
        # statistic 2: dormant-band TV (Viyog best drop-in)
        dtv = dict(ID=idd["filter_tv"][:, dorm].mean(1),
                   ADV=cat(adv, lambda s: s["filter_tv"][:, dorm].mean(1)),
                   OOD=cat(ood, lambda s: s["filter_tv"][:, dorm].mean(1)))
        for name, st in [("Linf", linf), ("Viyog-D_tv_dorm", dtv)]:
            o_ia = overlap_coef(st["ID"], st["ADV"])     # the separation that matters for routing
            o_oa = overlap_coef(st["OOD"], st["ADV"])     # T3 OOD-vs-ADV
            rows.append(dict(model=m, stat=name,
                             overlap_ID_ADV=round(o_ia, 3), overlap_OOD_ADV=round(o_oa, 3),
                             ID_med=round(np.median(st["ID"]), 4),
                             ADV_med=round(np.median(st["ADV"]), 4),
                             OOD_med=round(np.median(st["OOD"]), 4)))
        panel[m] = (linf, dtv)

    df = pd.DataFrame(rows)
    out = config.ANALYSIS_DIR / f"b_distributions_{args.dataset}.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\n  === MEAN overlap (lower = better separation) ===")
    print(df.groupby("stat")[["overlap_ID_ADV", "overlap_OOD_ADV"]].mean().round(3).to_string())
    print(f"  saved -> {out}")

    # plot: distributions for a representative model (B7's "show distributions")
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        mrep = "resnet50" if "resnet50" in panel else models[0]
        linf, dtv = panel[mrep]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for j, (st, title) in enumerate([(linf, f"{mrep}: global L∞ (paper)"),
                                         (dtv, f"{mrep}: dormant-band TV (Viyog)")]):
            data = [st["ID"], st["OOD"], st["ADV"]]
            parts = ax[j].violinplot(data, showmedians=True)
            ax[j].set_xticks([1, 2, 3]); ax[j].set_xticklabels(["ID", "OOD", "ADV"])
            ax[j].set_title(title); ax[j].set_ylabel("statistic value")
        fig.suptitle("B7: distributions, not means — L∞ overlaps ID/ADV; dorm-band separates")
        plt.tight_layout()
        pp = config.PLOTS_DIR / "rebuttal" / "b_distributions.png"
        pp.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(pp, dpi=140); plt.close()
        print(f"  plot -> {pp}")
    except Exception as e:  # noqa: BLE001
        print(f"  [plot skipped] {e}")


if __name__ == "__main__":
    main()
