"""Full-pipeline graphs: ID/OOD/ADV separation + accuracy-vs-cost comparison.

Renders, from the multi-seed pipeline results + the cost CSVs:
  A. Viyog* score separation (ID vs OOD vs ADV) for representative models,
  B. full-pipeline metrics per stage-2 detector with seed error bars,
  C. accuracy (T3) vs detector cost — memory, compute, CPU latency — showing Viyog
     on the favorable frontier (high accuracy, low cost).

    python experiments/exp_pipeline_plots.py --dataset cifar100
"""
from __future__ import annotations

import argparse
import glob
import os

import config
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# per-detector deployment cost (from detector_cost.csv / edge_latency.csv): Viyog runs
# only the first conv; logit/distance baselines need a full forward (+ their state).
COST = {
    "Viyog_D*":    {"mem_KB": 0.28,     "gmacs": 0.049, "lat_pct": 3.5,  "c": "#c0392b"},
    "Viyog_Linf":  {"mem_KB": 0.01,     "gmacs": 0.049, "lat_pct": 3.5,  "c": "#e67e22"},
    "viyog_dorm":  {"mem_KB": 0.28,     "gmacs": 0.049, "lat_pct": 3.5,  "c": "#d35400"},
    "s2_Energy":   {"mem_KB": 0.004,    "gmacs": 1.59,  "lat_pct": 100., "c": "#5b6770"},
    "s2_MSP":      {"mem_KB": 0.004,    "gmacs": 1.59,  "lat_pct": 100., "c": "#7f8c8d"},
    "s2_MaxLogit": {"mem_KB": 0.004,    "gmacs": 1.59,  "lat_pct": 100., "c": "#95a5a6"},
    "Mahalanobis": {"mem_KB": 7588.,    "gmacs": 1.59,  "lat_pct": 100., "c": "#2c3e50"},
    "KNN":         {"mem_KB": 25600.,   "gmacs": 1.59,  "lat_pct": 100., "c": "#34495e"},
    "ViM":         {"mem_KB": 7300.,    "gmacs": 1.59,  "lat_pct": 100., "c": "#16a085"},
}


def _dist_t3(adir, dataset):
    """Real per-method mean T3 for distance baselines (from baselines_feature_*.csv),
    replacing the old hard-coded 0.52/0.51 placeholders. Falls back to placeholders
    if the refresh CSV is absent."""
    p = adir / f"baselines_feature_{dataset}.csv"
    if not p.exists():
        return {"Mahalanobis": 0.52, "KNN": 0.51, "ViM": 0.55}
    d = pd.read_csv(p)
    g = d.groupby("method")["T3_OOD_ADV"].mean()
    return {k: float(g.get(k, np.nan)) for k in ("Mahalanobis", "KNN", "ViM")}


def adaptive_band(prof, p=5.0):
    live = np.where(prof > 1e-4 * prof.max())[0]
    if len(live) < 4:
        live = np.arange(len(prof))
    order = live[np.argsort(prof[live])]
    return order[: max(1, int(round(p / 100.0 * len(order))))]


def viyogd(d, dorm):
    return d["filter_tv"][:, dorm].mean(1)


def load(p):
    with h5py.File(p, "r") as f:
        return {k: f[k][:].astype(np.float64) for k in ("filter_means", "filter_tv")}


def panel_separation(ax, FD, model):
    idp = FD / f"featfull_{model}_id.h5"
    idd = load(str(idp))
    dorm = adaptive_band(idd["filter_means"].mean(0))
    sid = viyogd(idd, dorm)
    so = np.concatenate([viyogd(load(p), dorm) for p in sorted(glob.glob(str(FD / f"featfull_{model}_ood_*.h5")))[:5]])
    sa = np.concatenate([viyogd(load(p), dorm) for p in sorted(glob.glob(str(FD / f"featfull_{model}_adv_*.h5")))])
    lo, hi = np.percentile(np.r_[sid, so, sa], [1, 99])
    bins = np.linspace(lo, hi, 60)
    for s, lab, c in [(sa, "ADV", "#c0392b"), (sid, "ID", "#2c7fb8"), (so, "OOD", "#27ae60")]:
        ax.hist(s, bins=bins, density=True, alpha=0.55, label=lab, color=c)
    ax.set_title(f"A. Viyog* separation — {model}", fontsize=10)
    ax.set_xlabel("Viyog* score (first-conv dorm TV)")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)


def panel_metrics(ax, df):
    agg = df.groupby("detector").agg(T2=("T2", "mean"), T3=("T3", "mean"),
                                     OOD=("e2e_OOD_recall", "mean"), IDFP=("ID_FP_to_ADV", "mean"),
                                     T3sd=("T3", "std")).reindex(
        ["Viyog_D*", "viyog_dorm", "s2_Energy", "s2_MSP", "Viyog_Linf"])
    x = np.arange(len(agg))
    w = 0.2
    ax.bar(x - 1.5 * w, agg.T2, w, label="T2 (ID-ADV)", color="#2c7fb8")
    ax.bar(x - 0.5 * w, agg.T3, w, yerr=agg.T3sd, capsize=2, label="T3 (OOD-ADV)", color="#27ae60")
    ax.bar(x + 0.5 * w, agg.OOD, w, label="e2e OOD recall", color="#8e44ad")
    ax.bar(x + 1.5 * w, agg.IDFP, w, label="ID-FP→ADV (↓ safer)", color="#c0392b")
    ax.set_xticks(x)
    ax.set_xticklabels(agg.index, rotation=20, ha="right", fontsize=8)
    ax.set_title("B. Full-pipeline metrics by stage-2 detector (±seed std)", fontsize=10)
    ax.set_ylabel("score")
    ax.legend(fontsize=7, ncol=2)
    ax.set_ylim(0, 1.05)


def panel_acc_cost(ax, df, xkey, xlabel, dist_t3, logx=True):
    t3 = df.groupby("detector")["T3"].mean()
    for det, cfg in COST.items():
        y = t3.get(det, dist_t3.get(det, np.nan))
        if y != y:
            continue
        ax.scatter(cfg[xkey], y, s=90, color=cfg["c"], edgecolor="k", linewidth=0.5, zorder=3)
        ax.annotate(det.replace("s2_", "").replace("Viyog_", "V-"), (cfg[xkey], y),
                    fontsize=7, xytext=(4, 3), textcoords="offset points")
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("T3 (OOD-vs-ADV AUROC)")
    ax.axhline(0.5, ls=":", color="gray", lw=0.8)
    ax.grid(alpha=0.25)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar100", choices=list(config.DATASET_SPECS))
    ap.add_argument("--model", default="resnet50")
    args = ap.parse_args()
    config.set_dataset(args.dataset)
    A = config.ANALYSIS_DIR  # per-dataset path — must be read AFTER set_dataset
    FD = config.FEATURES_DIR
    df = pd.read_csv(A / f"pipeline_seeds_{args.dataset}.csv")
    plt.rcParams.update({"font.size": 9})

    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    panel_separation(axs[0, 0], FD, args.model)
    panel_metrics(axs[0, 1], df)
    dist_t3 = _dist_t3(A, args.dataset)
    panel_acc_cost(axs[1, 0], df, "mem_KB", "detector state memory (KB, log)", dist_t3)
    axs[1, 0].set_title("C. Accuracy vs detector memory (top-left = best)", fontsize=10)
    panel_acc_cost(axs[1, 1], df, "gmacs", "compute per routed sample (GMACs, log)", dist_t3)
    axs[1, 1].set_title("D. Accuracy vs compute (top-left = best)", fontsize=10)
    fig.suptitle(f"Viyog* full pipeline: separation + accuracy-vs-cost ({args.dataset}, ±seeds)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = str(config.PLOTS_DIR / f"pipeline_summary_{args.dataset}.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"saved → {out}")

    # standalone separation figure across 4 models
    f2, a2 = plt.subplots(1, 4, figsize=(18, 3.6))
    for ax, mdl in zip(a2, ["resnet50", "convnextv2_base", "densenet121", "vit_base"]):
        if (FD / f"featfull_{mdl}_id.h5").exists():
            panel_separation(ax, FD, mdl)
    f2.tight_layout()
    out2 = str(config.PLOTS_DIR / f"separation_{args.dataset}.png")
    f2.savefig(out2, dpi=140, bbox_inches="tight")
    print(f"saved → {out2}")


if __name__ == "__main__":
    main()
