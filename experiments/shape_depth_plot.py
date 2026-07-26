"""Plot shape-vs-raw AUROC by layer depth (D2 figure).

Reads shape_depth_<dataset>.csv and renders a figure showing, per model and on
average, the ID-vs-ADV (T2) AUROC of the deployed dormant-band SHAPE statistic
and the raw L-inf statistic at each depth. The narrative: shape peaks at the
first conv (depth 0) and decays; raw L-inf is weakest there and recovers
mid-depth but stays below the first-conv shape read.

    python experiments/shape_depth_plot.py --dataset cifar100
"""
from __future__ import annotations

import argparse

import config
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar100")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    csv = args.csv or str(config.ANALYSIS_DIR / f"shape_depth_{args.dataset}.csv")
    df = pd.read_csv(csv)
    # mean over attacks per (model, depth)
    g = df.groupby(["model", "depth"])[["T2_shape", "T2_linf"]].mean().reset_index()
    models = list(dict.fromkeys(df["model"]))

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.0), gridspec_kw={"width_ratios": [1.45, 1]})

    # (a) per-model shape curves + the mean
    ax = axes[0]
    NICE_M = {
        "resnet50": "ResNet-50", "resnet101": "ResNet-101", "resnet152": "ResNet-152",
        "densenet121": "DenseNet-121", "densenet161": "DenseNet-161",
        "mobilenetv3_l": "MobileNetV3-L", "convnextv2_base": "ConvNeXtV2-B",
        "swin_tiny": "Swin-T", "vit_base": "ViT-B",
    }
    for m in models:
        d = g[g.model == m].sort_values("depth")
        ax.plot(d.depth, d.T2_shape, marker="o", ms=3, lw=1.1, alpha=0.55,
                label=NICE_M.get(m, m))
    mean = g.groupby("depth")[["T2_shape", "T2_linf"]].mean().reset_index()
    ax.plot(mean.depth, mean.T2_shape, color="k", lw=2.4, marker="s", ms=4, label="mean (shape)")
    ax.axvline(0, color="green", ls=":", lw=1, alpha=0.7)
    ax.set_xlabel("layer depth (0 = first conv)")
    ax.set_ylabel("ID-vs-ADV AUROC (T2)")
    ax.set_title("(a) Dormant-band shape statistic by depth")
    ax.set_ylim(0.45, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=5.5, ncol=2, loc="lower left")

    # (b) mean shape vs mean raw L-inf
    ax = axes[1]
    ax.plot(mean.depth, mean.T2_shape, color="C2", lw=2.2, marker="o", ms=4, label="shape $V$")
    ax.plot(mean.depth, mean.T2_linf, color="C3", lw=2.2, ls="--", marker="x", ms=5, label=r"raw $L_\infty$")
    ax.axvline(0, color="green", ls=":", lw=1, alpha=0.7)
    ax.annotate("first conv\n(deployed)", xy=(0, mean.T2_shape.iloc[0]),
                xytext=(1.1, 0.83), fontsize=6,
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_xlabel("layer depth")
    ax.set_title("(b) Shape vs. raw norm (mean)")
    ax.set_ylim(0.45, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="lower right")

    fig.tight_layout()
    out = args.out or "figs/rebuttal/fig_shape_depth.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"saved -> {out}")
    # also print the key numbers for the paper text
    print("\nmean T2 by depth:")
    print(mean.round(3).to_string(index=False))
    fc = mean[mean.depth == 0].iloc[0]
    best_shape = mean.T2_shape.max(); best_linf_depth = int(mean.loc[mean.T2_linf.idxmax(), "depth"])
    print(f"\nfirst-conv shape={fc.T2_shape:.3f}  raw-Linf={fc.T2_linf:.3f}")
    print(f"best shape (any depth)={best_shape:.3f}  best raw-Linf at depth {best_linf_depth}={mean.T2_linf.max():.3f}")


if __name__ == "__main__":
    main()
