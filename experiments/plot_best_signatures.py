"""Visualize the 37-signature battery — show which signature separates best.

Two panels:
  (left)  heatmap of T3 (OOD-vs-ADV, the headline task) AUROC for every signature
          (rows, grouped by family A–J) × near-SOTA model (cols), best cell ringed.
  (right) ranked bars of mean-over-models AUROC for T2 and T3, NEW (G/H/J)
          signatures highlighted, the original L∞ (A_inf_norm) marked for contrast.

    python experiments/plot_best_signatures.py --dataset cifar100 \
        --models convnextv2_base swin_tiny vit_base densenet121
"""
from __future__ import annotations

import argparse

import config
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar100", choices=list(config.DATASET_SPECS))
    ap.add_argument("--models", nargs="+",
                    default=["convnextv2_base", "swin_tiny", "vit_base", "densenet121"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    config.set_dataset(args.dataset)

    dfs = {}
    for m in args.models:
        p = config.ANALYSIS_DIR / f"signature_auroc_full_{m}.csv"
        if p.exists():
            dfs[m] = pd.read_csv(p, index_col=0)
    models = list(dfs)
    sigs = list(dfs[models[0]].index)
    fam = lambda s: s.split("_")[0]
    new = {s for s in sigs if fam(s) in ("G", "H", "J")}

    T3 = pd.DataFrame({m: dfs[m]["T3_OOD_vs_ADV"] for m in models})
    T2 = pd.DataFrame({m: dfs[m]["T2_ID_vs_ADV"] for m in models})
    T3["MEAN"], T2["MEAN"] = T3.mean(1), T2.mean(1)
    order = T3.sort_values("MEAN", ascending=False).index.tolist()

    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], height_ratios=[1, 1],
                          wspace=0.32, hspace=0.30)

    # ---- left: T3 heatmap (signatures × models) ----
    axh = fig.add_subplot(gs[:, 0])
    H = T3.loc[order, models].values
    im = axh.imshow(H, aspect="auto", cmap="RdYlGn", vmin=0.5, vmax=1.0)
    axh.set_xticks(range(len(models)))
    axh.set_xticklabels([m.replace("_", "\n") for m in models], fontsize=8)
    axh.set_yticks(range(len(order)))
    axh.set_yticklabels([f"{'★' if s in new else ' '}{s}" for s in order], fontsize=7)
    axh.set_title("T3  OOD-vs-ADV AUROC  (★ = new signature)", fontsize=11, weight="bold")
    for i in range(len(order)):
        for j in range(len(models)):
            axh.text(j, i, f"{H[i, j]:.2f}", ha="center", va="center", fontsize=6,
                     color="black")
    # ring the per-model best
    for j, m in enumerate(models):
        bi = order.index(T3[m].idxmax())
        axh.add_patch(Rectangle((j - 0.5, bi - 0.5), 1, 1, fill=False,
                                edgecolor="blue", lw=2))
    fig.colorbar(im, ax=axh, fraction=0.046, pad=0.04, label="AUROC")

    # ---- right-top: T3 ranked bars ----
    def bars(ax, S, title):
        top = S.sort_values("MEAN", ascending=False).head(12)
        cols = ["#d62728" if s in new else "#1f77b4" for s in top.index]
        cols = ["#2ca02c" if s == top.index[0] else c for s, c in zip(top.index, cols)]
        ax.barh(range(len(top))[::-1], top["MEAN"], color=cols)
        ax.set_yticks(range(len(top))[::-1])
        ax.set_yticklabels(top.index, fontsize=7)
        ax.axvline(0.856, ls="--", c="gray", lw=1)
        ax.text(0.856, len(top) - 0.5, " 0.856\n B_low_frac\n ceiling", fontsize=6, color="gray")
        for i, v in enumerate(top["MEAN"][::-1]):
            ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=6)
        ax.set_xlim(0.5, 1.02)
        ax.set_xlabel("mean AUROC over models")
        ax.set_title(title, fontsize=10, weight="bold")

    bars(fig.add_subplot(gs[0, 1]), T3, "Top-12 by T3 (OOD-vs-ADV)  — green=winner, red=NEW")
    bars(fig.add_subplot(gs[1, 1]), T2, "Top-12 by T2 (ID-vs-ADV)  — green=winner, red=NEW")

    fig.suptitle(f"Viyog signature battery [{args.dataset}] — 37 signatures × {len(models)} near-SOTA models",
                 fontsize=13, weight="bold")
    out = args.out or str(config.PLOTS_DIR / f"best_signatures_{args.dataset}.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"  saved → {out}")


if __name__ == "__main__":
    main()
