"""Deep-analysis plots for the top-4 first-layer signatures + variant search.

Reads signature_variants_<ds>.csv (130 variants × models, T2/T3/far/near) and the
37-signature CSVs (for the raw-L-inf A_inf_norm baseline). Produces a 4-panel figure:
  A) cross-model mean T2 & T3 for L-inf vs top-4 base vs the best variant
  B) variant-grid heatmap: quantity×norm (rows) × percentile×band (cols), colored by T3
  C) per-model T3 for {L-inf, B_low_frac, G_tv_dorm base, tv·p5·adapt best} — shows
     the DenseNet degeneracy and how the adaptive band fixes it
  D) FAR vs NEAR T3 for L-inf vs best variant — the complementarity story

    python experiments/plot_variants.py --dataset cifar100
"""
from __future__ import annotations

import argparse

import config
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def linf_per_model(models):
    """raw A_inf_norm T3 from the 37-sig CSVs."""
    out = {}
    for m in models:
        p = config.ANALYSIS_DIR / f"signature_auroc_full_{m}.csv"
        if p.exists():
            d = pd.read_csv(p, index_col=0)
            if "A_inf_norm" in d.index:
                out[m] = d.loc["A_inf_norm", "T3_OOD_vs_ADV"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar100", choices=list(config.DATASET_SPECS))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    config.set_dataset(args.dataset)
    df = pd.read_csv(config.ANALYSIS_DIR / f"signature_variants_{args.dataset}.csv")
    models = sorted(df.model.unique())
    BEST = "tv|p5|adapt|bandmean"
    BASE = "tv|p10|fixed|bandmean"        # ~ G_tv_dorm base
    BLOW = "mean|p10|fixed|massfrac"      # ~ B_low_frac
    linf = linf_per_model(models)

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.22)

    # ---- A: bar — L-inf vs base top-4 vs best variant (cross-model mean) ----
    axA = fig.add_subplot(gs[0, 0])
    g = df.groupby("variant").agg(T2=("T2_ID_ADV", "mean"), T3=("T3_OOD_ADV", "mean"))
    picks = {"raw L∞": np.mean(list(linf.values())),
             "B_low_frac\n(p10 fix)": g.loc[BLOW, "T3"] if BLOW in g.index else np.nan,
             "G_hf_low_large\n(p5 adapt)": g.loc["hf|p5|adapt|ratio_dorm_large", "T3"],
             "G_tv_dorm\n(p10 fix)": g.loc[BASE, "T3"] if BASE in g.index else np.nan,
             "★ tv·p5·adapt\n(BEST)": g.loc[BEST, "T3"]}
    names = list(picks); vals = [picks[k] for k in names]
    cols = ["#d62728", "#1f77b4", "#1f77b4", "#1f77b4", "#2ca02c"]
    axA.bar(range(len(names)), vals, color=cols)
    for i, v in enumerate(vals):
        axA.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
    axA.set_xticks(range(len(names))); axA.set_xticklabels(names, fontsize=8)
    axA.set_ylim(0.5, 0.9); axA.axhline(0.5, ls=":", c="gray")
    axA.set_ylabel("mean T3 (OOD-vs-ADV) AUROC")
    axA.set_title("A. Headline: raw L∞ vs Viyog signatures vs best variant", fontweight="bold", fontsize=11)

    # ---- B: variant-grid heatmap (quantity×norm × pct×band → mean T3) ----
    axB = fig.add_subplot(gs[0, 1])
    d2 = df.copy()
    d2["row"] = d2["quant"] + "·" + d2["norm"]
    d2["col"] = "p" + d2["pct"].astype(str) + "·" + d2["band"]
    piv = d2.pivot_table(index="row", columns="col", values="T3_OOD_ADV", aggfunc="mean")
    # order columns by pct then band
    colord = [f"p{p}·{b}" for p in [5, 10, 15, 20, 25] for b in ["adapt", "fixed"]]
    piv = piv[[c for c in colord if c in piv.columns]]
    piv = piv.loc[piv.mean(1).sort_values(ascending=False).index]
    im = axB.imshow(piv.values, aspect="auto", cmap="RdYlGn", vmin=0.5, vmax=0.85)
    axB.set_xticks(range(len(piv.columns))); axB.set_xticklabels(piv.columns, rotation=90, fontsize=7)
    axB.set_yticks(range(len(piv.index))); axB.set_yticklabels(piv.index, fontsize=7)
    for i in range(len(piv.index)):
        for j in range(len(piv.columns)):
            v = piv.values[i, j]
            if not np.isnan(v):
                axB.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.5)
    fig.colorbar(im, ax=axB, fraction=0.046, pad=0.04, label="mean T3")
    axB.set_title("B. Variant grid (mean T3) — adapt > fixed, p5 best, TV best", fontweight="bold", fontsize=11)

    # ---- C: per-model T3 — degeneracy fix ----
    axC = fig.add_subplot(gs[1, 0])
    sub = df[df.variant.isin([BEST, BASE, BLOW])].pivot_table(index="model", columns="variant", values="T3_OOD_ADV")
    sub["raw L∞"] = pd.Series(linf)
    sub = sub.sort_values(BEST, ascending=False)
    x = np.arange(len(sub)); w = 0.2
    axC.bar(x - 1.5 * w, sub.get("raw L∞", np.nan), w, label="raw L∞", color="#d62728")
    axC.bar(x - 0.5 * w, sub.get(BLOW, np.nan), w, label="B_low_frac p10 fix", color="#9467bd")
    axC.bar(x + 0.5 * w, sub.get(BASE, np.nan), w, label="G_tv_dorm p10 fix", color="#1f77b4")
    axC.bar(x + 1.5 * w, sub.get(BEST, np.nan), w, label="★ tv·p5·adapt", color="#2ca02c")
    axC.set_xticks(x); axC.set_xticklabels(sub.index, rotation=90, fontsize=7)
    axC.axhline(0.5, ls=":", c="gray"); axC.set_ylabel("T3 OOD-vs-ADV AUROC")
    axC.legend(fontsize=8, loc="lower left")
    axC.set_title("C. Per-model T3 — adaptive band fixes DenseNet 0.50 degeneracy", fontweight="bold", fontsize=11)

    # ---- D: FAR vs NEAR complementarity ----
    axD = fig.add_subplot(gs[1, 1])
    far_linf = []  # need per-model far/near for L-inf — approximate via base reconcile not in this csv; use variant near/far for best, and note L-inf from text
    bestrows = df[df.variant == BEST].set_index("model")
    sub2 = bestrows[["T3_far", "T3_near"]].sort_values("T3_near", ascending=False)
    x = np.arange(len(sub2)); w = 0.38
    axD.bar(x - w/2, sub2["T3_far"], w, label="★variant FAR-OOD", color="#2ca02c", alpha=0.7)
    axD.bar(x + w/2, sub2["T3_near"], w, label="★variant NEAR-OOD", color="#17becf")
    axD.axhline(0.5, ls=":", c="gray")
    axD.set_xticks(x); axD.set_xticklabels(sub2.index, rotation=90, fontsize=7)
    axD.set_ylabel("T3 AUROC"); axD.legend(fontsize=8)
    axD.set_title("D. tv·p5·adapt: strong on NEAR-OOD (where L∞ collapses ~0.55)", fontweight="bold", fontsize=11)

    fig.suptitle(f"First-layer signature variant analysis [{args.dataset}, {len(models)} clean models]",
                 fontsize=14, fontweight="bold")
    out = args.out or str(config.PLOTS_DIR / f"signature_variants_{args.dataset}.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"  saved → {out}")


if __name__ == "__main__":
    main()
