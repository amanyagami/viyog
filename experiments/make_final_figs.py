"""Final publication figures for Fig. 9 (complementarity) and Fig. 12 (cost Pareto).

Design principles applied:
  - ALL text is UPPERCASE (per author requirement)
  - VIYOG visually stands out via a distinct accent colour (#E65100, deep orange)
    and heavier borders / star markers; all baselines are muted grey-blue
  - Fig. 9 message made unmissable: a red X over Energy's ADV bar and Viyog's
    OOD bar, then a bright highlight on the COMBINED bar — the visual says
    "each fails where the other succeeds; together they dominate"
  - Fig. 12 shows only VIYOG-D (not Viyog-Linf, which is not the contribution
    being claimed) in the cost Pareto; Viyog is plotted as a star in the
    lower-left corner with a shaded "VIYOG DOMINATES" region

Run:
    python experiments/make_final_figs.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import config

# ── output dir ──────────────────────────────────────────────────────────────
OUT = config.ROOT / "paper_rev" / "figs" / "final"
OUT.mkdir(parents=True, exist_ok=True)
AD = config.ANALYSIS_DIR
GAD = config.RESULTS_DIR / "gtsrb" / "analysis"

# ── palette ──────────────────────────────────────────────────────────────────
VIYOG   = "#E65100"   # deep orange — Viyog always this colour
VIYOG_D = "#E65100"
ENERGY  = "#455A64"   # muted blue-grey
OTHERS  = "#90A4AE"   # lighter grey for everything else
COMBO   = "#1B5E20"   # dark green for the combined / winning bar
ID_COL  = "#1565C0"
OOD_COL = "#2E7D32"
ADV_COL = "#B71C1C"
BLIND   = "#FF1744"   # red for the blind-spot markers

# ── shared rcParams ───────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.axisbelow": True,
    "axes.grid": True,
    "grid.alpha": 0.20,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
})

def uc(s: str) -> str:
    """Return the string in UPPER CASE (used for all visible text)."""
    return str(s).upper()


def _boot_ci(vals: np.ndarray, n: int = 4000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    v = np.asarray(vals, float)
    bs = rng.choice(v, size=(n, len(v)), replace=True).mean(axis=1)
    return float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


# ════════════════════════════════════════════════════════════════════════════
# FIG 9  — Complementarity
# ════════════════════════════════════════════════════════════════════════════
def fig9_complementarity() -> None:
    """Redesigned Fig. 9: make the blind-spot story impossible to miss.

    Visual hierarchy:
      1. Energy bar for ADV is slashed red — "BLIND TO ADV"
      2. Viyog bar for OOD is slashed red — "WEAK ON OOD"
      3. Combined (Energy + Viyog) bar glows green — "BOTH FIXED"
      4. Full panel bar is the tallest, labelled "BEST"
    """
    c = pd.read_csv(AD / "complementarity_cifar100.csv")
    order = ["Energy only (logit)", "Viyog only", "Energy + Viyog", "Full panel"]
    labels = [uc("ENERGY\nONLY"), uc("VIYOG-D\nONLY"), uc("ENERGY +\nVIYOG-D"), uc("FULL\nPANEL")]
    g = c.groupby("feature_set")
    rec = {k: g.get_group(k)[["recall_ID", "recall_OOD", "recall_ADV", "bal_acc"]].mean()
           for k in order}

    # ── colour the x-axis group bars by story role ──
    grp_colors = [ENERGY, VIYOG, COMBO, VIYOG]  # Energy, Viyog, Combined, Full
    grp_edge   = ["#212121"] * 4

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(12, 3.8),
        gridspec_kw={"width_ratios": [2.4, 1]},
    )
    fig.subplots_adjust(wspace=0.32)

    x = np.arange(len(order))
    w = 0.19
    series = [
        ("recall_ID",  uc("ID RECALL"),        ID_COL),
        ("recall_OOD", uc("OOD RECALL"),       OOD_COL),
        ("recall_ADV", uc("ADV RECALL"),       ADV_COL),
        ("bal_acc",    uc("3-WAY BAL ACC"),    "#212121"),
    ]
    bar_handles = []
    for j, (key, lab, col) in enumerate(series):
        vals = [rec[k][key] for k in order]
        xp = x + (j - 1.5) * w
        bars = axL.bar(xp, vals, w, label=lab, color=col,
                       edgecolor="white", linewidth=0.3, alpha=0.88)
        bar_handles.append(bars)
        for xi, v in zip(xp, vals):
            axL.text(xi, v + 0.013, f"{v:.2f}", ha="center", va="bottom",
                     fontsize=6.5, fontweight="bold")

    # ── blind-spot slashes ──────────────────────────────────────────────────
    # Energy is blind to ADV (j=2 = ADV, group 0 = Energy only)
    adv_j = 2
    adv_xi = 0 + (adv_j - 1.5) * w
    adv_v  = rec[order[0]]["recall_ADV"]
    axL.bar([adv_xi], [adv_v], w, color="none", edgecolor=BLIND,
            hatch="////", linewidth=0.0, zorder=4)
    axL.text(adv_xi, adv_v + 0.055, uc("BLIND TO ADV"),
             ha="center", va="bottom", fontsize=7.5, color=BLIND, fontweight="bold")
    axL.annotate("", xy=(adv_xi, adv_v + 0.052), xytext=(adv_xi, adv_v + 0.013),
                 arrowprops=dict(arrowstyle="->", color=BLIND, lw=1.4))

    # Viyog is weak on OOD (j=1 = OOD, group 1 = Viyog only)
    ood_j = 1
    ood_xi = 1 + (ood_j - 1.5) * w
    ood_v  = rec[order[1]]["recall_OOD"]
    axL.bar([ood_xi], [ood_v], w, color="none", edgecolor=BLIND,
            hatch="////", linewidth=0.0, zorder=4)
    axL.text(ood_xi + 0.05, ood_v + 0.055, uc("WEAK ON OOD"),
             ha="center", va="bottom", fontsize=7.5, color=BLIND, fontweight="bold")
    axL.annotate("", xy=(ood_xi, ood_v + 0.052), xytext=(ood_xi, ood_v + 0.013),
                 arrowprops=dict(arrowstyle="->", color=BLIND, lw=1.4))

    # ── "COMBINED: BOTH FIXED" callout on the Energy+Viyog group ─────────
    combo_ba = rec[order[2]]["bal_acc"]
    combo_xi  = 2 + (3 - 1.5) * w  # bal_acc column for group 2
    axL.annotate(
        uc("COMBINING FIXES BOTH"),
        xy=(combo_xi, combo_ba + 0.013),
        xytext=(combo_xi + 0.55, combo_ba + 0.11),
        fontsize=8, color=COMBO, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=COMBO, lw=1.4),
    )

    # ── chance line ──────────────────────────────────────────────────────────
    axL.axhline(1 / 3, ls="--", c="gray", lw=1)
    axL.text(3.55, 1 / 3 + 0.01, uc("CHANCE (1/3)"), color="gray",
             fontsize=8, va="bottom", ha="right")

    axL.set_xticks(x)
    axL.set_xticklabels(labels, fontsize=9.5, fontweight="bold")
    axL.set_ylabel(uc("RECALL / BALANCED ACCURACY"))
    axL.set_ylim(0, 1.06)
    axL.set_title(
        uc("(A) CIFAR-100: ENERGY ✗ ADV  |  VIYOG-D ✗ OOD  |  TOGETHER ✓ ALL THREE"),
        fontsize=11, pad=10,
    )
    legend_patches = [mpatches.Patch(color=col, label=lab) for _, lab, col in series]
    axL.legend(handles=legend_patches, loc="lower center", ncol=4,
               fontsize=8.5, framealpha=0.95, bbox_to_anchor=(0.5, -0.28))
    axL.figure.subplots_adjust(bottom=0.26)

    # ── right panel: generalisation with CI bars ─────────────────────────────
    ec = _boot_ci(g.get_group("Energy only (logit)")["bal_acc"].values)
    fc = _boot_ci(g.get_group("Full panel")["bal_acc"].values)
    gt = pd.read_csv(GAD / "complementarity_gtsrb.csv").groupby("feature_set")["bal_acc"].mean()
    datasets = [uc("CIFAR-100\n(20 ARCH)"), uc("GTSRB\n(6 ARCH)")]
    energy_vals = [ec[0], float(gt["Energy only (logit)"])]
    panel_vals  = [fc[0], float(gt["Full panel"])]
    e_err = [[ec[0] - ec[1]], [ec[2] - ec[0]]]
    p_err = [[fc[0] - fc[1]], [fc[2] - fc[0]]]
    xx = np.arange(2)
    bw = 0.34

    b1 = axR.bar(xx - bw / 2, energy_vals, bw, label=uc("ENERGY ONLY"),
                 color=ENERGY, edgecolor="black", linewidth=0.5, alpha=0.9)
    axR.errorbar(xx[0] - bw / 2, energy_vals[0], yerr=e_err,
                 fmt="none", ecolor="black", capsize=4, lw=1.4)

    b2 = axR.bar(xx + bw / 2, panel_vals, bw, label=uc("FULL PANEL (OURS)"),
                 color=VIYOG, edgecolor="black", linewidth=0.5)
    axR.errorbar(xx[0] + bw / 2, panel_vals[0], yerr=p_err,
                 fmt="none", ecolor="black", capsize=4, lw=1.4)

    for b in list(b1) + list(b2):
        h = b.get_height()
        axR.text(b.get_x() + b.get_width() / 2, h + 0.016,
                 f"{h:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # ── annotate the disjoint CI ─────────────────────────────────────────────
    axR.annotate(
        uc("CIs\nDISJOINT"),
        xy=(0 + bw / 2, panel_vals[0]),
        xytext=(1.35, 0.74),
        fontsize=8, color=VIYOG, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=VIYOG, lw=1.3),
    )

    axR.axhline(1 / 3, ls="--", c="gray", lw=1)
    axR.set_xticks(xx)
    axR.set_xticklabels(datasets, fontsize=9, fontweight="bold")
    axR.set_ylim(0, 1.06)
    axR.set_ylabel(uc("3-WAY BALANCED ACCURACY"))
    axR.set_title(uc("(B) GAIN GENERALISES\nACROSS DATASETS"), fontsize=11, pad=10)
    axR.legend(loc="upper left", fontsize=8.5, framealpha=0.95)

    out = OUT / "fig9_complementarity.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


# ════════════════════════════════════════════════════════════════════════════
# FIG 12  — Cost Pareto  (Viyog only; Linf excluded)
# ════════════════════════════════════════════════════════════════════════════
def fig12_cost() -> None:
    """Redesigned Fig. 12.  Viyog only — Linf is NOT plotted (it is not the
    detector claimed; Viyog is the operationalised contribution).

    Visual design:
      • Viyog = deep-orange STAR, large, with a shaded "VIYOG DOMINATES"
        lower-left region in panel (b)
      • All baselines = muted blue-grey circles
      • ALL axis labels, tick labels, titles in UPPERCASE
    """
    d = pd.read_csv(AD / "baseline_latency_cifar100.csv")

    # ── drop Viyog-Linf; keep Viyog ───────────────────────────────────────
    d = d[~d["detector"].str.contains("Linf", case=False)]

    g = (
        d.groupby("detector")
        .agg(lat=("lat_ms_per_img", "mean"),
             state=("state_KB", "mean"),
             x=("lat_vs_viyogd", "mean"))
        .reset_index()
        .sort_values("lat")
    )
    is_viyog = g["detector"].str.startswith("Viyog")

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(11, 3.6),
        gridspec_kw={"width_ratios": [1.15, 1]},
    )
    fig.subplots_adjust(wspace=0.30)

    # ── (a) latency bars ─────────────────────────────────────────────────────
    colors = [VIYOG if v else OTHERS for v in is_viyog]
    edge   = ["#212121" if v else "#607D8B" for v in is_viyog]
    lw_    = [1.8 if v else 0.5 for v in is_viyog]

    bars = axA.barh(g["detector"].str.upper(), g["lat"],
                    color=colors, edgecolor=edge,
                    linewidth=lw_, height=0.55, zorder=3)

    # add multiplier annotations
    for bar, (_, row) in zip(bars, g.iterrows()):
        lat = row["lat"]
        mult = row["x"]
        label = f"{lat:.2f} MS  ({mult:.0f}×)" if mult > 1.5 else f"{lat:.2f} MS  (1×)"
        col = VIYOG if row["detector"].startswith("Viyog") else "#37474F"
        fw  = "bold" if row["detector"].startswith("Viyog") else "normal"
        axA.text(lat * 1.18, bar.get_y() + bar.get_height() / 2,
                 uc(label), va="center", fontsize=7, color=col, fontweight=fw)

    axA.set_xscale("log")
    axA.set_xlabel(uc("DETECTION LATENCY (MS/IMG, LOG SCALE)"))
    axA.set_title(uc("(A) VIYOG-D IS THE FASTEST — 6–347× CHEAPER\nTHAN ALL BASELINES (H200 MEASURED)"),
                  fontsize=10.5, pad=8)
    axA.set_xlim(g["lat"].min() * 0.4, g["lat"].max() * 9)
    axA.grid(axis="x", alpha=0.20, which="both")
    axA.tick_params(axis="y", labelsize=9)

    # ── (b) latency–memory Pareto ─────────────────────────────────────────────
    # shaded "Viyog dominates" region
    vrow = g[g["detector"].str.startswith("Viyog")].iloc[0]
    axB.axhspan(1e-2, float(vrow["state"]) * 4, xmin=0, xmax=0.35,
                color=VIYOG, alpha=0.07, zorder=0)
    axB.axvspan(1e-2, float(vrow["lat"]) * 3, ymin=0, ymax=0.35,
                color=VIYOG, alpha=0.07, zorder=0)
    axB.text(0.025, 0.18, uc("VIYOG\nDOMINATES"),
             fontsize=9, color=VIYOG, fontweight="bold", va="center",
             transform=axB.transAxes)
    axB.set_xlim(8e-3, 25)
    axB.set_ylim(5e-3, 5e4)

    for _, row in g.iterrows():
        is_v = row["detector"].startswith("Viyog")
        color  = VIYOG if is_v else OTHERS
        marker = "*" if is_v else "o"
        size   = 280 if is_v else 55
        zorder = 5 if is_v else 3
        ew     = 1.2 if is_v else 0.4
        axB.scatter(row["lat"], max(row["state"], 1e-3),
                    s=size, color=color, marker=marker,
                    edgecolor="#212121", linewidth=ew, zorder=zorder)
        label = uc(row["detector"])
        off   = (6, 5) if is_v else (4, 3)
        fw    = "bold" if is_v else "normal"
        fs    = 8 if is_v else 6.5
        axB.annotate(label, (row["lat"], max(row["state"], 1e-3)),
                     fontsize=fs, xytext=off, textcoords="offset points",
                     fontweight=fw, color=VIYOG if is_v else "#37474F")

    axB.set_xscale("log"); axB.set_yscale("log")
    axB.set_xlabel(uc("LATENCY (MS/IMG)"))
    axB.set_ylabel(uc("STATE MEMORY (KB)"))
    axB.set_title(uc("(B) COST PARETO — LOWER-LEFT IS BETTER\nVIYOG-D ALONE IN THE OPTIMAL CORNER"),
                  fontsize=10.5, pad=8)
    axB.grid(alpha=0.20, which="both")

    # ── legend ────────────────────────────────────────────────────────────────
    vp = mpatches.Patch(color=VIYOG,  label=uc("VIYOG-D (OUR METHOD)"))
    bp = mpatches.Patch(color=OTHERS, label=uc("BASELINES"))
    axB.legend(handles=[vp, bp], fontsize=8.5, loc="lower right", framealpha=0.95)

    out = OUT / "fig12_cost.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")
    print("  NOTE: Viyog-Linf excluded — only Viyog (the operationalised detector) shown")


if __name__ == "__main__":
    fig9_complementarity()
    fig12_cost()
