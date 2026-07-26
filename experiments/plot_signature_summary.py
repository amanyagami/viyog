"""Visualize the 28-signature × 3-task AUROC battery (mean over 4 models + per-model).

Reads results/analysis/signature_auroc_<model>.csv and renders:
  1. signatures_heatmap_mean.png  — 28 signatures (grouped by family) × 3 tasks
  2. signatures_t3_per_model.png   — per-model ADV-vs-OOD for the top signatures
  3. signatures_task_space.png     — ID/ADV vs ID/OOD scatter, coloured by ADV/OOD
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ANALYSIS = Path("/mnt/data1/asing725/viyog/results/analysis")
PLOTS = Path("/mnt/data1/asing725/viyog/results/plots")
MODELS = ["convnextv2_base", "efficientnetv2_l", "swin_tiny", "vit_base"]
SHORT = {"convnextv2_base": "convnext", "efficientnetv2_l": "effnet*",
         "swin_tiny": "swin", "vit_base": "vit"}
TASKS = [("T2_ID_vs_ADV", "ID vs ADV"),
         ("T1_ID_vs_OOD", "ID vs OOD"),
         ("T3_OOD_vs_ADV", "ADV vs OOD")]
FAMILY = {"A": "Norm", "B": "Group-ratio", "C": "Sparsity",
          "D": "Crest", "E": "Profile-dev", "F": "Logit/energy"}


def load() -> dict[str, dict[str, dict[str, float]]]:
    """{model: {signature: {col: auroc}}}"""
    out: dict[str, dict[str, dict[str, float]]] = {}
    for m in MODELS:
        rows: dict[str, dict[str, float]] = {}
        for r in csv.DictReader(open(ANALYSIS / f"signature_auroc_{m}.csv")):
            rows[r["signature"]] = {k: float(v) for k, v in r.items() if k != "signature"}
        out[m] = rows
    return out


def main() -> None:
    data = load()
    sigs = list(data[MODELS[0]].keys())  # preserve family order from CSV
    # mean over models per (signature, task)
    mean = {s: {col: float(np.mean([data[m][s][col] for m in MODELS]))
                for col, _ in TASKS} for s in sigs}

    # ---- relabel Viyog statistic ----
    disp = {s: (s + " ★" if s in ("A_inf_norm",) else s) for s in sigs}

    # ============================================================ #
    # 1. HEATMAP (mean over 4 models)
    # ============================================================ #
    M = np.array([[mean[s][col] for col, _ in TASKS] for s in sigs])
    fig, ax = plt.subplots(figsize=(7.2, 11))
    im = ax.imshow(M, cmap="RdYlGn", vmin=0.5, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(TASKS)))
    ax.set_xticklabels([t[1] for t in TASKS], fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(sigs)))
    ax.set_yticklabels([disp[s] for s in sigs], fontsize=9, family="monospace")
    for i in range(len(sigs)):
        for j in range(len(TASKS)):
            v = M[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="black" if 0.62 < v < 0.92 else "white", fontsize=8)
    # family separators + labels
    prev = None
    for i, s in enumerate(sigs):
        fam = s[0]
        if fam != prev:
            if i:
                ax.axhline(i - 0.5, color="black", lw=1.4)
            ax.text(-1.55, i, FAMILY.get(fam, fam), rotation=90, va="top",
                    ha="center", fontsize=9, fontweight="bold", color="#333")
            prev = fam
    ax.set_title("First-layer signature AUROC  (mean over 4 models)\n"
                 "green = separable · 0.5 = chance", fontsize=12, fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("AUROC", fontsize=10)
    fig.tight_layout()
    fig.savefig(PLOTS / "signatures_heatmap_mean.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # ============================================================ #
    # 2. PER-MODEL  ADV vs OOD  for top-N signatures
    # ============================================================ #
    topN = sorted(sigs, key=lambda s: -mean[s]["T3_OOD_vs_ADV"])[:10]
    x = np.arange(len(topN))
    w = 0.2
    fig, ax = plt.subplots(figsize=(12, 5.5))
    colors = {"convnext": "#1f77b4", "effnet*": "#ff7f0e",
              "swin": "#2ca02c", "vit": "#d62728"}
    for k, m in enumerate(MODELS):
        vals = [data[m][s]["T3_OOD_vs_ADV"] for s in topN]
        ax.bar(x + (k - 1.5) * w, vals, w, label=SHORT[m], color=colors[SHORT[m]])
    ax.axhline(0.5, color="grey", ls="--", lw=1, label="chance")
    ax.set_xticks(x)
    ax.set_xticklabels(topN, rotation=35, ha="right", family="monospace", fontsize=9)
    ax.set_ylabel("ADV-vs-OOD AUROC", fontsize=11)
    ax.set_ylim(0.45, 1.0)
    ax.set_title("Headline task (ADV vs OOD): top-10 signatures, per model\n"
                 "convnext/effnet favour B_low_frac · vit/swin favour F_energy",
                 fontsize=12, fontweight="bold")
    ax.legend(ncol=5, fontsize=10, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "signatures_t3_per_model.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # ============================================================ #
    # 3. TASK-SPECIALISATION SCATTER  (why no single winner)
    # ============================================================ #
    fig, ax = plt.subplots(figsize=(9, 7.5))
    xs = [mean[s]["T2_ID_vs_ADV"] for s in sigs]
    ys = [mean[s]["T1_ID_vs_OOD"] for s in sigs]
    cs = [mean[s]["T3_OOD_vs_ADV"] for s in sigs]
    sc = ax.scatter(xs, ys, c=cs, cmap="viridis", vmin=0.5, vmax=0.9,
                    s=140, edgecolor="black", linewidth=0.6, zorder=3)
    ax.axhline(0.5, color="grey", lw=0.8); ax.axvline(0.5, color="grey", lw=0.8)
    highlight = {"E_mahalanobis", "B_low_frac", "B_ratio_low_large", "F_energy",
                 "F_max_logit", "A_inf_norm", "C_gini", "D_crest_mean", "A_peak_l2"}
    for s, xx, yy in zip(sigs, xs, ys):
        if s in highlight:
            ax.annotate(s, (xx, yy), fontsize=8.5, family="monospace",
                        xytext=(5, 4), textcoords="offset points", zorder=4)
    ax.set_xlabel("ID vs ADV  AUROC  (adversarial detector axis)", fontsize=11)
    ax.set_ylabel("ID vs OOD  AUROC  (OOD detector axis)", fontsize=11)
    ax.set_title("Signature task-specialisation\n"
                 "colour = ADV-vs-OOD (headline). Top-right = strong on both base tasks",
                 fontsize=12, fontweight="bold")
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("ADV vs OOD AUROC (T3)", fontsize=10)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOTS / "signatures_task_space.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    print("Wrote 3 figures to", PLOTS)
    for p in ["signatures_heatmap_mean", "signatures_t3_per_model", "signatures_task_space"]:
        print("  ", PLOTS / f"{p}.png")


if __name__ == "__main__":
    main()
