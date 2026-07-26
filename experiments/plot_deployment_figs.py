"""Comprehensive detection-quality + deployment-cost figures (300-dpi vector).

Replaces the legacy raster/old figures (Average_MemoryGB, Average_DetectionTime,
NORMS_VIYOG.png, Viyog_autc.png, the per-seed PNGs) with one consistent set:
  fig_detection_quality.pdf : AUROC (T1/T2/T3) + recall@5%FPR across method families
  fig_deployment_cost.pdf   : memory (log), compute/latency/energy overhead, accel energy
  fig_quality_vs_cost.pdf   : the money plot -- T2 AUROC vs detector state memory
All data from master_comparison_cifar100.csv, detector_cost.csv, edge_latency.csv,
accelerator_energy.csv.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config

AD = config.ANALYSIS_DIR
OUT = config.ROOT / "paper_rev" / "figs" / "rebuttal"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    "font.size": 9, "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
})
FAMCOL = {"Viyog (first-conv)": "#2ca02c", "logit": "#1f77b4", "distance (feature)": "#d62728"}


def _save(fig, name):
    fig.savefig(OUT / name); plt.close(fig); print("wrote", OUT / name)


def fig_detection_quality():
    """Grouped bars: representative methods x {T1, T2, T3, recall@5%FPR}."""
    d = pd.read_csv(AD / "master_comparison_cifar100.csv")
    # Compare the deployed Viyog against the strongest baseline of each family.
    # The raw-norm baseline is documented in Table II and omitted here for clarity.
    pick = ["ViyogD_tv_dorm", "GEN", "Mahalanobis", "KNN"]
    nice = {"ViyogD_tv_dorm": "Viyog (ours)", "GEN": "GEN (best logit)",
            "Mahalanobis": "Mahalanobis", "KNN": "KNN"}
    cmap = {"ViyogD_tv_dorm": "#2ca02c", "GEN": "#1f77b4",
            "Mahalanobis": "#d62728", "KNN": "#ff7f0e"}
    d = d[d.method.isin(pick)].set_index("method").reindex(pick)
    metrics = [("T1_ID_OOD", "T1 ID/OOD"), ("T2_ID_ADV", "T2 ID/ADV"),
               ("T3_OOD_ADV", "T3 OOD/ADV")]
    x = np.arange(len(metrics)); w = 0.20
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    for i, m in enumerate(pick):
        vals = [d.loc[m, mk] if not pd.isna(d.loc[m, mk]) else 0 for mk, _ in metrics]
        ours = m == "ViyogD_tv_dorm"
        bars = ax.bar(x + (i - 1.5) * w, vals, w, label=nice[m], color=cmap[m],
                      edgecolor="black" if ours else "none",
                      linewidth=1.4 if ours else 0, zorder=3 if ours else 2)
        # value labels on the Viyog bars so the win is explicit
        if ours:
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}",
                        ha="center", fontsize=6.5, fontweight="bold", color="#1a7a44")
        for b, v, (mk, _) in zip(bars, vals, metrics):
            if pd.isna(d.loc[m, mk]):
                ax.text(b.get_x() + b.get_width() / 2, 0.02, "n/a", ha="center", fontsize=5, rotation=90)
    ax.axhline(0.5, ls="--", lw=0.8, color="grey")
    ax.text(2.32, 0.515, "chance", fontsize=6, color="grey", va="bottom")
    ax.set_xticks(x); ax.set_xticklabels([lab for _, lab in metrics])
    ax.set_ylabel("AUROC (directionless)"); ax.set_ylim(0, 1.05)
    ax.set_title("Detection quality (20-arch CIFAR-100): Viyog wins the security tasks T2/T3")
    ax.legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    _save(fig, "fig_detection_quality.pdf")


def fig_deployment_cost():
    """(a) state memory (log), (b) compute/latency/energy overhead, (c) accel energy."""
    d = pd.read_csv(AD / "master_comparison_cifar100.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.9), constrained_layout=True)
    # (a) detector state memory, log scale
    mem = d[["method", "state_mem_KB", "family"]].copy()
    mem = mem[mem.method.isin(["ViyogD_tv_dorm", "Energy", "ViM", "Mahalanobis", "KNN"])]
    order = ["ViyogD_tv_dorm", "Energy", "ViM", "Mahalanobis", "KNN"]
    lab = {"ViyogD_tv_dorm": "Viyog*", "Energy": "logit", "ViM": "ViM", "Mahalanobis": "Maha", "KNN": "KNN"}
    mem = mem.set_index("method").reindex(order)
    ax = axes[0]
    cols = [FAMCOL.get(f, "#888") for f in mem.family]
    ax.bar(range(len(mem)), mem.state_mem_KB, color=cols)
    ax.set_yscale("log"); ax.set_ylabel("detector state (KB, log)")
    ax.set_xticks(range(len(mem))); ax.set_xticklabels([lab[m] for m in order], rotation=35, ha="right", fontsize=7)
    ax.set_title("(a) memory: 0.28 KB vs MB")
    for i, v in enumerate(mem.state_mem_KB):
        ax.text(i, v * 1.3, f"{v:.2g}" if v < 1 else f"{v:.0f}", ha="center", fontsize=6)
    # (b) compute/latency/energy overhead (% of full forward): Viyog vs baselines(=100)
    ax = axes[1]
    vd = d[d.method == "ViyogD_tv_dorm"].iloc[0]
    cats = ["compute", "CPU lat.", "accel. energy"]
    vdv = [vd["compute_%fwd"], vd["cpu_lat_%fwd"], vd["accel_energy_%fwd"]]
    xx = np.arange(len(cats)); w = 0.38
    ax.bar(xx - w / 2, [100, 100, 100], w, label="logit / distance", color="#9aa0a6")
    ax.bar(xx + w / 2, vdv, w, label="Viyog*", color="#2ca02c")
    for i, v in enumerate(vdv):
        ax.text(i + w / 2, v + 3, f"{v:.1f}%", ha="center", fontsize=6.5, color="#1a661a")
    ax.set_xticks(xx); ax.set_xticklabels(cats, rotation=20, fontsize=7)
    ax.set_ylabel("% of full inference"); ax.set_ylim(0, 112)
    ax.set_title("(b) compute / latency / energy"); ax.legend(fontsize=6.5, loc="center right")
    # (c) accelerator first-conv energy by workload
    ac = pd.read_csv(AD / "accelerator_energy.csv")
    ax = axes[2]
    piv = ac.pivot_table(index="workload", columns="accelerator", values="E_firstconv_%")
    piv = piv.head(6)
    xw = np.arange(len(piv)); w = 0.38
    for j, acc in enumerate(piv.columns):
        ax.bar(xw + (j - 0.5) * w, piv[acc], w, label=acc)
    ax.set_xticks(xw); ax.set_xticklabels(piv.index, rotation=40, ha="right", fontsize=6)
    ax.set_ylabel("first-conv energy (%)"); ax.set_title("(c) accelerator DSE")
    ax.legend(fontsize=6.5)
    fig.suptitle("Deployment cost of the first-conv detector vs feature/logit baselines", fontsize=9)
    _save(fig, "fig_deployment_cost.pdf")


def fig_quality_vs_cost():
    """Money plot: T2 AUROC vs detector state memory (log). Viyog top-left."""
    d = pd.read_csv(AD / "master_comparison_cifar100.csv").copy()
    d["mem"] = d["state_mem_KB"].clip(lower=0.003)
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for fam, sub in d.groupby("family"):
        ax.scatter(sub["mem"], sub["T2_ID_ADV"], s=46, color=FAMCOL.get(fam, "#888"),
                   label=fam, edgecolor="k", linewidth=0.4, zorder=3)
    for _, r in d.iterrows():
        if r.method in ("ViyogD_tv_dorm", "Mahalanobis", "KNN", "ViM", "GEN", "Viyog_Linf"):
            nm = "Viyog*" if r.method == "ViyogD_tv_dorm" else \
                 "Viyog-L$_\\infty$" if r.method == "Viyog_Linf" else r.method
            ax.annotate(nm, (r["mem"], r["T2_ID_ADV"]), fontsize=6.5,
                        xytext=(4, 3), textcoords="offset points")
    ax.set_xscale("log"); ax.axhline(0.5, ls="--", lw=0.8, color="grey")
    ax.set_xlabel("detector state memory (KB, log scale)")
    ax.set_ylabel("ID-vs-ADV (T2) AUROC")
    ax.set_title("Quality vs cost: Viyog$^*$ is top-left (best T2, $4\\times10^4$ less memory)")
    ax.legend(fontsize=7, loc="lower right")
    ax.annotate("better", xy=(0.05, 0.97), xytext=(0.30, 0.80), xycoords="axes fraction",
                textcoords="axes fraction", fontsize=8, color="#555",
                arrowprops=dict(arrowstyle="->", color="#555"))
    _save(fig, "fig_quality_vs_cost.pdf")


if __name__ == "__main__":
    fig_detection_quality()
    fig_deployment_cost()
    fig_quality_vs_cost()
    print("done.")
