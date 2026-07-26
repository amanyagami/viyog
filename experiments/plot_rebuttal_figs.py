"""Publication-quality (PDF, 300 dpi) figures that REPLACE the rebuttal tables.

Emits to paper_rev/figs/rebuttal/:
  fig_deployable.pdf       per-task deployable AUROC; direction-inconsistent bars
                           hatched + marked X (not deployable); value labels.
  fig_complementarity.pdf  (a) per-class recall + 3-way bal-acc by detector on
                           CIFAR-100; (b) Full-panel vs Energy-only 3-way bal-acc
                           on CIFAR-100 AND GTSRB with bootstrap 95% CIs.

Run:  python experiments/plot_rebuttal_figs.py
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import config

AD = config.ANALYSIS_DIR
GAD = config.RESULTS_DIR / "gtsrb" / "analysis"
OUT = config.ROOT / "paper_rev" / "figs" / "rebuttal"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 12, "font.family": "serif", "axes.axisbelow": True,
    "axes.grid": True, "grid.alpha": 0.25, "savefig.bbox": "tight",
    "savefig.dpi": 300, "pdf.fonttype": 42,
})
C = {"T1": "#4C72B0", "T2": "#DD8452", "T3": "#55A868",
     "ID": "#4C72B0", "OOD": "#55A868", "ADV": "#DD8452", "BAL": "#2b2b2b"}


def _boot_ci(vals, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(vals, float)
    bs = rng.choice(v, size=(n, len(v)), replace=True).mean(axis=1)
    return float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


# ============================ Figure 1 ============================
def fig_deployable():
    s = pd.read_csv(AD / "full_eval_cifar100_summary.csv").set_index(["method", "task"])
    methods = ["Viyog_Linf", "ViyogD_tv_dorm", "ViyogD_hf_dorm", "MSP",
               "MaxLogit", "Energy", "Entropy", "GEN", "KLMatching"]
    labels = ["Viyog $L_\\infty$\n(original)", "Viyog\n(tv_dorm)", "Viyog\n(hf_dorm)",
              "MSP", "MaxLogit", "Energy", "Entropy", "GEN", "KL-M"]
    tasks = ["T1", "T2", "T3"]
    tasknames = ["T1  ID-vs-OOD", "T2  ID-vs-ADV", "T3  OOD-vs-ADV (deploy)"]
    avail = [m for m in methods if (m, "T2") in s.index]
    labels = [labels[methods.index(m)] for m in avail]
    methods = avail
    x = np.arange(len(methods)); w = 0.26
    fig, ax = plt.subplots(figsize=(11.5, 2.95))
    for j, t in enumerate(tasks):
        vals, dirc = [], []
        for m in methods:
            if (m, t) in s.index:
                vals.append(float(s.loc[(m, t), "deployable"]))
                dirc.append(bool(s.loc[(m, t), "dir_consistent"]))
            else:
                vals.append(np.nan); dirc.append(True)
        xpos = x + (j - 1) * w
        bars = ax.bar(xpos, vals, w, label=tasknames[j], color=C[t],
                      edgecolor="black", linewidth=0.4)
        for xi, v, dc in zip(xpos, vals, dirc):
            if np.isnan(v):
                continue
            if not dc:                      # direction flips across models -> not deployable
                ax.bar(xi, v, w, color="none", edgecolor="black",
                       hatch="////", linewidth=0.0)
                ax.text(xi, v + 0.012, r"$\times$", ha="center", va="bottom",
                        color="#b22222", fontsize=13, fontweight="bold")
            else:
                ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=7.4)
    ax.axhline(0.5, ls="--", c="gray", lw=1)
    ax.text(len(methods) - 0.5, 0.505, "chance", color="gray", fontsize=9, va="bottom", ha="right")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("Deployable AUROC\n(credited only if direction-consistent)")
    ax.set_ylim(0.3, 1.02)
    ax.set_title("Per-task separation on CIFAR-100 (20 architectures): "
                 "no single detector wins all three", fontsize=12.5, pad=8)
    ax.legend(loc="upper center", ncol=3, fontsize=10, framealpha=0.95,
              bbox_to_anchor=(0.5, -0.22))
    fig.subplots_adjust(bottom=0.24)
    fig.savefig(OUT / "fig_deployable.pdf"); plt.close(fig)
    print("wrote", OUT / "fig_deployable.pdf")


# ============================ Figure 2 ============================
def fig_complementarity():
    c = pd.read_csv(AD / "complementarity_cifar100.csv")
    order = ["Energy only (logit)", "Viyog only", "Energy + Viyog", "Full panel"]
    short = ["Energy\nonly", "Viyog\nonly", "Energy +\nViyog", "All detectors\n(LDA panel)"]
    g = c.groupby("feature_set")
    rec = {k: g.get_group(k)[["recall_ID", "recall_OOD", "recall_ADV", "bal_acc"]].mean()
           for k in order}

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 3.15),
                                   gridspec_kw={"width_ratios": [2.1, 1]})
    # --- left: per-class recall + 3-way bal-acc ---
    x = np.arange(len(order)); w = 0.2
    # highlight the groups that include our Viyog statistic
    axL.axvspan(0.5, 3.5, color="#2ca02c", alpha=0.06, zorder=0)
    axL.text(2.0, 0.06, "configurations using Viyog (ours)", ha="center", va="bottom",
             fontsize=7.5, color="#1a7a44", style="italic", fontweight="bold",
             transform=axL.transAxes)
    series = [("recall_ID", "ID recall", C["ID"]), ("recall_OOD", "OOD recall", C["OOD"]),
              ("recall_ADV", "ADV recall", C["ADV"]), ("bal_acc", "3-way bal-acc", C["BAL"])]
    for j, (key, lab, col) in enumerate(series):
        vals = [rec[k][key] for k in order]
        xp = x + (j - 1.5) * w
        axL.bar(xp, vals, w, label=lab, color=col, edgecolor="black", linewidth=0.4)
        for xi, v in zip(xp, vals):
            axL.text(xi, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    axL.axhline(1 / 3, ls="--", c="gray", lw=1)
    axL.text(3.5, 0.34, "chance", color="gray", fontsize=9, va="bottom", ha="right")
    # annotate the two opposite blind spots
    axL.annotate("blind to ADV", xy=(0 + 0.5 * w, rec[order[0]]["recall_ADV"]),
                 xytext=(0.1, 0.16), fontsize=9, color="#b22222",
                 arrowprops=dict(arrowstyle="->", color="#b22222", lw=1.2))
    axL.annotate("weak on OOD", xy=(1 - 0.5 * w, rec[order[1]]["recall_OOD"]),
                 xytext=(1.05, 0.14), fontsize=9, color="#b22222",
                 arrowprops=dict(arrowstyle="->", color="#b22222", lw=1.2))
    axL.set_xticks(x); axL.set_xticklabels(short, fontsize=10)
    axL.set_ylabel("recall / balanced accuracy"); axL.set_ylim(0, 1.02)
    axL.set_title("(a) CIFAR-100: the two families fail on opposite boundaries;\n"
                  "combining them separates all three classes", fontsize=11)
    axL.legend(loc="lower center", ncol=2, fontsize=9, framealpha=0.95)

    # --- right: 3-way bal-acc generalization, Energy-only vs Full panel, both datasets ---
    ec = _boot_ci(g.get_group("Energy only (logit)")["bal_acc"].values)
    fc = _boot_ci(g.get_group("Full panel")["bal_acc"].values)
    gt = pd.read_csv(GAD / "complementarity_gtsrb.csv").groupby("feature_set")["bal_acc"].mean()
    datasets = ["CIFAR-100\n(20 arch)", "GTSRB"]
    energy = [ec[0], float(gt["Energy only (logit)"])]
    panel = [fc[0], float(gt["Full panel"])]
    e_err = [[ec[0] - ec[1]], [ec[2] - ec[0]]]
    p_err = [[fc[0] - fc[1]], [fc[2] - fc[0]]]
    xx = np.arange(len(datasets)); bw = 0.34
    b1 = axR.bar(xx - bw / 2, energy, bw, label="Energy only", color=C["T1"],
                 edgecolor="black", linewidth=0.4)
    axR.errorbar(xx[0] - bw / 2, energy[0], yerr=e_err, fmt="none", ecolor="black", capsize=3)
    b2 = axR.bar(xx + bw / 2, panel, bw, label="Full panel (ours)", color=C["T3"],
                 edgecolor="black", linewidth=0.4)
    axR.errorbar(xx[0] + bw / 2, panel[0], yerr=p_err, fmt="none", ecolor="black", capsize=3)
    for b in list(b1) + list(b2):
        axR.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012,
                 f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=8.5)
    axR.axhline(1 / 3, ls="--", c="gray", lw=1)
    axR.set_xticks(xx); axR.set_xticklabels(datasets, fontsize=10)
    axR.set_ylim(0, 1.02); axR.set_ylabel("3-way balanced accuracy")
    axR.set_title("(b) Generalizes across datasets\n(CIs disjoint on CIFAR-100)", fontsize=11)
    axR.legend(loc="upper right", fontsize=9, framealpha=0.95)
    fig.savefig(OUT / "fig_complementarity.pdf"); plt.close(fig)
    print("wrote", OUT / "fig_complementarity.pdf")
    print(f"  energy CI={ec}, panel CI={fc}")


# ============================ Figure 3 ============================
def fig_adaptive():
    import glob
    modes = ["pgd", "normpresv", "dormaware", "hfaware", "allaware"]
    mlab = ["plain PGD", "norm-\npreserving", "dorm-aware\n(targets TV)",
            "HF-aware\n(targets HF)", "both-aware\n(TV+HF)"]
    agg = {m: {"succ": [], "dorm": [], "hf": []} for m in modes}
    for f in glob.glob(str(AD / "adaptive_cifar100_*.csv")):
        d = pd.read_csv(f)
        for m in modes:
            sub = d[d["mode"] == m]
            if len(sub):
                r = sub.iloc[-1]
                agg[m]["succ"].append(r.attack_success)
                agg[m]["dorm"].append(r.auroc_dorm)
                agg[m]["hf"].append(r.auroc_hf)
    dorm = [np.mean(agg[m]["dorm"]) for m in modes]
    hf = [np.mean(agg[m]["hf"]) for m in modes]
    succ = [np.mean(agg[m]["succ"]) for m in modes]
    nmod = len(agg["pgd"]["dorm"])

    fig, ax = plt.subplots(figsize=(8.6, 3.05))
    x = np.arange(len(modes)); w = 0.34
    b1 = ax.bar(x - w / 2, dorm, w, label="dorm-band AUROC (OOD-vs-ADV)",
                color=C["OOD"], edgecolor="black", linewidth=0.4)
    b2 = ax.bar(x + w / 2, hf, w, label="high-freq AUROC (OOD-vs-ADV)",
                color=C["ADV"], edgecolor="black", linewidth=0.4)
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012,
                f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(0.5, ls="--", c="gray", lw=1)
    ax.text(len(modes) - 0.5, 0.51, "chance", color="gray", fontsize=8.5, va="bottom", ha="right")
    # attack success as a line on a twin axis
    ax2 = ax.twinx()
    ax2.plot(x, succ, "k--o", lw=1.6, ms=6, label="attack success")
    for xi, s_ in zip(x, succ):
        ax2.text(xi, s_ - 0.06, f"{s_:.2f}", ha="center", va="top", fontsize=8, color="black")
    ax2.set_ylim(0, 1.05); ax2.set_ylabel("attack success rate")
    ax.set_ylim(0, 1.05); ax.set_ylabel("detector AUROC (OOD-vs-ADV)")
    ax.set_xticks(x); ax.set_xticklabels(mlab, fontsize=9)
    ax.set_title(f"Signature-aware adaptive attack (mean over {nmod} architectures, "
                 "max $\\lambda$)", fontsize=11.5)
    # callouts
    ax.annotate("norm-preserving:\n0% evasion", xy=(1, dorm[1]), xytext=(1.0, 0.18),
                fontsize=8.5, color="#2c7a2c", ha="center",
                arrowprops=dict(arrowstyle="->", color="#2c7a2c", lw=1.1))
    ax.annotate("attacking TV\nleaves HF up", xy=(2 + w / 2, hf[2]), xytext=(2.0, 0.2),
                fontsize=8, color="#b8860b", ha="center",
                arrowprops=dict(arrowstyle="->", color="#b8860b", lw=1.0))
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper center", ncol=3, fontsize=8.3,
              framealpha=0.95, bbox_to_anchor=(0.5, -0.22))
    fig.subplots_adjust(bottom=0.24)
    fig.savefig(OUT / "fig_adaptive.pdf"); plt.close(fig)
    print("wrote", OUT / "fig_adaptive.pdf",
          f"| succ={[round(s,2) for s in succ]} dorm={[round(d,2) for d in dorm]}")


# ===================== Figure 4 (refreshed earlier plot) =====================
def fig_main_auroc():
    """Refreshed, consistent version of the headline 11-method AUROC bar chart
    (Average_AUROC_11methods). Values are the published per-method means from
    Table I (OOD-vs-ADV separation across all 20 model-dataset combinations)."""
    FIG = config.ROOT / "paper_rev" / "figs"
    data = [("Viyog (V1K)", 92.38), ("Energy", 67.50), ("MaxLogit", 66.86),
            ("Entropy", 66.20), ("MSP", 64.51), ("MCD", 64.51), ("KNN", 64.12),
            ("KL-Match", 63.14), ("OpenMax", 59.86), ("Maha", 55.30), ("ODIN", 45.52)]
    names = [d[0] for d in data]; vals = [d[1] for d in data]
    fig, ax = plt.subplots(figsize=(8.6, 2.75))
    colors = ["#2e7d32"] + ["#6f8faf"] * (len(vals) - 1)
    bars = ax.bar(names, vals, color=colors, edgecolor="black", linewidth=0.4)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.1f}",
                ha="center", va="bottom", fontsize=8.5,
                fontweight="bold" if v == 92.38 else "normal")
    ax.axhline(50, ls="--", c="gray", lw=1)
    ax.text(len(vals) - 0.4, 51, "chance", color="gray", fontsize=9, va="bottom", ha="right")
    # gain annotation
    ax.annotate("", xy=(0, 92.38), xytext=(0, 67.5),
                arrowprops=dict(arrowstyle="<->", color="#2e7d32", lw=1.4))
    ax.text(0.35, 80, "+24.9\nAUROC", color="#2e7d32", fontsize=9, fontweight="bold", va="center")
    ax.set_ylabel("Average AUROC (%)"); ax.set_ylim(40, 100)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=28, ha="right", fontsize=9.5)
    ax.set_title("OOD-vs-ADV separation across all 20 model–dataset combinations",
                 fontsize=12)
    fig.savefig(FIG / "fig_main_auroc.pdf"); plt.close(fig)
    print("wrote", FIG / "fig_main_auroc.pdf")


# ===================== Figure 5 (refreshed earlier plot) =====================
def fig_score():
    """Refreshed Viyog score function (replaces VIYOG_FIGS/output*.png).
    Viyog(t;T) = sign(t) / (1 + exp(-exp(|t|/T)))."""
    FIG = config.ROOT / "paper_rev" / "figs"
    t = np.linspace(-6, 6, 1200)
    def viyog(t, T):
        s = np.sign(t)
        return s / (1 + np.exp(-np.exp(np.abs(t) / T)))
    fig, ax = plt.subplots(figsize=(8.0, 2.5))
    ax.plot(t, viyog(t, 1), lw=2.2, color="#c0392b", label="$T=1$ (sharp: confident $\\pm1$)")
    ax.plot(t, viyog(t, 1000), lw=2.2, color="#2c6fbb",
            label="$T=1000$ (graded: threshold-robust)")
    ax.axvline(0, ls="--", c="gray", lw=1); ax.axhline(0, ls=":", c="gray", lw=0.8)
    ax.fill_betweenx([-1.1, 1.1], -6, 0, color="#c0392b", alpha=0.05)
    ax.fill_betweenx([-1.1, 1.1], 0, 6, color="#2e7d32", alpha=0.05)
    ax.text(-4.5, -0.9, "ADV region\n(score $\\to-1$)", fontsize=9, color="#c0392b", ha="center")
    ax.text(4.5, 0.9, "OOD region\n(score $\\to+1$)", fontsize=9, color="#2e7d32", ha="center")
    ax.set_xlabel("centred statistic $t(x)=\\|f_0(x)\\|_\\infty-\\mu_{\\mathrm{ID}}$")
    ax.set_ylabel("Viyog$(x;T)$"); ax.set_ylim(-1.12, 1.12)
    ax.set_title("Bounded routing score: decision boundary at $t=0$", fontsize=12)
    ax.legend(loc="center left", fontsize=9, framealpha=0.95)
    fig.savefig(FIG / "fig_score.pdf"); plt.close(fig)
    print("wrote", FIG / "fig_score.pdf")


if __name__ == "__main__":
    fig_deployable()
    fig_complementarity()
    fig_adaptive()
    fig_main_auroc()
    fig_score()
