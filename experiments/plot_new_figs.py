"""Generate NEW rebuttal figures from completed-but-unused experiments.

Targets the lowest-scoring reviewer concerns with evidence that already exists:
  fig_seed_ci.pdf        -- B3/C-w2: evaluation-resampling stability on BOTH datasets
  fig_adaptive_strong.pdf-- C-w1/D-d3: the stronger held-out-basis adaptive attacker
  fig_embedded_cost.pdf  -- A-w5/B1: honest full-panel first-conv cost range

Run: .venv/bin/python experiments/plot_new_figs.py
Writes to ../paper_rev/figs/rebuttal/ (next to the existing rebuttal figs).
"""
from __future__ import annotations
import glob
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8,
    "axes.labelsize": 8, "legend.fontsize": 6.5, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "figure.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
})

A = "/mnt/data1/asing725/viyog/results/analysis"
G = "/mnt/data1/asing725/viyog/results/gtsrb/analysis"
OUT = "/mnt/data1/asing725/viyog/paper_rev/figs/rebuttal"
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(0)


def boot_ci(x, n=2000):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    bs = np.array([RNG.choice(x, len(x), replace=True).mean() for _ in range(n)])
    return x.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


# ---------------------------------------------------------------- fig 1: seed CI
def fig_seed_ci():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
    dsets = [("CIFAR-100", f"{A}/pipeline_seeds_cifar100.csv"),
             ("GTSRB", f"{G}/pipeline_seeds_gtsrb.csv")]
    dets = [("Viyog_D*", "Viyog (panel)", "#1b7837"), ("Viyog_Linf", r"Global $L_\infty$", "#b2182b")]
    tasks = ["T2", "T3"]
    for ax, (dname, path) in zip(axes, dsets):
        df = pd.read_csv(path)
        nseed = df["seed"].nunique()
        nmod = df["model"].nunique()
        xpos = np.arange(len(tasks))
        w = 0.36
        for j, (dkey, dlab, col) in enumerate(dets):
            sub = df[df.detector == dkey]
            means, los, his = [], [], []
            for t in tasks:
                m, lo, hi = boot_ci(sub[t].values)
                means.append(m); los.append(m - lo); his.append(hi - m)
            ax.bar(xpos + (j - 0.5) * w, means, w, yerr=[los, his], capsize=2.5,
                   color=col, alpha=0.85, label=dlab, error_kw=dict(lw=0.8))
            for k, t in enumerate(tasks):
                ax.text(xpos[k] + (j - 0.5) * w, means[k] + his[k] + 0.02,
                        f"{means[k]:.2f}", ha="center", va="bottom", fontsize=6)
        ax.axhline(0.5, ls=":", c="grey", lw=0.7)
        ax.set_xticks(xpos)
        ax.set_xticklabels(["T2\n(ID-vs-ADV)", "T3\n(OOD-vs-ADV)"])
        ax.set_ylim(0.4, 1.05)
        ax.set_title(f"{dname}  ({nmod} archs $\\times$ {nseed} seeds)")
        if ax is axes[0]:
            ax.set_ylabel("AUROC (mean, 95\\% CI)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(loc="lower left", frameon=False, ncol=1)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_seed_ci.pdf", bbox_inches="tight")
    plt.close(fig)
    # report numbers for the caption
    for dname, path in dsets:
        df = pd.read_csv(path)
        for dkey in ("Viyog_D*",):
            sub = df[df.detector == dkey]
            for t in tasks:
                m, lo, hi = boot_ci(sub[t].values)
                print(f"  [seed_ci] {dname} {dkey} {t}: {m:.3f} [{lo:.3f},{hi:.3f}] hw={max(m-lo,hi-m):.3f}")


# ----------------------------------------------------- fig 2: strong adaptive
def fig_adaptive_strong():
    files = sorted(glob.glob(f"{A}/adaptive_strong_cifar100_*.csv"))
    modes = ["pgd", "normpresv", "dormaware", "hfaware", "allaware"]
    mlab = ["PGD\n(base)", "norm-\npresv", "dorm-\naware", "hf-\naware", "both-\naware"]
    rows = {m: {"dorm": [], "hf": [], "succ": []} for m in modes}
    models = []
    for f in files:
        name = os.path.basename(f).replace("adaptive_strong_cifar100_", "").replace(".csv", "")
        models.append(name)
        d = pd.read_csv(f)
        for m in modes:
            sub = d[d["mode"] == m]
            if len(sub):
                r = sub.iloc[-1]  # max lambda (worst case for detector)
                rows[m]["dorm"].append(r.auroc_dorm)
                rows[m]["hf"].append(r.auroc_hf)
                rows[m]["succ"].append(r.attack_success)
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    x = np.arange(len(modes))
    w = 0.36
    dm = [np.nanmean(rows[m]["dorm"]) for m in modes]
    hm = [np.nanmean(rows[m]["hf"]) for m in modes]
    de = [np.nanstd(rows[m]["dorm"]) for m in modes]
    he = [np.nanstd(rows[m]["hf"]) for m in modes]
    ax.bar(x - w / 2, dm, w, yerr=de, capsize=2, color="#1b7837", alpha=0.85,
           label="dorm-band", error_kw=dict(lw=0.7))
    ax.bar(x + w / 2, hm, w, yerr=he, capsize=2, color="#e08214", alpha=0.9,
           label="high-freq", error_kw=dict(lw=0.7))
    ax.axhline(0.5, ls=":", c="grey", lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels(mlab)
    ax.set_ylabel("OOD-vs-ADV AUROC"); ax.set_ylim(0.4, 1.0)
    ax2 = ax.twinx()
    sm = [np.nanmean(rows[m]["succ"]) for m in modes]
    ax2.plot(x, sm, "k--o", ms=3, lw=1.0, label="attack success")
    ax2.set_ylabel("attack success"); ax2.set_ylim(0.4, 1.05)
    ax.set_title(f"Strong held-out-basis attacker ({len(models)} archs)")
    ax.spines[["top"]].set_visible(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower left", frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_adaptive_strong.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  [adaptive_strong] models={models}")
    for m in modes:
        print(f"    {m:10s} dorm={np.nanmean(rows[m]['dorm']):.3f} hf={np.nanmean(rows[m]['hf']):.3f} succ={np.nanmean(rows[m]['succ']):.3f}")


# ----------------------------------------------------- fig 3: embedded cost
def fig_embedded_cost():
    d = pd.read_csv(f"{A}/systems_cifar100.csv").sort_values("firstconv_mac_ratio_%")
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    is_t = d["arch"].str.contains("vit|swin", case=False, na=False) | d["model"].str.contains("vit|swin", case=False)
    colors = ["#762a83" if t else "#1b7837" for t in is_t]
    y = np.arange(len(d))
    ax.barh(y, d["firstconv_mac_ratio_%"], color=colors, alpha=0.85)
    ax.set_yticks(y); ax.set_yticklabels(d["model"], fontsize=5.5)
    ax.set_xlabel("first-conv MACs (\\% of model)")
    med = d[~is_t]["firstconv_mac_ratio_%"].median()
    ax.axvline(med, ls="--", c="grey", lw=0.8)
    ax.text(med + 1, 0.5, f"CNN median {med:.1f}\\%", fontsize=6, color="grey")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#1b7837", label="CNN"), Patch(color="#762a83", label="patch-embed (ViT/Swin)")],
              loc="lower right", frameon=False, fontsize=6)
    ax.set_title("First-conv cost across 20 architectures")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_embedded_cost.pdf", bbox_inches="tight")
    plt.close(fig)
    cnn = d[~is_t]["firstconv_mac_ratio_%"]
    print(f"  [embedded] CNN range {cnn.min():.2f}-{cnn.max():.2f}% median {cnn.median():.2f}%; "
          f"transformers {d[is_t]['firstconv_mac_ratio_%'].min():.1f}-{d[is_t]['firstconv_mac_ratio_%'].max():.1f}%")


if __name__ == "__main__":
    fig_seed_ci()
    fig_adaptive_strong()
    fig_embedded_cost()
    print("\nWrote 3 figures to", OUT)
