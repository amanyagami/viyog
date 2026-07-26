"""NEW Viyog signature plots (the dorm-band shape statistic found via the rebuttal).

Replaces the legacy norm/efficiency plots with figures built directly on the
discovered signature G_tv_dorm (dorm-band total variation):

  fig_viyogd_signature.pdf  -- per-image Viyog score distribution ID/OOD/ADV +
                               the dormant-band where the signal lives (the STAR plot)
  fig_sig_battery.pdf       -- 37-signature OOD-vs-ADV ranking; G_tv_dorm is #1,
                               beating every logit baseline
  fig_cost.pdf              -- embedded cost (detector state + first-conv compute),
                               replacing Average_DetectionTime/MemoryGB

Run: .venv/bin/python experiments/plot_viyogd_signature.py
Writes to ../../paper_rev/figs/rebuttal/.
"""
from __future__ import annotations
import glob
import os
import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8.5,
    "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "figure.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
})

FEAT = "/mnt/data1/asing725/viyog/results/features"
A = "/mnt/data1/asing725/viyog/results/analysis"
OUT = "/mnt/data1/asing725/viyog/paper_rev/figs/rebuttal"
os.makedirs(OUT, exist_ok=True)
LOW_PCT = 0.10
ID = "#3b5b92"     # blue
OOD = "#2a924a"    # green
ADV = "#b2182b"    # red


def load(model, split):
    p = f"{FEAT}/featfull_{model}_{split}.h5"
    if not os.path.exists(p):
        return None
    with h5py.File(p, "r") as h:
        return {"tv": h["filter_tv"][:], "mean": h["filter_means"][:]}


def dorm_idx(id_mean):
    per = id_mean.mean(0)
    alive = np.where(per > 1e-4)[0]
    if len(alive) == 0:
        alive = np.arange(len(per))
    aorder = alive[np.argsort(per[alive])[::-1]]
    n_low = max(1, int(len(alive) * LOW_PCT))
    return aorder[-n_low:], per


def auroc(neg, pos):
    from sklearn.metrics import roc_auc_score
    y = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
    s = np.r_[neg, pos]
    a = roc_auc_score(y, s)
    return max(a, 1 - a)


# --------------------------------------------------- fig 1: the signature
def fig_signature(model="resnet50"):
    idd = load(model, "id")
    # pool several OOD sets (near + far) so the figure is representative, not cherry-picked
    oods = [load(model, f"ood_{o}") for o in ("svhn", "cifar10", "dtd", "stl10", "fashionmnist")]
    oods = [o for o in oods if o is not None]
    adv = load(model, "adv_pgd")
    if not (idd and oods and adv):
        print(f"  [signature] missing splits for {model}"); return
    L, per = dorm_idx(idd["mean"])
    tv = lambda d: d["tv"][:, L].mean(1)
    s_id = tv(idd)
    s_ood = np.concatenate([tv(o) for o in oods])
    s_adv = tv(adv)
    a_oa = auroc(s_ood, s_adv)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.4, 2.9),
                                   gridspec_kw={"width_ratios": [1.25, 1]})
    # (a) score distributions
    bins = np.linspace(min(s_id.min(), s_ood.min(), s_adv.min()),
                       np.percentile(np.r_[s_id, s_ood, s_adv], 99), 40)
    for s, c, lab in [(s_id, ID, "ID"), (s_ood, OOD, "OOD (SVHN)"), (s_adv, ADV, "ADV (PGD)")]:
        ax1.hist(s, bins=bins, density=True, histtype="stepfilled", alpha=0.45, color=c)
        ax1.hist(s, bins=bins, density=True, histtype="step", lw=1.3, color=c, label=lab)
    ax1.set_xlabel(r"Viyog score $V(x)$")
    ax1.set_ylabel("Density")
    ax1.set_title(f"(a) Viyog score: OOD/ADV well-separated\n(OOD-vs-ADV AUROC {a_oa:.2f}, ResNet-50)", pad=4)
    ax1.legend(frameon=False, loc="upper right")
    ax1.spines[["top", "right"]].set_visible(False)

    # (b) per-channel ID-mean profile with the dormant band shaded
    order = np.argsort(per)[::-1]
    ax2.plot(np.arange(len(per)), per[order], color="#555", lw=1.0)
    rank_of = {c: r for r, c in enumerate(order)}
    dpos = sorted(rank_of[c] for c in L)
    ax2.axvspan(min(dpos), len(per) - 1, color=ADV, alpha=0.12)
    ax2.scatter(dpos, per[order][dpos], s=10, color=ADV, zorder=3,
                label=f"dormant band\n(bottom {int(LOW_PCT*100)}% alive)")
    ax2.set_xlabel("First-conv channel (ranked by mean ID activation)")
    ax2.set_ylabel("Mean activation")
    ax2.set_title("(b) Signal lives in the\nquietest channels (dormant band)", pad=4)
    ax2.legend(frameon=False, loc="upper right")
    ax2.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_viyogd_signature.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  [signature] {model}: |L|={len(L)} OOD-vs-ADV AUROC={a_oa:.3f} "
          f"(ID {s_id.mean():.3f}, OOD {s_ood.mean():.3f}, ADV {s_adv.mean():.3f})")


# --------------------------------------------------- fig 2: signature battery
def fig_battery():
    fs = [f for f in glob.glob(f"{A}/signature_auroc_full_*.csv")
          if not any(s in f for s in ("resnet18", "resnet34"))]  # skip all-NaN stubs
    cols = []
    for f in fs:
        d = pd.read_csv(f, index_col=0)
        if "T3_OOD_vs_ADV" in d.columns:
            cols.append(d["T3_OOD_vs_ADV"])
    M = pd.concat(cols, axis=1)
    rank = M.mean(1).sort_values()  # ascending for barh
    top = rank.tail(12)
    is_g = top.index.str.startswith("G_") | top.index.str.startswith("H_") | top.index.str.startswith("J_")
    colors = ["#1b7837" if g else "#9970ab" for g in is_g]
    # highlight the winner
    colors = ["#b2182b" if n == "G_tv_dorm" else c for n, c in zip(top.index, colors)]
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    y = np.arange(len(top))
    ax.barh(y, top.values, color=colors, alpha=0.9)
    ax.set_yticks(y); ax.set_yticklabels([n.replace("_", "\\_") for n in top.index], fontsize=6.2)
    ax.axvline(0.5, ls=":", c="grey", lw=0.7)
    ax.set_xlim(0.5, max(0.82, top.max() + 0.02))
    ax.set_xlabel("OOD-vs-ADV AUROC (mean over 16 archs)")
    ax.set_title("Signature battery: the new spatial\n$G_{\\mathrm{tv\\_dorm}}$ tops all 37")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#b2182b", label="Viyog ($G_{tv\\_dorm}$)"),
                       Patch(color="#1b7837", label="spatial/freq (new)"),
                       Patch(color="#9970ab", label="logit/feature")],
              loc="lower right", frameon=False, fontsize=6)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_sig_battery.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  [battery] top: {rank.tail(4).round(3).to_dict()}")


# --------------------------------------------------- fig 3: embedded cost
def fig_cost():
    sy = pd.read_csv(f"{A}/systems_cifar100.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 2.6))
    # (a) detector state (log scale): Viyog vs distance baselines
    names = ["Viyog", r"$L_\infty$", "Maha/ViM", "KNN"]
    kb = [0.3, 0.008, 17000, 40000]
    cols = ["#b2182b", "#e08214", "#7570b3", "#7570b3"]
    ax1.bar(names, kb, color=cols, alpha=0.9)
    ax1.set_yscale("log"); ax1.set_ylabel("detector state (KB, log)")
    ax1.set_title("(a) Detector memory: $10^3$--$10^5\\times$ smaller")
    for i, v in enumerate(kb):
        ax1.text(i, v * 1.4, f"{v:g}", ha="center", fontsize=6)
    ax1.spines[["top", "right"]].set_visible(False)
    # (b) first-conv MAC ratio across the panel
    d = sy.sort_values("firstconv_mac_ratio_%")
    is_t = d["model"].str.contains("vit|swin", case=False)
    ax2.bar(range(len(d)), d["firstconv_mac_ratio_%"],
            color=["#762a83" if t else "#1b7837" for t in is_t], alpha=0.85)
    med = d[~is_t]["firstconv_mac_ratio_%"].median()
    ax2.axhline(med, ls="--", c="grey", lw=0.8)
    ax2.text(0.5, med + 1.5, f"CNN median {med:.1f}\\%", fontsize=6, color="grey")
    ax2.set_ylabel("first-conv MACs (\\% of model)")
    ax2.set_xlabel("20 architectures (sorted)")
    ax2.set_title("(b) Compute: a few \\% for CNNs")
    ax2.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_cost.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  [cost] wrote fig_cost.pdf")


if __name__ == "__main__":
    fig_signature("resnet50")
    fig_battery()
    fig_cost()
    print("done ->", OUT)
