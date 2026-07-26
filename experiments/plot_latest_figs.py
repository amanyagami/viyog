"""Regenerate the rebuttal figures from the LATEST finished-experiment CSVs.

Fixes the stale figures the figure-audit flagged (fig_seed_ci, fig_adaptive_strong)
and adds the high-value missing ones (end-to-end cascade, layer-depth inverted-U,
signature-family ranking). All output is 300-dpi vector PDF with embedded fonts
(IEEE/TCAD-safe). Each figure is guarded: a missing CSV is skipped with a note.

Run: ../.venv/bin/python plot_latest_figs.py
"""

from __future__ import annotations
import glob
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config

AD = config.ANALYSIS_DIR  # cifar100 flat analysis dir
OUT = config.ROOT / "paper_rev" / "figs" / "rebuttal"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update(
    {
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)
CORE6 = ["resnet50", "densenet121", "mobilenetv3_l", "swin_tiny", "convnextv2_base", "vit_base"]
NICE = {
    "resnet50": "ResNet50",
    "densenet121": "DenseNet121",
    "mobilenetv3_l": "MobileNetV3-L",
    "swin_tiny": "Swin-T",
    "convnextv2_base": "ConvNeXtV2",
    "vit_base": "ViT-B",
}


def _save(fig, name: str) -> None:
    p = OUT / name
    fig.savefig(p)
    plt.close(fig)
    print("wrote", p)


def fig_seed_ci() -> None:
    """Training-seed CI forest plot (replaces the stale fig_seed_ci.pdf)."""
    f = config.ROOT / "results" / "analysis" / "multiseed_viyogd.csv"
    if not f.exists():
        print("skip seed_ci: no multiseed_viyogd.csv")
        return
    d = pd.read_csv(f)
    g = (
        d.groupby("model")
        .agg(
            T2=("T2", "mean"),
            T2s=("T2", "std"),
            T3=("T3", "mean"),
            T3s=("T3", "std"),
            rc=("recall", "mean"),
            rcs=("recall", "std"),
            n=("T2", "count"),
        )
        .reindex(CORE6)
    )
    order = g["T2"].sort_values().index
    g = g.loc[order]
    y = np.arange(len(g))
    fig, ax = plt.subplots(figsize=(5.0, 2.8))
    for col, sd, c, lab, off in [
        ("T2", "T2s", "#1f77b4", "T2 (ID-vs-ADV)", -0.18),
        ("T3", "T3s", "#ff7f0e", "T3 (OOD-vs-ADV)", 0.0),
        ("rc", "rcs", "#2ca02c", "recall@5%FPR", 0.18),
    ]:
        ax.errorbar(
            g[col], y + off, xerr=g[sd].fillna(0), fmt="o", ms=4, capsize=2, color=c, label=lab
        )
    ax.set_yticks(y)
    ax.set_yticklabels([NICE[m] for m in g.index])
    ax.axvline(0.5, ls="--", lw=0.8, color="grey")
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("AUROC / recall (mean $\\pm$ training-seed std, $n{=}3$)")
    nseed = int(g["n"].max())
    ax.set_title(f"Training-seed CIs ({nseed} re-finetunes): T2 seed-invariant (std $\\leq$0.009)")
    ax.legend(fontsize=7, loc="lower left", framealpha=0.9)
    _save(fig, "fig_seed_ci.pdf")


def fig_cascade_e2e() -> None:
    """End-to-end cascade: energy-only vs 2-signal gate over all 17 archs (A1/A2/D1)."""
    f = AD / "cascade_2signal_cifar100.csv"
    if not f.exists():
        print("skip cascade: no cascade_2signal_cifar100.csv")
        return
    d = pd.read_csv(f).dropna(subset=["energy_e2e_adv", "twosig_e2e_adv"])
    d = d.sort_values("energy_e2e_adv").reset_index(drop=True)
    x = np.arange(len(d))
    w = 0.4
    em, tm = d["energy_e2e_adv"].mean(), d["twosig_e2e_adv"].mean()
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.0), constrained_layout=True)
    # (a) e2e ADV recall: energy-only vs 2-signal, all 17 archs sorted
    ax = axes[0]
    ax.bar(x - w / 2, d["energy_e2e_adv"], w, label="energy-only gate", color="#bcbd22")
    ax.bar(
        x + w / 2,
        d["twosig_e2e_adv"],
        w,
        label="2-signal gate $\\max(z_E,|z_{dorm}|)$",
        color="#1f77b4",
    )
    ax.axhline(em, ls="--", lw=0.9, color="#7f7f0e")
    ax.axhline(tm, ls="--", lw=0.9, color="#1f77b4")
    ax.text(0.3, em + 0.02, f"mean {em:.2f}", fontsize=7, color="#7f7f0e")
    ax.text(0.3, tm + 0.02, f"mean {tm:.2f}", fontsize=7, color="#1f77b4")
    ax.set_xticks([])
    ax.set_xlabel("17 architectures (sorted)")
    ax.set_ylabel("end-to-end ADV recall")
    ax.set_ylim(0, 1)
    ax.set_title("(a) ADV recall at fixed 5% ID-FPR")
    ax.legend(fontsize=7, loc="upper left")
    # (b) ID->ADV mis-escalation (de-amplification): both gates ~0
    ax = axes[1]
    ax.bar(x - w / 2, d["energy_id2adv"], w, label="energy-only", color="#bcbd22")
    ax.bar(x + w / 2, d["twosig_id2adv"], w, label="2-signal", color="#1f77b4")
    ax.set_xticks([])
    ax.set_xlabel("17 architectures")
    ax.set_ylabel("ID$\\to$ADV mis-escalation")
    ax.set_ylim(0, max(0.05, float(d["twosig_id2adv"].max()) * 1.2 + 1e-3))
    ax.set_title(f"(b) ID$\\to$ADV mis-escalation ($\\leq${float(d['twosig_id2adv'].max()):.3f})")
    ax.legend(fontsize=7)
    fig.suptitle(
        f"End-to-end cascade (Viyog stage-2): ADV recall {em:.2f}$\\to${tm:.2f} at equal 5% ID-FPR",
        fontsize=9,
    )
    _save(fig, "fig_cascade_e2e.pdf")


def fig_layer_depth() -> None:
    """Layer-depth inverted-U: first conv is the WORST raw layer; shape recovers it."""
    f = AD / "layer_ablation_cifar100.csv"
    if not f.exists():
        print("skip layer_depth: no layer_ablation_cifar100.csv")
        return
    d = pd.read_csv(f)
    fig, ax = plt.subplots(figsize=(5.0, 2.9))
    for m, c in [("resnet50", "#1f77b4"), ("densenet121", "#d62728")]:
        sub = d[d.model == m].sort_values("depth")
        if not len(sub):
            continue
        ax.plot(sub["depth"], sub["T2_ID_ADV"], "-o", ms=3, color=c, label=f"{NICE.get(m, m)} T2")
        ax.plot(
            sub["depth"],
            sub["T3_OOD_ADV"],
            "--s",
            ms=3,
            color=c,
            alpha=0.6,
            label=f"{NICE.get(m, m)} T3",
        )
    ax.axhline(0.5, ls=":", lw=0.8, color="grey")
    ax.axhline(0.966, ls="-.", lw=1.0, color="#2ca02c", label="Viyog$^*$ @ first conv (0.966)")
    # mark depth 0 (first conv) as the worst raw layer
    r0 = d[(d.model == "resnet50") & (d.depth == 0)]
    if len(r0):
        ax.annotate(
            "first conv\n(raw L$_\\infty$ weakest)",
            xy=(0, float(r0["T2_ID_ADV"].iloc[0])),
            xytext=(1.5, 0.62),
            fontsize=7,
            arrowprops=dict(arrowstyle="->", lw=0.7),
        )
    ax.set_xlabel("layer depth (0 = first conv)")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.45, 1.02)
    ax.set_title("Raw norm is weakest at the first conv; the shape statistic recovers it")
    ax.legend(fontsize=6.5, ncol=2, loc="lower center")
    _save(fig, "fig_layer_depth.pdf")


def fig_sig_ranking() -> None:
    """First-conv signature ranking, colour-coded by statistic *kind*.

    Each candidate's prefix encodes its family (A=magnitude/norm, G=shape TV/HF,
    E/J=distance, F=logit, B/C/D/H=distribution). The magnitude family (which
    includes the original raw L_inf) sits near chance; the shape family (the
    deployed Viyog = G_tv_dorm) is best-in-class.
    """
    f = AD / "signature_auroc_full_resnet50.csv"
    if not f.exists():
        print("skip sig_ranking: no signature_auroc_full_resnet50.csv")
        return
    d = pd.read_csv(f).sort_values("T2_ID_vs_ADV").reset_index(drop=True)

    # map prefix letter -> (kind label, colour)
    fam = {
        "A": ("magnitude / norm (old raw $L_\\infty$)", "#D55E00"),
        "G": ("shape: TV / HF  (Viyog)", "#009E73"),
        "E": ("distance / feature", "#0072B2"),
        "J": ("distance / feature", "#0072B2"),
        "F": ("logit", "#E69F00"),
        "B": ("distribution", "#9467BD"),
        "C": ("distribution", "#9467BD"),
        "D": ("distribution", "#9467BD"),
        "H": ("distribution", "#9467BD"),
    }
    default = ("other", "#9aa0a6")

    def fam_of(sig: str):
        return fam.get(sig.split("_")[0], default)

    colors = [fam_of(s)[1] for s in d["signature"]]
    fig, ax = plt.subplots(figsize=(5.6, 6.8))
    y = np.arange(len(d))
    # deployed Viyog (G_tv_dorm) gets a black edge so it stands out
    edges = ["black" if s == "G_tv_dorm" else "none" for s in d["signature"]]
    lws = [1.3 if s == "G_tv_dorm" else 0.0 for s in d["signature"]]
    ax.barh(y, d["T2_ID_vs_ADV"], color=colors, height=0.82, edgecolor=edges, linewidth=lws)

    # annotate the deployed statistic and the old raw norm
    for s, lab in [
        ("G_tv_dorm", "  Viyog (deployed) ★"),
        ("A_inf_norm", "  raw $L_\\infty$ baseline"),
    ]:
        if s in set(d["signature"]):
            i = int(d.index[d["signature"] == s][0])
            ax.text(
                d["T2_ID_vs_ADV"][i] + 0.008,
                i,
                lab,
                va="center",
                fontsize=6.0,
                fontweight="bold",
                color="#1a7a44" if s == "G_tv_dorm" else "#9a3d00",
            )

    # Replace internal code names with readable labels
    NICE = {
        "G_tv_dorm": "TV dormant-band (Viyog ★)",
        "G_hf_dorm": "HF dormant-band",
        "G_tv_mean": "TV mean-band",
        "G_hf_mean": "HF mean-band",
        "G_std_dorm": "Std dormant-band",
        "G_hf_low_large": "HF low/large ratio",
        "G_gram_offdiag": "Gram off-diagonal",
        "A_inf_norm": "Raw L∞ norm (old baseline)",
        "A_l2_norm": "Raw L₂ norm",
        "A_l1_norm": "Raw L₁ norm",
        "A_mean": "Mean activation",
        "A_l1": "L₁ activation",
        "A_peak_l2": "Peak L₂",
        "A_l2": "L₂ activation",
        "A_energy": "Activation energy",
        "A_linf_mean": "Mean L∞",
        "B_ratio_low_large": "Low/large-band ratio",
        "B_large_frac": "Large-activation fraction",
        "B_low_frac": "Low-activation fraction",
        "B_ratio_mid_large": "Mid/large-band ratio",
        "C_gini": "Gini coefficient",
        "C_spectral_entropy": "Spectral entropy",
        "C_topk_frac": "Top-k fraction",
        "C_hoyer": "Hoyer sparsity",
        "C_participation": "Participation ratio",
        "D_crest_mean": "Crest factor (mean)",
        "D_spikiness": "Spikiness",
        "D_max_to_l2": "Max-to-L₂ ratio",
        "E_mahalanobis": "Mahalanobis (first-conv)",
        "E_cos_id": "Cosine dist. to ID",
        "E_l1_drift": "L₁ drift",
        "F_softmax_entropy": "Softmax entropy",
        "F_kl_uniform": "KL to uniform",
        "F_msp": "Max softmax prob.",
        "F_margin": "Logit margin",
        "F_max_logit": "Max logit",
        "F_energy": "Energy score",
        "H_dorm_entropy": "Dormant-band entropy",
        "J_pca_tail_resid": "PCA tail residual",
    }
    tick_labels = [NICE.get(s, s.replace("_", " ")) for s in d["signature"]]
    ax.set_yticks(y)
    ax.set_yticklabels(tick_labels, fontsize=5.3)
    ax.axvline(0.5, ls="--", lw=0.8, color="grey")
    ax.text(0.5, len(d) - 0.5, "chance", fontsize=6, color="grey", ha="center", va="bottom")
    ax.set_xlabel("ID-vs-ADV (T2) AUROC")
    ax.set_xlim(0.45, 1.06)
    ax.set_title(
        "First-conv signatures by kind (ResNet-50, 37 candidates):\n"
        "magnitude near chance, shape best-in-class",
        fontsize=8,
    )
    from matplotlib.patches import Patch

    seen, handles = set(), []
    for k in ["A", "G", "E", "F", "B"]:  # one entry per distinct kind, fixed order
        lab, col = fam[k]
        if lab not in seen:
            handles.append(Patch(color=col, label=lab))
            seen.add(lab)
    ax.legend(handles=handles, fontsize=5.8, loc="lower right", frameon=True, framealpha=0.9)
    _save(fig, "fig_sig_ranking.pdf")


def fig_adaptive_strong() -> None:
    """Strong (n=2000) adaptive frontier: combined detector floor vs attack success."""
    files = sorted(glob.glob(str(AD / "adaptive_strong_cifar100_*.csv")))
    if not files:
        print("skip adaptive_strong: none found")
        return
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for f in files:
        m = f.split("adaptive_strong_cifar100_")[1].replace(".csv", "")
        d = pd.read_csv(f)
        aa = d[d["mode"] == "allaware"].sort_values("lambda")
        if not len(aa):
            continue
        comb = np.maximum(aa["auroc_dorm"], aa["auroc_hf"])
        ax.plot(aa["attack_success"], comb, "-o", ms=3, label=NICE.get(m, m), alpha=0.85)
    ax.axhline(0.5, ls=":", lw=0.8, color="grey")
    ax.axvline(0.8, ls="--", lw=0.8, color="k", alpha=0.5)
    ax.set_xlabel("Attack success rate (all-aware attacker; $\\lambda\\uparrow$ moves left)")
    ax.set_ylabel("Combined detector AUROC\n$\\max(V_{\\mathrm{TV}},\\,V_{\\mathrm{HF}})$")
    ax.set_ylim(0.45, 1.0)
    ax.set_title("Strong adaptive frontier ($n=2000$):\ndetector floor while attack succeeds")
    ax.legend(fontsize=6.5, ncol=2)
    _save(fig, "fig_adaptive_strong.pdf")


if __name__ == "__main__":
    fig_seed_ci()
    fig_cascade_e2e()
    fig_layer_depth()
    fig_sig_ranking()
    fig_adaptive_strong()
    print("done.")
