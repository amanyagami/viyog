"""OOD-vs-ADV (T3) detection broken down by architecture, attack, and dataset.

Answers the reviewer/PI questions directly with one figure:
  (a) Per-architecture  : Viyog T3 AUROC across all 20 CIFAR-100 backbones,
      coloured by architecture family, with the 0.824 mean band.
  (b) Per-attack        : Viyog T3 AUROC across the 4 standard attacks
      (FGSM / BIM / PGD / APGD-CE) on all three datasets.
  (c) Per-dataset       : Viyog vs the strongest logit detector (GEN) on the
      OOD-vs-ADV task for CIFAR-10 / CIFAR-100 / GTSRB.

Data: results/{ds}/analysis/full_eval_{ds}_{permodel,perattack,summary}.csv
(``cifar100`` lives directly under results/analysis/).

Run::

    uv run python experiments/plot_t3_breakdown.py \
        --out figs/rebuttal/fig_t3_breakdown.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import config
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import viyog_plotstyle as vs  # noqa: E402

CHANCE = 0.5
DEPLOYED = "ViyogD_tv_dorm"  # the deployed Viyog statistic
BEST_LOGIT = "GEN"  # strongest output-side baseline on T3

DATASETS = ["cifar10", "cifar100", "gtsrb"]
DS_NICE = {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100", "gtsrb": "GTSRB"}
DS_COL = {"cifar10": "#0072B2", "cifar100": "#E69F00", "gtsrb": "#CC79A7"}

ATTACK_ORDER = ["fgsm", "bim", "pgd", "apgd_ce"]
ATTACK_NICE = {"fgsm": "FGSM", "bim": "BIM", "pgd": "PGD", "apgd_ce": "APGD-CE"}

FAMILY = {
    "resnet18": "ResNet",
    "resnet34": "ResNet",
    "resnet50": "ResNet",
    "resnet101": "ResNet",
    "resnet152": "ResNet",
    "densenet121": "DenseNet",
    "densenet161": "DenseNet",
    "densenet169": "DenseNet",
    "densenet201": "DenseNet",
    "vit_base": "Transformer",
    "swin_tiny": "Transformer",
    "convnextv2_base": "Efficient/Hybrid",
    "edgenext_small": "Efficient/Hybrid",
    "efficientnetv2_l": "Efficient/Hybrid",
    "efficientvit_b1": "Efficient/Hybrid",
    "effnet_lite0": "Efficient/Hybrid",
    "fastvit_sa12": "Efficient/Hybrid",
    "mobilenetv3_l": "Efficient/Hybrid",
    "mobilenetv4_m": "Efficient/Hybrid",
    "mobileone_s1": "Efficient/Hybrid",
}
FAMCOL = {
    "ResNet": "#0072B2",
    "DenseNet": "#E69F00",
    "Transformer": "#CC79A7",
    "Efficient/Hybrid": "#56B4E9",
}
NAME = {
    "resnet18": "ResNet-18",
    "resnet34": "ResNet-34",
    "resnet50": "ResNet-50",
    "resnet101": "ResNet-101",
    "resnet152": "ResNet-152",
    "densenet121": "DenseNet-121",
    "densenet161": "DenseNet-161",
    "densenet169": "DenseNet-169",
    "densenet201": "DenseNet-201",
    "vit_base": "ViT-B",
    "swin_tiny": "Swin-T",
    "convnextv2_base": "ConvNeXtV2-B",
    "edgenext_small": "EdgeNeXt-S",
    "efficientnetv2_l": "EffNetV2-L",
    "efficientvit_b1": "EfficientViT-B1",
    "effnet_lite0": "EffNet-Lite0",
    "fastvit_sa12": "FastViT-SA12",
    "mobilenetv3_l": "MobileNetV3-L",
    "mobilenetv4_m": "MobileNetV4-M",
    "mobileone_s1": "MobileOne-S1",
}


def _analysis_dir(ds: str) -> Path:
    """Return the analysis directory for a dataset."""
    return config.ANALYSIS_DIR if ds == "cifar100" else config.RESULTS_DIR / ds / "analysis"


def _load(ds: str, kind: str) -> pd.DataFrame | None:
    """Load a full_eval CSV for *ds* of the given *kind* (permodel/perattack/summary)."""
    f = _analysis_dir(ds) / f"full_eval_{ds}_{kind}.csv"
    return pd.read_csv(f) if f.exists() else None


def _panel_architecture(ax: plt.Axes) -> None:
    """Panel (a): Viyog T3 AUROC per architecture (CIFAR-100, 20 backbones)."""
    pm = _load("cifar100", "permodel")
    d = pm[pm.method == DEPLOYED][["model", "T3_dl"]].copy()
    d["fam"] = d.model.map(FAMILY)
    d = d.sort_values("T3_dl").reset_index(drop=True)
    x = np.arange(len(d))
    colors = [FAMCOL[f] for f in d.fam]
    ax.bar(x, d.T3_dl, color=colors, edgecolor="white", linewidth=0.4, zorder=3)
    mean = d.T3_dl.mean()
    ax.axhline(mean, ls="--", lw=1.1, color=vs.C_REF, zorder=4)
    ax.text(
        len(d) - 0.3,
        mean + 0.012,
        f"mean {mean:.2f}",
        ha="right",
        fontsize=7,
        color=vs.C_REF,
        fontweight="bold",
    )
    ax.axhline(CHANCE, ls=":", lw=1.0, color="#999999", zorder=2)
    ax.text(0.1, CHANCE + 0.01, "chance", fontsize=6.5, color="#999999")
    ax.set_xticks(x)
    ax.set_xticklabels([NAME[m] for m in d.model], rotation=42, ha="right", fontsize=6.3)
    ax.set_ylabel("OOD-vs-ADV (T3) AUROC")
    ax.set_ylim(0.45, 1.0)
    ax.set_title(
        "(a) Per-architecture — CIFAR-100, 20 backbones (bars = Viyog, coloured by family)",
        fontsize=8.5,
    )
    handles = [plt.Rectangle((0, 0), 1, 1, color=FAMCOL[f]) for f in FAMCOL]
    ax.legend(
        handles,
        list(FAMCOL),
        fontsize=6.3,
        ncol=4,
        loc="upper left",
        frameon=False,
        handlelength=1.0,
        columnspacing=1.0,
    )


def _panel_attack(ax: plt.Axes) -> None:
    """Panel (b): Viyog T3 AUROC per attack, across the three datasets."""
    w = 0.26
    xx = np.arange(len(ATTACK_ORDER))
    for j, ds in enumerate(DATASETS):
        pa = _load(ds, "perattack")
        g = pa[pa.method == DEPLOYED].groupby("attack")["T3_dl"].mean()
        vals = [g.get(a, np.nan) for a in ATTACK_ORDER]
        ax.bar(
            xx + (j - 1) * w,
            vals,
            w,
            color=DS_COL[ds],
            label=DS_NICE[ds],
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )
    ax.axhline(CHANCE, ls=":", lw=1.0, color="#999999", zorder=2)
    ax.text(3.35, CHANCE + 0.008, "chance", fontsize=6.5, color="#999999", ha="right")
    ax.set_xticks(xx)
    ax.set_xticklabels([ATTACK_NICE[a] for a in ATTACK_ORDER], fontsize=7.5)
    ax.set_ylabel("T3 AUROC")
    ax.set_ylim(0.45, 1.0)
    ax.set_title("(b) Per-attack (Viyog, mean over models)", fontsize=8.5)
    ax.legend(fontsize=6.5, loc="upper right", ncol=3, frameon=False,
              columnspacing=0.9, handlelength=1.0, bbox_to_anchor=(1.0, 1.02))


def _panel_dataset(ax: plt.Axes) -> None:
    """Panel (c): Viyog vs best logit detector (GEN) on T3, per dataset."""
    rows = []
    for ds in DATASETS:
        s = _load(ds, "summary")
        t3 = s[s.task == "T3"].set_index("method")["deployable"]
        rows.append((ds, t3.get(DEPLOYED, np.nan), t3.get(BEST_LOGIT, np.nan)))
    df = pd.DataFrame(rows, columns=["ds", "viyog", "gen"])
    xx = np.arange(len(df))
    w = 0.36
    b1 = ax.bar(
        xx - w / 2,
        df.viyog,
        w,
        color=vs.C_OURS,
        label="Viyog (ours)",
        edgecolor="black",
        linewidth=0.6,
        zorder=3,
    )
    b2 = ax.bar(
        xx + w / 2, df.gen, w, color=vs.C_BASE, label="GEN (best logit)", edgecolor="none", zorder=3
    )
    for bars in (b1, b2):
        for b in bars:
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 0.008,
                f"{b.get_height():.2f}",
                ha="center",
                fontsize=6.3,
            )
    ax.axhline(CHANCE, ls=":", lw=1.0, color="#999999", zorder=2)
    ax.set_xticks(xx)
    ax.set_xticklabels([DS_NICE[d] for d in df.ds], fontsize=7.5)
    ax.set_ylabel("T3 AUROC")
    ax.set_ylim(0.45, 1.0)
    ax.set_title(
        "(c) Per-dataset: free input-side signal\nmatches output-side detectors", fontsize=8.5
    )
    ax.legend(fontsize=6.5, loc="upper center", ncol=2, frameon=False,
              columnspacing=1.2, handlelength=1.0, bbox_to_anchor=(0.5, 1.0))


def main() -> None:
    """Build and save the three-panel T3 breakdown figure."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="figs/rebuttal/fig_t3_breakdown.pdf")
    args = ap.parse_args()

    fig = plt.figure(figsize=(7.16, 4.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.18, 1.0], hspace=0.72, wspace=0.26)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    _panel_architecture(ax_a)
    _panel_attack(ax_b)
    _panel_dataset(ax_c)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    vs.savefig_pdf(fig, str(out))
    plt.close(fig)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
