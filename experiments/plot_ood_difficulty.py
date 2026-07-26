"""fig_ood_difficulty.pdf --- OOD-difficulty breakdown with TV/HF complementarity.

Shows mean OOD-vs-ADV (T3) AUROC across the 17-architecture CIFAR-100 panel,
split by OOD *difficulty* (far / near / texture), for the deployed dormant-band
total-variation read (Viyog-D), its high-frequency complement (HF), and the
combined panel (best-of TV/HF). The story: TV holds far/near, HF rescues
texture, and together the panel separates ADV from every OOD difficulty while
the raw $L_\\infty$ magnitude norm stays near chance.

Usage::

    uv run python experiments/plot_ood_difficulty.py \\
        --out ../paper_rev/figs/rebuttal/fig_ood_difficulty.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

import viyog_plotstyle as vs  # noqa: E402

_CSV = Path("/mnt/data1/asing725/viyog/results/analysis/near_ood_breakdown_cifar100.csv")

_KINDS = ["far", "near", "texture"]
_KIND_LABELS = ["Far-OOD", "Near-OOD", "Texture-OOD"]


def _load(csv: Path) -> pd.DataFrame:
    """Load the near/far/texture breakdown, dropping any header-repeat rows."""
    df = pd.read_csv(csv)
    df = df[df["signature"] != "signature"].copy()
    for c in ("T3_far", "T3_near", "T3_texture"):
        df[c] = df[c].astype(float)
    return df


def _series(df: pd.DataFrame, signature: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean, std) over architectures for one signature across kinds."""
    sub = df[df["signature"] == signature]
    mean = np.array([sub[f"T3_{k}"].mean() for k in _KINDS])
    std = np.array([sub[f"T3_{k}"].std() for k in _KINDS])
    return mean, std


def _best_of(df: pd.DataFrame, sig_a: str, sig_b: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-architecture max of two signatures, then mean/std over architectures."""
    a = df[df["signature"] == sig_a].set_index("model")
    b = df[df["signature"] == sig_b].set_index("model")
    mean, std = [], []
    for k in _KINDS:
        best = np.maximum(a[f"T3_{k}"], b[f"T3_{k}"])
        mean.append(best.mean())
        std.append(best.std())
    return np.array(mean), np.array(std)


def main() -> None:
    """Build the OOD-difficulty complementarity figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="figs/rebuttal/fig_ood_difficulty.pdf")
    parser.add_argument("--csv", default=str(_CSV))
    args = parser.parse_args()

    df = _load(Path(args.csv))
    n_arch = df["model"].nunique()

    tv_m, tv_s = _series(df, "Viyog_D*(tv|p5|adapt)")
    hf_m, hf_s = _series(df, "G_hf_mean")
    panel_m, panel_s = _best_of(df, "Viyog_D*(tv|p5|adapt)", "G_hf_mean")
    linf_m, _ = _series(df, "A_inf_norm")

    vs.apply_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3.5, 2.7))

    x = np.arange(len(_KINDS))
    w = 0.26
    series = [
        ("Viyog-D (TV)", tv_m, tv_s, vs.C_OURS, "//"),
        ("HF complement", hf_m, hf_s, vs.C_COMP, "\\\\"),
        ("Panel (TV$\\vee$HF)", panel_m, panel_s, vs.C_ACCENT, None),
    ]
    for i, (label, m, s, color, hatch) in enumerate(series):
        ax.bar(
            x + (i - 1) * w, m, w, yerr=s, label=label, color=color,
            edgecolor="black", linewidth=0.5, alpha=0.92, hatch=hatch,
            error_kw=dict(elinewidth=0.7, capsize=1.8, ecolor="#333333"),
            zorder=3,
        )

    # raw Linf magnitude norm: faint grey reference markers (near chance)
    ax.plot(
        x, linf_m, "_", color="#999999", ms=14, markeredgewidth=2.0,
        zorder=4, label="raw $L_\\infty$",
    )

    # chance line
    ax.axhline(0.5, color=vs.C_REF, ls=":", lw=1.0, alpha=0.7, zorder=1)
    ax.text(2.42, 0.508, "chance", fontsize=6, color=vs.C_REF, va="bottom", ha="right")

    # highlight the texture rescue: TV 0.67 -> panel 0.92
    ax.annotate(
        "",
        xy=(2 + w, panel_m[2] - 0.01), xytext=(2 - w, tv_m[2] + 0.012),
        arrowprops=dict(arrowstyle="->", color=vs.C_BASE, lw=1.2,
                        connectionstyle="arc3,rad=-0.3"), zorder=5,
    )
    ax.text(
        1.0, 0.93,
        f"HF rescues texture-OOD:  {tv_m[2]:.2f} $\\to$ {panel_m[2]:.2f}",
        fontsize=6.4, color=vs.C_BASE, ha="center", va="center", style="italic",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(_KIND_LABELS, fontsize=8)
    ax.set_ylabel("OOD-vs-ADV AUROC (T3)", fontsize=8)
    ax.set_ylim(0.4, 1.24)
    ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_title(
        f"Complementary reads cover every OOD difficulty (mean $\\pm$ std, {n_arch} archs)",
        fontsize=7.5, pad=14,
    )
    ax.legend(fontsize=6.0, loc="upper center", bbox_to_anchor=(0.5, 1.04),
              ncol=4, columnspacing=1.1, handlelength=1.3, handletextpad=0.4,
              frameon=False)
    vs.despine(ax)
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.11, top=0.82)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    vs.savefig_pdf(fig, str(out))
    plt.close(fig)

    size_kb = out.stat().st_size / 1024
    print(f"OK: {size_kb:.1f} KB | TV tex {tv_m[2]:.3f} -> panel {panel_m[2]:.3f} "
          f"| far TV {tv_m[0]:.3f} HF {hf_m[0]:.3f}")


if __name__ == "__main__":
    main()
