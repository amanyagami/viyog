"""fig_mechanism.pdf — figure showing V(x) mechanism using real ResNet-50 feature data.

Three panels illustrating how dormant-band total variation (TV) behaves
differently for ID vs OOD vs ADV inputs at the channel level.

Usage::

    uv run python experiments/plot_mechanism.py --out figs/rebuttal/fig_mechanism.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# Local imports — add experiments/ to sys.path so viyog_plotstyle is found
# ---------------------------------------------------------------------------

_EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

import viyog_plotstyle as vs  # noqa: E402  (after sys.path patch)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEAT_DIR = Path("/mnt/data1/asing725/viyog/results/multiseed/seed1/results/features")
N_SAMPLES: int = 2000
N_CHANNELS: int = 64
DORMANT_FRAC: float = 0.10  # bottom 10 % of alive channels
EPS: float = 1e-6


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def load_features(
    path: Path,
    n: int = N_SAMPLES,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (filter_means, filter_tv) arrays of shape (n, 64) as float32."""
    with h5py.File(path, "r") as fh:
        means = fh["filter_means"][:n].astype(np.float32)
        tv = fh["filter_tv"][:n].astype(np.float32)
    return means, tv


def dormant_indices(id_means: np.ndarray) -> np.ndarray:
    """Compute dormant-band indices from ID channel-mean magnitudes.

    Parameters
    ----------
    id_means:
        Shape (N, C).  Dormancy is determined by the *per-channel* mean
        across ID samples, sorted ascending, keeping the bottom 10 %.
    """
    ch_mean = id_means.mean(axis=0)  # (C,)
    thresh = np.percentile(ch_mean, DORMANT_FRAC * 100)
    dorm_idx = np.where(ch_mean <= thresh)[0]
    return dorm_idx


def compute_v(
    means: np.ndarray,
    tv: np.ndarray,
    dorm_idx: np.ndarray,
) -> np.ndarray:
    """Compute per-sample V(x) score over dormant-band channels.

    V(x) = mean_c∈B [ TV_c(x) / (|mean_c(x)| + ε) ]
    """
    dorm_tv = tv[:, dorm_idx]  # (N, |B|)
    dorm_means = means[:, dorm_idx]  # (N, |B|)
    ratio = dorm_tv / (dorm_means + EPS)  # (N, |B|)
    return ratio.mean(axis=1)  # (N,)


# ---------------------------------------------------------------------------
# Panel helpers
# ---------------------------------------------------------------------------


def _panel_a(
    ax: plt.Axes,
    id_means: np.ndarray,
    adv_means: np.ndarray,
    adv_tv: np.ndarray,
    id_tv: np.ndarray,
    dorm_idx: np.ndarray,
) -> None:
    """Panel (a): channel activation magnitudes — ID vs ADV vs OOD."""
    ch_mean_id = id_means.mean(axis=0)  # (64,)
    ch_tv_id = id_tv.mean(axis=0)  # (64,)
    ch_tv_adv = adv_tv.mean(axis=0)  # (64,)

    sort_order = np.argsort(ch_mean_id)
    sorted_means = ch_mean_id[sort_order]
    sorted_tv_id = ch_tv_id[sort_order]
    sorted_tv_adv = ch_tv_adv[sort_order]

    # Determine dormant positions in sorted order
    is_dorm = np.isin(sort_order, dorm_idx)
    colors = [vs.C_OURS if d else "#AAAAAA" for d in is_dorm]

    x = np.arange(N_CHANNELS)
    ax.bar(x, sorted_means, color=colors, width=0.9, zorder=2, alpha=0.85)

    # Dormant band shading
    dorm_positions = np.where(is_dorm)[0]
    if len(dorm_positions) > 0:
        x0 = dorm_positions.min() - 0.5
        x1 = dorm_positions.max() + 0.5
        ax.axvspan(x0, x1, color=vs.C_OURS, alpha=0.12, zorder=1, label="_nolegend_")
        ax.annotate(
            "dormant\nband $\\mathcal{B}$",
            xy=((x0 + x1) / 2, sorted_means[dorm_positions].max() * 1.05),
            xytext=((x0 + x1) / 2 + 4, sorted_means.max() * 0.55),
            fontsize=6.5,
            color=vs.C_OURS,
            arrowprops=dict(arrowstyle="-", color=vs.C_OURS, lw=0.8),
            ha="left",
        )

    # Overlay: ADV TV vs ID TV in dormant band as scatter dots at top of bars
    for pos, ch in zip(np.where(is_dorm)[0], sort_order[is_dorm]):
        y_id = sorted_tv_id[pos]
        y_adv = sorted_tv_adv[pos]
        ax.plot(pos, y_id, ".", color=vs.C_OURS, ms=4, zorder=5)
        ax.plot(pos, y_adv, "x", color=vs.C_BASE, ms=5, zorder=5, markeredgewidth=1.2)

    # Legend proxies
    bar_id = mpatches.Patch(color=vs.C_OURS, label="ID mean |act.|")
    bar_grey = mpatches.Patch(color="#AAAAAA", label="Active channels")
    dot_id = plt.Line2D([0], [0], marker=".", color=vs.C_OURS, lw=0, ms=5, label="TV_ID (dorm.)")
    dot_adv = plt.Line2D(
        [0],
        [0],
        marker="x",
        color=vs.C_BASE,
        lw=0,
        ms=5,
        markeredgewidth=1.2,
        label="TV_ADV (dorm.)",
    )
    ax.legend(handles=[bar_id, bar_grey, dot_id, dot_adv], fontsize=5.5, loc="upper center")

    ax.set_xlabel("Channel index (sorted by |act.|)")
    ax.set_ylabel("Mean |activation|")
    ax.set_title("Channel activations\n(ID vs ADV)", pad=3)
    ax.set_xlim(-0.5, N_CHANNELS - 0.5)
    vs.despine(ax)


def _panel_b(
    ax: plt.Axes,
    v_id: np.ndarray,
    v_ood: np.ndarray,
    v_adv: np.ndarray,
    auroc_adv: float,
) -> None:
    """Panel (b): per-sample V(x) distribution — ID vs OOD vs ADV."""
    # KDE curves
    all_vals = np.concatenate([v_id, v_ood, v_adv])
    x_min, x_max = all_vals.min(), all_vals.max()
    x_grid = np.linspace(x_min, x_max, 400)

    for v_arr, color, label in [
        (v_id, vs.C_OURS, "ID"),
        (v_ood, vs.C_COMP, "OOD (CIFAR-10)"),
        (v_adv, vs.C_BASE, "ADV-PGD"),
    ]:
        bw = 1.06 * v_arr.std() * len(v_arr) ** (-0.2)  # Silverman BW
        kde = gaussian_kde(v_arr, bw_method=bw / v_arr.std())
        density = kde(x_grid)
        ax.plot(x_grid, density, color=color, lw=1.5, label=label)
        ax.fill_between(x_grid, density, alpha=0.12, color=color)

    # Threshold tau = 90th percentile of ID
    tau = float(np.percentile(v_id, 90))
    ax.axvline(tau, color=vs.C_REF, ls="--", lw=1.2, label="$\\tau$ (90th pct ID)")

    # AUROC annotation box
    ax.text(
        0.97,
        0.70,
        f"AUROC\n= {auroc_adv:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#AAAAAA", lw=0.7),
    )

    ax.set_xlabel("$V(x)$  (dormant-band roughness)")
    ax.set_ylabel("Density")
    ax.set_title("Per-sample $V(x)$\ndistribution", pad=3)
    ax.legend(fontsize=6, loc="upper center")
    vs.despine(ax)


def _panel_c(ax: plt.Axes) -> None:
    """Panel (c): why scale-invariance matters (P1).

    Shows schematically that doubling activation magnitude leaves V = TV/mean
    unchanged, while ADV raises the ratio by inflating TV.
    """
    # Conceptual values — chosen to illustrate the property clearly
    mean_clean = 2.0
    tv_clean = 0.4
    ratio_clean = tv_clean / mean_clean  # 0.2

    mean_scaled = 4.0  # 2x scale
    tv_scaled = 0.8  # 2x scale → same ratio
    ratio_scaled = tv_scaled / mean_scaled  # 0.2

    mean_adv = 2.0  # same mean as ID
    tv_adv = 1.2  # inflated TV
    ratio_adv = tv_adv / mean_adv  # 0.6

    labels = ["ID\n(clean)", "ID\n(x2 scale)", "ADV\n(PGD)"]
    means = [mean_clean, mean_scaled, mean_adv]
    tvs = [tv_clean, tv_scaled, tv_adv]
    ratios = [ratio_clean, ratio_scaled, ratio_adv]
    colors = [vs.C_OURS, vs.C_ACCENT, vs.C_BASE]

    x = np.array([0, 1, 2])
    width = 0.24

    # Sub-axis 1: mean |activation|
    bars1 = ax.bar(x - width, means, width, color=colors, alpha=0.55, label="mean |act.|")

    # Sub-axis 2: TV (same axis, offset)
    bars2 = ax.bar(x, tvs, width, color=colors, alpha=0.80, hatch="//", label="TV")

    # Sub-axis 3: V = TV/mean ratio as a secondary y-axis
    ax2 = ax.twinx()
    ax2.bar(x + width, ratios, width, color=colors, alpha=0.95, edgecolor="k", lw=0.5)
    ax2.set_ylim(0, 0.82)  # headroom so V=0.60 label clears the panel title
    ax2.set_ylabel("$V = $ TV / mean", fontsize=7, color="#333333")
    ax2.tick_params(axis="y", labelsize=6)
    ax2.spines["top"].set_visible(False)

    # Brace annotations for equal ratios
    for xi, r in zip(x + width, ratios):
        ax2.text(
            xi,
            r + 0.03,
            f"V={r:.2f}",
            ha="center",
            va="bottom",
            fontsize=6,
            fontweight="bold" if r > 0.3 else "normal",
            color=vs.C_BASE if r > 0.3 else "#333333",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Magnitude (arbitrary units)", fontsize=7)
    ax.set_title("Scale-invariance:\n$V$ = TV / mean", pad=3)
    ax.legend(
        handles=[bars1, bars2],
        labels=["mean |act.|", "TV"],
        fontsize=6,
        loc="upper left",
    )
    vs.despine(ax)

    # Explanatory caption below the panel
    ax.text(
        0.5,
        -0.22,
        "V = TV/mean — magnitude cancels, roughness stays",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=6.5,
        style="italic",
        color="#444444",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point: load data, build figure, save PDF."""
    parser = argparse.ArgumentParser(description="Plot V(x) mechanism figure.")
    parser.add_argument(
        "--out",
        default="figs/rebuttal/fig_mechanism.pdf",
        help="Output PDF path (relative to repo root or absolute).",
    )
    parser.add_argument(
        "--feat-dir",
        default=str(FEAT_DIR),
        help="Directory containing featfull_resnet50_*.h5 files.",
    )
    args = parser.parse_args()

    feat_dir = Path(args.feat_dir)
    out_path = Path(args.out)

    # ---- Load data ---------------------------------------------------------
    id_means, id_tv = load_features(feat_dir / "featfull_resnet50_id.h5")
    adv_means, adv_tv = load_features(feat_dir / "featfull_resnet50_adv_pgd.h5")
    ood_means, ood_tv = load_features(feat_dir / "featfull_resnet50_ood_cifar10.h5")

    # ---- Dormant band from ID distribution ---------------------------------
    dorm_idx = dormant_indices(id_means)

    # ---- Per-sample V(x) ---------------------------------------------------
    v_id = compute_v(id_means, id_tv, dorm_idx)
    v_adv = compute_v(adv_means, adv_tv, dorm_idx)
    v_ood = compute_v(ood_means, ood_tv, dorm_idx)

    # ---- AUROC (ID=negative, ADV=positive) ---------------------------------
    y_true = np.concatenate([np.zeros(len(v_id)), np.ones(len(v_adv))])
    y_score = np.concatenate([v_id, v_adv])
    auroc_adv = float(roc_auc_score(y_true, y_score))

    # ---- Layout ------------------------------------------------------------
    vs.apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 3.4))
    fig.subplots_adjust(wspace=0.60, left=0.08, right=0.97, bottom=0.22, top=0.88)

    _panel_a(axes[0], id_means, adv_means, adv_tv, id_tv, dorm_idx)
    _panel_b(axes[1], v_id, v_ood, v_adv, auroc_adv)
    _panel_c(axes[2])

    # Panel labels
    for ax, lbl in zip(axes, ["(a)", "(b)", "(c)"]):
        vs.add_panel_label(ax, lbl)

    # ---- Save --------------------------------------------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vs.savefig_pdf(fig, str(out_path))
    plt.close(fig)

    size_kb = out_path.stat().st_size / 1024
    print(f"OK: {size_kb:.1f} KB | T2_auroc={auroc_adv:.4f}")


if __name__ == "__main__":
    main()
