r"""Forest plot of per-architecture training-seed CIs for the 20-arch Viyog panel.

Produces ``figs/rebuttal/fig_seed_forest.pdf`` with two side-by-side panels:

  (a) T2 (ID-vs-ADV) forest — 20 archs sorted descending by mean T2
  (b) T3 (OOD-vs-ADV) forest — same arch order as panel (a)

Each panel shows:
  - Filled circle at per-arch mean AUROC
  - Horizontal errorbar = mean ± std across seeds
  - Faint individual-seed scatter (alpha=0.35, marker='x')
  - Vertical dashed line at the panel mean with ±1 overall-std shaded band
  - ViT-B outlier annotation
  - "per-arch std ≤ 0.004" annotation in panel (a)

Usage::

    uv run python experiments/plot_seed_forest.py \\
        --out figs/rebuttal/fig_seed_forest_new.pdf

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup: allow importing viyog_plotstyle from the experiments/ directory
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import viyog_plotstyle as vs  # noqa: E402 (after sys.path patch)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_PATH: Path = Path("/mnt/data1/asing725/viyog/results/analysis/multiseed_viyogd_full20.csv")

ARCH_NAMES: dict[str, str] = {
    "convnextv2_base": "ConvNeXtV2",
    "vit_base": "ViT-B/16",
    "swin_tiny": "Swin-T",
    "mobilenetv3_l": "MobNetV3-L",
    "mobilenetv4_m": "MobNetV4-M",
    "efficientnetv2_l": "EffNetV2-L",
    "efficientvit_b1": "EffViT-B1",
    "effnet_lite0": "EffLite0",
    "fastvit_sa12": "FastViT-SA12",
    "mobileone_s1": "MobileOne-S1",
    "edgenext_small": "EdgeNeXt-S",
    "densenet121": "DenseNet-121",
    "densenet161": "DenseNet-161",
    "densenet169": "DenseNet-169",
    "densenet201": "DenseNet-201",
    "resnet18": "ResNet-18",
    "resnet34": "ResNet-34",
    "resnet50": "ResNet-50",
    "resnet101": "ResNet-101",
    "resnet152": "ResNet-152",
}

VIT_B_KEY: str = "ViT-B/16"


# ---------------------------------------------------------------------------
# Data loading and aggregation
# ---------------------------------------------------------------------------


def load_and_aggregate(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the multiseed CSV and compute per-arch mean/std.

    Args:
        csv_path: Path to the multiseed CSV with columns
            ``seed_idx, feat_dir, model, T2, T3, recall``.

    Returns:
        A tuple ``(raw_df, agg_df)`` where:
          - ``raw_df`` has the original rows with a ``short_name`` column added.
          - ``agg_df`` has columns ``short_name, T2_mean, T2_std, T3_mean,
            T3_std, n_seeds``, indexed by short arch name, sorted descending
            by ``T2_mean``.
    """
    raw: pd.DataFrame = pd.read_csv(csv_path)
    raw["short_name"] = raw["model"].map(ARCH_NAMES).fillna(raw["model"])

    agg: pd.DataFrame = (
        raw.groupby("short_name")
        .agg(
            T2_mean=("T2", "mean"),
            T2_std=("T2", "std"),
            T3_mean=("T3", "mean"),
            T3_std=("T3", "std"),
            n_seeds=("T2", "count"),
        )
        .reset_index()
    )
    # std is NaN for single-seed archs; replace with 0
    agg[["T2_std", "T3_std"]] = agg[["T2_std", "T3_std"]].fillna(0.0)
    agg = agg.sort_values("T2_mean", ascending=False).reset_index(drop=True)
    return raw, agg


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _draw_panel(
    ax: plt.Axes,
    agg: pd.DataFrame,
    raw: pd.DataFrame,
    metric: str,
    panel_mean: float,
    panel_std: float,
    xlim: tuple[float, float],
    title: str,
    show_annotation: bool,
) -> None:
    """Render one forest-plot panel onto *ax*.

    Args:
        ax: Target :class:`~matplotlib.axes.Axes`.
        agg: Aggregated per-arch statistics (sorted descending by T2_mean).
        raw: Raw seed-level rows with ``short_name`` column.
        metric: Either ``'T2'`` or ``'T3'``.
        panel_mean: Overall (across-arch) mean for the reference line.
        panel_std: Overall std for the shaded band.
        xlim: ``(xmin, xmax)`` for the x-axis.
        title: Panel title string.
        show_annotation: If ``True`` add the "per-arch std ≤ 0.004" note.
    """
    n_arch = len(agg)
    y_positions: np.ndarray = np.arange(n_arch)

    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    # Shaded ±1 overall-std band behind everything
    ax.axvspan(
        panel_mean - panel_std,
        panel_mean + panel_std,
        color=vs.C_REF,
        alpha=0.08,
        zorder=0,
    )

    # Reference line at panel mean
    ax.axvline(
        panel_mean,
        color=vs.C_REF,
        linestyle="--",
        linewidth=1.0,
        zorder=1,
        label=f"panel mean = {panel_mean:.3f}",
    )

    # Individual seed scatter (faint)
    for i, row in agg.iterrows():
        arch = row["short_name"]
        seed_vals = raw.loc[raw["short_name"] == arch, metric].values
        y_jitter = np.full(len(seed_vals), y_positions[i])
        ax.scatter(
            seed_vals,
            y_jitter,
            marker="x",
            s=30,
            alpha=0.35,
            color=vs.C_OURS,
            linewidths=0.8,
            zorder=2,
        )

    # Error bars (whiskers)
    ax.errorbar(
        agg[mean_col].values,
        y_positions,
        xerr=agg[std_col].values,
        fmt="none",
        ecolor=vs.C_OURS,
        elinewidth=1.2,
        capsize=3,
        capthick=1.0,
        zorder=3,
    )

    # Filled circles at mean
    colors = [
        vs.C_WARN if row["short_name"] == VIT_B_KEY else vs.C_OURS for _, row in agg.iterrows()
    ]
    ax.scatter(
        agg[mean_col].values,
        y_positions,
        c=colors,
        s=28,
        zorder=4,
        edgecolors="white",
        linewidths=0.4,
    )

    # ViT-B annotation
    vitb_mask = agg["short_name"] == VIT_B_KEY
    if vitb_mask.any():
        vitb_idx = agg.index[vitb_mask][0]
        vitb_y = y_positions[vitb_idx]
        vitb_x = agg.loc[vitb_idx, mean_col]
        ax.annotate(
            "ViT-B\n(patch-embed)",
            xy=(vitb_x, vitb_y),
            xytext=(vitb_x + 0.04, vitb_y - 1.4),
            fontsize=6,
            color="gray",
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.7),
            va="top",
        )

    # Arch labels on y-axis with seed count
    y_labels = [f"{row['short_name']} (n={row['n_seeds']})" for _, row in agg.iterrows()]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=7)
    ax.set_xlim(xlim)
    ax.set_ylim(-0.8, n_arch - 0.2)
    ax.set_xlabel("AUROC (training-seed mean ± std)", fontsize=8)
    ax.set_title(title, fontsize=9, pad=4)

    # Per-arch std annotation in panel (a) only
    if show_annotation:
        ax.text(
            0.02,
            0.97,
            "per-arch std ≤ 0.004",
            transform=ax.transAxes,
            fontsize=6.5,
            color=vs.C_REF,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8),
        )


def build_figure(agg: pd.DataFrame, raw: pd.DataFrame, out_path: Path) -> None:
    """Build the full two-panel forest figure and save to *out_path*.

    Args:
        agg: Per-arch aggregated statistics sorted descending by T2_mean.
        raw: Raw seed-level DataFrame with ``short_name`` column.
        out_path: Destination PDF path.
    """
    vs.apply_style()

    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(7.0, 4.2),
        gridspec_kw={"width_ratios": [1.6, 1]},
        constrained_layout=True,
    )

    # Panel-level statistics
    t2_panel_mean: float = float(agg["T2_mean"].mean())
    t2_panel_std: float = float(agg["T2_mean"].std())
    t3_panel_mean: float = float(agg["T3_mean"].mean())
    t3_panel_std: float = float(agg["T3_mean"].std())

    _draw_panel(
        ax=ax_a,
        agg=agg,
        raw=raw,
        metric="T2",
        panel_mean=t2_panel_mean,
        panel_std=t2_panel_std,
        xlim=(0.50, 1.02),
        title="(a) ID-vs-ADV (T2), sorted",
        show_annotation=True,
    )

    _draw_panel(
        ax=ax_b,
        agg=agg,
        raw=raw,
        metric="T3",
        panel_mean=t3_panel_mean,
        panel_std=t3_panel_std,
        xlim=(0.45, 1.02),
        title="(b) OOD-vs-ADV (T3)",
        show_annotation=False,
    )

    # Remove y-tick labels from panel (b) — shared order with panel (a)
    ax_b.set_yticklabels([])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    vs.savefig_pdf(fig, str(out_path))
    plt.close(fig)

    # Print key numbers
    max_t2_std: float = float(agg["T2_std"].max())
    max_t3_std: float = float(agg["T3_std"].max())
    print(f"mean T2  = {t2_panel_mean:.4f}")
    print(f"mean T3  = {t3_panel_mean:.4f}")
    print(f"max per-arch std T2 = {max_t2_std:.4f}")
    print(f"max per-arch std T3 = {max_t3_std:.4f}")

    size_kb = out_path.stat().st_size / 1024
    if size_kb >= 10:
        print(f"PDF OK: {size_kb:.1f} KB  →  {out_path}")
    else:
        print(f"WARNING: PDF too small ({size_kb:.1f} KB)  →  {out_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed :class:`argparse.Namespace` with attribute ``out``.
    """
    parser = argparse.ArgumentParser(
        description="Produce a forest plot of per-arch seed CIs for Viyog."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("figs/rebuttal/fig_seed_forest.pdf"),
        help="Output PDF path (default: figs/rebuttal/fig_seed_forest.pdf)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_PATH,
        help="Path to multiseed_viyogd_full20.csv",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point: load data, aggregate, build and save the figure."""
    args = _parse_args()
    raw, agg = load_and_aggregate(args.data)
    build_figure(agg, raw, args.out)


if __name__ == "__main__":
    main()
