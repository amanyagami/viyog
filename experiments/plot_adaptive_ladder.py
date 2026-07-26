"""Plot the adaptive-attack ladder summary figure for the Viyog rebuttal.

Produces ``figs/rebuttal/fig_adaptive_ladder.pdf`` — a two-panel figure that
summarises the full defence-in-depth story across the A0→A3 attacker-knowledge
ladder plus the stochastic-band EOT defence.

Usage::

    uv run python experiments/plot_adaptive_ladder.py
    uv run python experiments/plot_adaptive_ladder.py --out figs/rebuttal/fig_adaptive_ladder.pdf

Data sources (relative to repo root)::

    results/analysis/adaptive_tv_cifar100_*.csv   (18 files)
    results/analysis/eot_stochastic_cifar100_*.csv (18 files)
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure the experiments/ dir is on the path so viyog_plotstyle can be found
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import viyog_plotstyle as vs

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_REPO_ROOT = pathlib.Path(__file__).parent.parent
_RESULTS_DIR = _REPO_ROOT.parent / "results" / "analysis"

# Rung x-positions
_RUNG_X: list[float] = [0, 1, 2, 3, 4.2]
_RUNG_LABELS: list[str] = [
    "A0\n(PGD)",
    "A1\nnorm-presv",
    "A2\nTV-aware",
    "A3\nboth-aware",
    "EOT\nstochastic",
]

# Architecture families for scatter colouring
_FAMILY_COLORS: dict[str, str] = {
    "resnet": vs.C_OURS,
    "densenet": "#56B4E9",
    "mobile": vs.C_ACCENT,
}


def _family_color(model: str) -> str:
    """Return a colour for *model* based on architecture family.

    Args:
        model: Model name string, e.g. ``'resnet50'``.

    Returns:
        A hex colour string.
    """
    for prefix, colour in _FAMILY_COLORS.items():
        if model.startswith(prefix):
            return colour
    return vs.C_REF


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_adaptive_tv(results_dir: pathlib.Path) -> pd.DataFrame:
    """Load and concatenate all adaptive-TV CSV files.

    Args:
        results_dir: Directory that contains ``adaptive_tv_cifar100_*.csv``.

    Returns:
        Concatenated :class:`pandas.DataFrame` with an added ``model`` column.
    """
    paths = sorted(results_dir.glob("adaptive_tv_cifar100_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No adaptive_tv CSV files found in {results_dir}")
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        df["model"] = p.stem.replace("adaptive_tv_cifar100_", "")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_eot_stochastic(results_dir: pathlib.Path) -> pd.DataFrame:
    """Load and concatenate all EOT-stochastic CSV files.

    Args:
        results_dir: Directory that contains ``eot_stochastic_cifar100_*.csv``.

    Returns:
        Concatenated :class:`pandas.DataFrame` with an added ``model`` column.
    """
    paths = sorted(results_dir.glob("eot_stochastic_cifar100_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No eot_stochastic CSV files found in {results_dir}")
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        df["model"] = p.stem.replace("eot_stochastic_cifar100_", "")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Rung aggregation
# ---------------------------------------------------------------------------


def compute_rung_means(
    tv_df: pd.DataFrame,
    eot_df: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """Compute worst-case-lambda means for each attacker-knowledge rung.

    Args:
        tv_df: Concatenated adaptive-TV data (all 18 models).
        eot_df: Concatenated EOT-stochastic data (all 18 models).

    Returns:
        Dict keyed by rung name, each value a sub-dict with keys
        ``auroc_tvdorm``, ``auroc_hf``, ``attack_success`` (and for EOT
        also ``T2_stoch``, ``T2_fixed``).
    """

    def _worst(df: pd.DataFrame, mode: str, col: str) -> float:
        sub = df[df["mode"] == mode]
        lam_max = sub["lambda"].max()
        return float(sub[sub["lambda"] == lam_max][col].mean())

    def _worst_row(df: pd.DataFrame, mode: str) -> dict[str, float]:
        sub = df[df["mode"] == mode]
        lam_max = sub["lambda"].max()
        row = sub[sub["lambda"] == lam_max]
        return {
            "auroc_tvdorm": float(row["auroc_tvdorm"].mean()),
            "auroc_hf": float(row["auroc_hf"].mean()),
            "attack_success": float(row["attack_success"].mean()),
        }

    a0_sub = tv_df[(tv_df["mode"] == "pgd") & (tv_df["lambda"] == 0)]
    a0: dict[str, float] = {
        "auroc_tvdorm": float(a0_sub["auroc_tvdorm"].mean()),
        "auroc_hf": float(a0_sub["auroc_hf"].mean()),
        "attack_success": float(a0_sub["attack_success"].mean()),
    }

    a1 = _worst_row(tv_df, "normpresv")
    a2 = _worst_row(tv_df, "tvaware")
    a3 = _worst_row(tv_df, "allaware")

    # EOT: worst lambda for mode='eot'
    eot_sub = eot_df[eot_df["mode"] == "eot"]
    lam_max_eot = eot_sub["lambda"].max()
    eot_row = eot_sub[eot_sub["lambda"] == lam_max_eot]
    eot: dict[str, float] = {
        "T2_stoch": float(eot_row["T2_stoch"].mean()),
        "T2_fixed": float(eot_row["T2_fixed"].mean()),
        "attack_success": float(eot_row["attack_success"].mean()),
    }

    return {"A0": a0, "A1": a1, "A2": a2, "A3": a3, "EOT": eot}


# ---------------------------------------------------------------------------
# Panel (a) — adaptive ladder
# ---------------------------------------------------------------------------


def draw_panel_a(
    ax: plt.Axes,
    rungs: dict[str, dict[str, float]],
) -> None:
    """Draw the adaptive-ladder AUROC panel.

    Args:
        ax: Target :class:`matplotlib.axes.Axes`.
        rungs: Rung means from :func:`compute_rung_means`.
    """
    xs = _RUNG_X  # [0, 1, 2, 3, 3.8]

    # --- three AUROC lines -------------------------------------------------
    v_tvdorm = [
        rungs["A0"]["auroc_tvdorm"],
        rungs["A1"]["auroc_tvdorm"],
        rungs["A2"]["auroc_tvdorm"],
        rungs["A3"]["auroc_tvdorm"],
        rungs["EOT"]["T2_stoch"],
    ]
    v_hf = [
        rungs["A0"]["auroc_hf"],
        rungs["A1"]["auroc_hf"],
        rungs["A2"]["auroc_hf"],
        rungs["A3"]["auroc_hf"],
        rungs["EOT"]["T2_stoch"],  # stochastic defence restores both
    ]
    v_eot_fixed = rungs["EOT"]["T2_fixed"]

    ax.plot(
        xs,
        v_tvdorm,
        color=vs.C_OURS,
        linestyle="-",
        linewidth=2.5,
        marker="o",
        markersize=7,
        label="Deployed V(x)",
        zorder=3,
    )
    ax.plot(
        xs,
        v_hf,
        color=vs.C_COMP,
        linestyle="--",
        linewidth=2.0,
        marker="s",
        markersize=6,
        label="Complement",
        zorder=3,
    )
    # Fixed-band EOT: single point at EOT x position
    ax.plot(
        [xs[-1]],
        [v_eot_fixed],
        color=vs.C_BASE,
        linestyle=":",
        linewidth=1.5,
        marker="x",
        markersize=7,
        label="Fixed band (EOT)",
        zorder=3,
    )

    # --- chance reference line --------------------------------------------
    ax.axhline(0.5, color=vs.C_REF, linestyle=":", linewidth=1.0, alpha=0.6, zorder=1)
    ax.text(3.85, 0.505, "chance", fontsize=6, color=vs.C_REF, va="bottom")

    # --- right twin axis: attack success ----------------------------------
    ax_r = ax.twinx()
    atk_vals = [
        rungs["A0"]["attack_success"] * 100,
        rungs["A1"]["attack_success"] * 100,
        rungs["A2"]["attack_success"] * 100,
        rungs["A3"]["attack_success"] * 100,
    ]
    ax_r.plot(
        xs[:4],
        atk_vals,
        color=vs.C_REF,
        linestyle="--",
        linewidth=1.2,
        marker=None,
        alpha=0.55,
        zorder=2,
    )
    ax_r.set_ylabel("Attack success (%)", fontsize=7, color=vs.C_REF)
    ax_r.tick_params(axis="y", labelsize=6, colors=vs.C_REF)
    ax_r.set_ylim(60, 115)
    ax_r.spines["right"].set_visible(True)
    ax_r.spines["top"].set_visible(False)

    # --- annotations -------------------------------------------------------
    # A1: "0% evasion (P1)" above the V line
    ax.annotate(
        "0% evasion\n(P1)",
        xy=(xs[1], v_tvdorm[1]),
        xytext=(xs[1] - 0.05, v_tvdorm[1] + 0.055),
        fontsize=6.5,
        color=vs.C_OURS,
        ha="center",
        va="bottom",
        arrowprops=dict(arrowstyle="->", color=vs.C_OURS, lw=0.8),
    )

    # A2→A3 gap: attack success cost
    a2_atk = rungs["A2"]["attack_success"] * 100
    a3_atk = rungs["A3"]["attack_success"] * 100
    cost = abs(a2_atk - a3_atk)
    mid_x = (xs[2] + xs[3]) / 2
    mid_y_auroc = (v_tvdorm[2] + v_tvdorm[3]) / 2
    ax.annotate(
        f"{cost:.0f}% success\ncost",
        xy=(mid_x, mid_y_auroc),
        xytext=(mid_x - 0.45, mid_y_auroc - 0.08),
        fontsize=6.5,
        color=vs.C_BASE,
        ha="center",
        va="top",
        arrowprops=dict(arrowstyle="->", color=vs.C_BASE, lw=0.8),
    )

    # EOT stoch point annotation
    ax.text(
        xs[-1] + 0.04,
        rungs["EOT"]["T2_stoch"] + 0.025,
        f"{rungs['EOT']['T2_stoch']:.3f} ↑",
        fontsize=7,
        color=vs.C_ACCENT,
        fontweight="bold",
        va="bottom",
        ha="left",
    )

    # --- axes formatting ---------------------------------------------------
    ax.set_xlim(-0.4, 4.4)
    ax.set_ylim(0.45, 1.05)
    ax.set_xticks(xs)
    ax.set_xticklabels(_RUNG_LABELS, fontsize=7)
    ax.set_ylabel("OOD-vs-ADV AUROC", fontsize=8)
    ax.set_title("(a) The adaptive ladder", fontsize=9, fontweight="bold", pad=4)
    ax.legend(loc="lower left", fontsize=6.5, framealpha=0.85)


# ---------------------------------------------------------------------------
# Panel (b) — EOT scatter
# ---------------------------------------------------------------------------


def draw_panel_b(
    ax: plt.Axes,
    eot_df: pd.DataFrame,
) -> None:
    """Draw the per-architecture EOT scatter panel.

    Args:
        ax: Target :class:`matplotlib.axes.Axes`.
        eot_df: Concatenated EOT-stochastic data frame.
    """
    sub = eot_df[(eot_df["mode"] == "eot") & (eot_df["lambda"] == eot_df["lambda"].max())]

    x_vals = sub["T2_fixed"].values
    y_vals = sub["T2_stoch"].values
    models = sub["model"].values

    colors = [_family_color(m) for m in models]

    # --- diagonal reference -----------------------------------------------
    lim_min = min(x_vals.min(), y_vals.min()) - 0.02
    lim_max = max(x_vals.max(), y_vals.max()) + 0.02
    diag = np.array([lim_min, lim_max])
    ax.plot(diag, diag, color=vs.C_REF, linewidth=1.0, linestyle="-", zorder=1)

    # --- green shading above diagonal ------------------------------------
    ax.fill_between(
        diag,
        diag,
        np.full_like(diag, lim_max),
        color="#009E73",
        alpha=0.07,
        zorder=0,
    )

    # --- scatter ----------------------------------------------------------
    ax.scatter(x_vals, y_vals, c=colors, s=32, zorder=3, edgecolors="white", linewidths=0.4)

    # --- "17/18 above line" annotation ------------------------------------
    above = int((y_vals > x_vals).sum())
    ax.text(
        0.96,
        0.05,
        f"{above}/{len(y_vals)} above line",
        transform=ax.transAxes,
        fontsize=7,
        color=vs.C_OURS,
        ha="right",
        va="bottom",
        fontweight="bold",
    )

    # --- family legend ----------------------------------------------------
    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=vs.C_OURS,
            markersize=6,
            label="ResNets",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#56B4E9",
            markersize=6,
            label="DenseNets",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=vs.C_ACCENT,
            markersize=6,
            label="MobileNets",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=vs.C_REF,
            markersize=6,
            label="Others",
        ),
    ]
    ax.legend(handles=legend_handles, fontsize=6, loc="upper left", framealpha=0.85)

    # --- axes formatting --------------------------------------------------
    ax.set_xlabel("fixed band (EOT)", fontsize=8)
    ax.set_ylabel("stochastic band (EOT)", fontsize=8)
    ax.set_title(
        "(b) EOT defence: 20 architectures",
        fontsize=9,
        fontweight="bold",
        pad=4,
    )
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_aspect("equal")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_figure(
    results_dir: pathlib.Path,
    out_path: pathlib.Path,
) -> dict[str, dict[str, float]]:
    """Build and save the adaptive-ladder figure.

    Args:
        results_dir: Directory containing the analysis CSVs.
        out_path: Destination PDF path.

    Returns:
        Dict of per-rung means (same structure as :func:`compute_rung_means`).
    """
    vs.apply_style()

    tv_df = load_adaptive_tv(results_dir)
    eot_df = load_eot_stochastic(results_dir)

    rungs = compute_rung_means(tv_df, eot_df)

    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(7.0, 3.2),
        gridspec_kw={"width_ratios": [1.4, 1]},
    )

    draw_panel_a(ax_a, rungs)
    draw_panel_b(ax_b, eot_df)

    fig.tight_layout(pad=1.2, w_pad=2.0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    vs.savefig_pdf(fig, str(out_path))
    plt.close(fig)

    return rungs


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Parsed :class:`argparse.Namespace`.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        default="figs/rebuttal/fig_adaptive_ladder.pdf",
        help="Output PDF path (relative to repo root or absolute).",
    )
    p.add_argument(
        "--results-dir",
        default=None,
        help="Override path to results/analysis directory.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the adaptive-ladder plot script.

    Args:
        argv: Optional argument list for testing; defaults to ``sys.argv[1:]``.
    """
    args = _parse_args(argv)

    results_dir = pathlib.Path(args.results_dir) if args.results_dir else _RESULTS_DIR

    out_path = (
        pathlib.Path(args.out) if pathlib.Path(args.out).is_absolute() else _REPO_ROOT / args.out
    )

    rungs = build_figure(results_dir, out_path)

    # --- print rung means -------------------------------------------------
    print("Rung means (mean over 18 models, worst-case lambda):")
    print(
        f"  A0 (PGD)        auroc_tvdorm={rungs['A0']['auroc_tvdorm']:.4f}  "
        f"auroc_hf={rungs['A0']['auroc_hf']:.4f}  "
        f"attack_success={rungs['A0']['attack_success']:.4f}"
    )
    print(
        f"  A1 (norm-presv) auroc_tvdorm={rungs['A1']['auroc_tvdorm']:.4f}  "
        f"auroc_hf={rungs['A1']['auroc_hf']:.4f}  "
        f"attack_success={rungs['A1']['attack_success']:.4f}"
    )
    print(
        f"  A2 (TV-aware)   auroc_tvdorm={rungs['A2']['auroc_tvdorm']:.4f}  "
        f"auroc_hf={rungs['A2']['auroc_hf']:.4f}  "
        f"attack_success={rungs['A2']['attack_success']:.4f}"
    )
    print(
        f"  A3 (both-aware) auroc_tvdorm={rungs['A3']['auroc_tvdorm']:.4f}  "
        f"auroc_hf={rungs['A3']['auroc_hf']:.4f}  "
        f"attack_success={rungs['A3']['attack_success']:.4f}"
    )
    print(
        f"  EOT             T2_stoch={rungs['EOT']['T2_stoch']:.4f}  "
        f"T2_fixed={rungs['EOT']['T2_fixed']:.4f}  "
        f"attack_success={rungs['EOT']['attack_success']:.4f}"
    )

    size_kb = out_path.stat().st_size / 1024
    if size_kb < 10:
        print(f"WARNING: PDF may be too small ({size_kb:.1f} KB)")
    else:
        print(f"PDF OK: {size_kb:.1f} KB  →  {out_path}")


if __name__ == "__main__":
    main()
