"""Full two-stage Viyog deployment pipeline — horizontal banner diagram.

Produces ``fig_pipeline.pdf`` for §IV Method (Deployment section).
Layout is a compact LEFT→RIGHT flow across a 7.0 × 1.9 inch double-column
figure.

Run::

    uv run python experiments/plot_pipeline.py \
        --out figs/rebuttal/fig_pipeline.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).parent))
from viyog_plotstyle import C_COMP, C_OURS, apply_style

apply_style()

# ---------------------------------------------------------------------------
# Colour / style constants
# ---------------------------------------------------------------------------
BG = "#F7F9FC"
ARROW_C = "#444444"
BOX_INPUT_FC = "#EBEBEB"
BOX_INPUT_EC = "#888888"
BOX_DNN_FC = "#D6E8F7"
BOX_DNN_EC = C_COMP
BOX_GATE_FC = "#FEF0DC"
BOX_GATE_EC = "#E69F00"
BOX_HOOK_FC = "#E8F8F0"
BOX_HOOK_EC = C_OURS
BOX_VIYOG_FC = "#E8F8F0"
BOX_VIYOG_EC = C_OURS
BOX_ID_FC = "#F0F0F0"
BOX_ID_EC = "#888888"
BOX_OOD_FC = "#D6E8F7"
BOX_OOD_EC = C_COMP
BOX_ADV_FC = "#FDDEDE"
BOX_ADV_EC = "#CC0000"
DASHED_RECT_EC = C_OURS
HOOK_DOT_C = C_OURS


def _smooth_patch(rng: np.random.Generator, size: int = 28) -> np.ndarray:
    """Return a smooth (size, size) float32 image via box-blur."""
    im = rng.standard_normal((size, size)).astype(np.float32)
    pad = np.pad(im, 2, mode="edge")
    for _ in range(5):
        im = (
            sum(pad[r : r + size, s : s + size] for r in range(5) for s in range(5)).astype(
                np.float32
            )
            / 25.0
        )
        pad = np.pad(im, 2, mode="edge")
    mn, mx = im.min(), im.max()
    return (im - mn) / (mx - mn + 1e-9)


# ---------------------------------------------------------------------------
# Low-level drawing helpers — all coordinates in *figure* fraction (0..1)
# ---------------------------------------------------------------------------


def _fancy_box(
    fig: plt.Figure,
    cx: float,
    cy: float,
    w: float,
    h: float,
    fc: str,
    ec: str,
    lw: float = 1.4,
    pad: float = 0.012,
    zorder: int = 4,
) -> None:
    """Draw a rounded-rect FancyBboxPatch centred at (cx, cy) in figure fraction."""
    fig.add_artist(
        FancyBboxPatch(
            (cx - w / 2, cy - h / 2),
            w,
            h,
            boxstyle=f"round,pad={pad}",
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
            transform=fig.transFigure,
            zorder=zorder,
            clip_on=False,
        )
    )


def _text(
    fig: plt.Figure,
    x: float,
    y: float,
    s: str,
    fontsize: float = 7.0,
    fontweight: str = "normal",
    color: str = "#111111",
    ha: str = "center",
    va: str = "center",
    zorder: int = 6,
    style: str = "normal",
    linespacing: float = 1.35,
) -> None:
    """Place text at figure-fraction coordinates."""
    fig.text(
        x,
        y,
        s,
        fontsize=fontsize,
        fontweight=fontweight,
        color=color,
        ha=ha,
        va=va,
        zorder=zorder,
        style=style,
        linespacing=linespacing,
    )


def _arrow(
    fig: plt.Figure,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    color: str = ARROW_C,
    lw: float = 1.4,
    rad: float = 0.0,
    zorder: int = 8,
) -> None:
    """Draw an arrow between two figure-fraction points."""
    fig.add_artist(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            transform=fig.transFigure,
            arrowstyle="->,head_width=0.010,head_length=0.008",
            connectionstyle=f"arc3,rad={rad}",
            color=color,
            lw=lw,
            zorder=zorder,
            clip_on=False,
        )
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Build and save the Viyog deployment pipeline diagram."""
    ap = argparse.ArgumentParser(description="Draw the Viyog deployment pipeline diagram.")
    ap.add_argument(
        "--out",
        default="figs/rebuttal/fig_pipeline.pdf",
        help="Output PDF path.",
    )
    args = ap.parse_args()

    rng = np.random.default_rng(42)

    # ── figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(7.0, 2.05))
    fig.patch.set_facecolor(BG)

    # A single invisible axes covering the whole figure — used only for the
    # 28×28 thumbnail inset.  All other drawing uses fig.transFigure directly.
    ax_bg = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax_bg.set_axis_off()
    ax_bg.set_facecolor(BG)
    ax_bg.grid(False)

    # ── layout parameters (all in figure-fraction units) ──────────────────────
    # Vertical centre of the main flow
    MID_Y = 0.50
    # Five main horizontal positions for the five stages
    X_INPUT = 0.07  # Stage 1 — Input
    X_DNN = 0.24  # Stage 2 — DNN Backbone
    X_GATE = 0.44  # Stage 3a — OOD Gate (above midline)
    X_HOOK = 0.44  # Stage 3b — Hook box (below midline)
    X_VIYOG = 0.63  # Stage 4 — Viyog (centre)
    X_OUT = 0.86  # Stage 5 — Outputs

    BOX_W_SMALL = 0.085
    BOX_W_MED = 0.110
    BOX_W_VIYOG = 0.130
    BOX_W_OUT = 0.120
    BOX_H_MAIN = 0.38  # height for stages 1-2 (in figure fraction, figure is 1.9 in tall)
    BOX_H_BRANCH = 0.26  # height for branch boxes (gate / hook)
    BOX_H_VIYOG = 0.50  # taller box for Viyog
    BOX_H_OUT = 0.20  # individual output box height

    GATE_Y = MID_Y + 0.25  # stage 3a vertical position
    HOOK_Y = MID_Y - 0.25  # stage 3b vertical position
    OUT_TOP_Y = MID_Y + 0.28  # ID output
    OUT_MID_Y = MID_Y  # OOD output
    OUT_BOT_Y = MID_Y - 0.28  # ADV output

    # ── dashed rectangle for "Viyog (this paper)" second-stage region ─────────
    # Enclose Stage 3b hook + Stage 4 in the dashed rect
    DASHED_X0 = X_HOOK - BOX_W_MED / 2 - 0.015
    DASHED_Y0 = 0.04
    DASHED_W = X_VIYOG + BOX_W_VIYOG / 2 + 0.015 - DASHED_X0
    DASHED_H = 0.86
    fig.add_artist(
        FancyBboxPatch(
            (DASHED_X0, DASHED_Y0),
            DASHED_W,
            DASHED_H,
            boxstyle="round,pad=0.008",
            facecolor="none",
            edgecolor=DASHED_RECT_EC,
            linewidth=1.0,
            linestyle=(0, (4, 3)),
            transform=fig.transFigure,
            zorder=2,
            clip_on=False,
        )
    )
    # Label for the dashed region (sits in the top label row, between the
    # "Full forward" and "Outputs" stage headers — no per-stage labels collide).
    _text(
        fig,
        DASHED_X0 + DASHED_W / 2,
        0.955,
        "Viyog (this paper)",
        fontsize=6.8,
        fontweight="bold",
        color=DASHED_RECT_EC,
        zorder=9,
    )

    # ── Stage 1: Input ─────────────────────────────────────────────────────────
    _fancy_box(fig, X_INPUT, MID_Y, BOX_W_SMALL, BOX_H_MAIN, BOX_INPUT_FC, BOX_INPUT_EC, lw=1.2)

    # 28×28 thumbnail — use an inset_axes on ax_bg
    thumb_img = _smooth_patch(rng, 28)
    # Convert figure-fraction box corners to axes-fraction (ax_bg spans [0,1]×[0,1])
    thumb_w_f = 0.046
    thumb_h_f = 0.28
    ins = ax_bg.inset_axes(
        [X_INPUT - thumb_w_f / 2, MID_Y + 0.03, thumb_w_f, thumb_h_f],
        transform=fig.transFigure,
    )
    ins.imshow(thumb_img, cmap="Greys_r", aspect="equal", interpolation="nearest")
    ins.set_xticks([])
    ins.set_yticks([])
    for sp in ins.spines.values():
        sp.set_color(BOX_INPUT_EC)
        sp.set_linewidth(0.8)

    _text(fig, X_INPUT, MID_Y + 0.01, "Input x", fontsize=6.5, fontweight="bold")
    _text(
        fig,
        X_INPUT,
        MID_Y - 0.12,
        "(normal, OOD,\nor ADV)",
        fontsize=5.5,
        color="#555555",
        style="italic",
    )

    # Arrow Stage 1 → Stage 2
    _arrow(fig, X_INPUT + BOX_W_SMALL / 2, MID_Y, X_DNN - BOX_W_MED / 2, MID_Y)

    # ── Stage 2: DNN Backbone ──────────────────────────────────────────────────
    _fancy_box(fig, X_DNN, MID_Y, BOX_W_MED, BOX_H_MAIN, BOX_DNN_FC, BOX_DNN_EC, lw=1.4)
    _text(
        fig,
        X_DNN,
        MID_Y + 0.09,
        "DNN Backbone\nf(x)",
        fontsize=6.5,
        fontweight="bold",
        color=BOX_DNN_EC,
    )
    _text(
        fig,
        X_DNN,
        MID_Y - 0.12,
        "prediction\n+ logits",
        fontsize=5.5,
        color="#555555",
        style="italic",
    )

    # Green "hook" circle on the left side of the DNN box
    hook_x_fig = X_DNN - BOX_W_MED / 2
    hook_y_fig = MID_Y - 0.08
    fig.add_artist(
        mpatches.Circle(
            (hook_x_fig, hook_y_fig),
            radius=0.012,
            facecolor=HOOK_DOT_C,
            edgecolor="white",
            linewidth=0.8,
            transform=fig.transFigure,
            zorder=7,
            clip_on=False,
        )
    )
    _text(
        fig,
        hook_x_fig - 0.02,
        hook_y_fig + 0.02,
        "hook",
        fontsize=5.2,
        color=C_OURS,
        fontweight="bold",
        ha="right",
    )

    # ── Branch arrows from Stage 2 ─────────────────────────────────────────────
    # UP arrow → Stage 3a OOD gate
    _arrow(
        fig,
        X_DNN + BOX_W_MED / 2,
        MID_Y + 0.08,
        X_GATE - BOX_W_MED / 2,
        GATE_Y,
        color=BOX_GATE_EC,
        lw=1.2,
        rad=-0.2,
    )
    # Hook arrow from DNN → Stage 3b hook box (downward)
    _arrow(
        fig,
        hook_x_fig,
        hook_y_fig,
        X_HOOK - BOX_W_MED / 2,
        HOOK_Y,
        color=C_OURS,
        lw=1.2,
        rad=0.25,
    )

    # ── Stage 3a: First-stage OOD Gate ────────────────────────────────────────
    _fancy_box(fig, X_GATE, GATE_Y, BOX_W_MED, BOX_H_BRANCH, BOX_GATE_FC, BOX_GATE_EC, lw=1.3)
    _text(
        fig,
        X_GATE,
        GATE_Y + 0.05,
        "Stage 1\nOOD Gate",
        fontsize=6.5,
        fontweight="bold",
        color=BOX_GATE_EC,
    )
    _text(
        fig,
        X_GATE,
        GATE_Y - 0.07,
        "Energy / MSP\n(logit score)",
        fontsize=5.5,
        color="#555555",
        style="italic",
    )

    # Arrow from OOD Gate → Viyog (if flagged as non-ID)
    _arrow(
        fig,
        X_GATE + BOX_W_MED / 2,
        GATE_Y,
        X_VIYOG - BOX_W_VIYOG / 2,
        MID_Y + 0.12,
        color=BOX_GATE_EC,
        lw=1.2,
        rad=-0.15,
    )
    _text(
        fig,
        (X_GATE + BOX_W_MED / 2 + X_VIYOG - BOX_W_VIYOG / 2) / 2,
        GATE_Y + 0.075,
        "if non-ID",
        fontsize=5.2,
        color=BOX_GATE_EC,
        style="italic",
    )

    # ── Stage 3b: First-conv Hook activations ─────────────────────────────────
    _fancy_box(fig, X_HOOK, HOOK_Y, BOX_W_MED, BOX_H_BRANCH, BOX_HOOK_FC, BOX_HOOK_EC, lw=1.2)
    _text(
        fig,
        X_HOOK,
        HOOK_Y + 0.04,
        "hook: first-conv\nactivations a",
        fontsize=6.0,
        fontweight="bold",
        color=BOX_HOOK_EC,
    )
    _text(
        fig,
        X_HOOK,
        HOOK_Y - 0.08,
        r"$a\in\mathbb{R}^{C\times H\times W}$",
        fontsize=6.0,
        color="#555555",
        style="italic",
    )

    # Arrow from hook box → Viyog-D
    _arrow(
        fig,
        X_HOOK + BOX_W_MED / 2,
        HOOK_Y,
        X_VIYOG - BOX_W_VIYOG / 2,
        MID_Y - 0.12,
        color=C_OURS,
        lw=1.2,
        rad=0.15,
    )

    # ── Stage 4: Viyog (main highlighted box) ────────────────────────────────
    # Thick coloured border, larger box
    _fancy_box(
        fig,
        X_VIYOG,
        MID_Y,
        BOX_W_VIYOG,
        BOX_H_VIYOG,
        BOX_VIYOG_FC,
        BOX_VIYOG_EC,
        lw=2.2,
        pad=0.016,
        zorder=4,
    )
    _text(
        fig,
        X_VIYOG,
        MID_Y + 0.21,
        "Viyog",
        fontsize=8.5,
        fontweight="bold",
        color=C_OURS,
        zorder=6,
    )
    # Formula inside the box
    _text(
        fig,
        X_VIYOG,
        MID_Y + 0.05,
        r"$V(x)=\langle\widetilde{\mathrm{TV}}(a_c)\rangle_{c\in\mathcal{B}}$",
        fontsize=7.5,
        color="#1A3A1A",
        zorder=6,
    )
    _text(
        fig,
        X_VIYOG,
        MID_Y - 0.13,
        "0.3 KB · no extra\nforward pass",
        fontsize=5.8,
        color="#555555",
        style="italic",
        zorder=6,
    )

    # Arrow Stage 4 → Output column
    # We draw three arrows below, one per output
    for out_y, col in [
        (OUT_TOP_Y, BOX_ID_EC),
        (OUT_MID_Y, BOX_OOD_EC),
        (OUT_BOT_Y, BOX_ADV_EC),
    ]:
        _arrow(
            fig,
            X_VIYOG + BOX_W_VIYOG / 2,
            MID_Y + (out_y - MID_Y) * 0.35,
            X_OUT - BOX_W_OUT / 2,
            out_y,
            color=col,
            lw=1.1,
            rad=0.0,
        )

    # ── Stage 5: Three outputs ─────────────────────────────────────────────────
    # ID
    _fancy_box(fig, X_OUT, OUT_TOP_Y, BOX_W_OUT, BOX_H_OUT, BOX_ID_FC, BOX_ID_EC, lw=1.1)
    _text(
        fig,
        X_OUT,
        OUT_TOP_Y + 0.03,
        "ID ✓",
        fontsize=7.0,
        fontweight="bold",
        color="#555555",
    )
    _text(
        fig,
        X_OUT,
        OUT_TOP_Y - 0.05,
        "normal prediction",
        fontsize=5.5,
        color="#555555",
        style="italic",
    )

    # OOD
    _fancy_box(fig, X_OUT, OUT_MID_Y, BOX_W_OUT, BOX_H_OUT, BOX_OOD_FC, BOX_OOD_EC, lw=1.2)
    _text(
        fig,
        X_OUT,
        OUT_MID_Y + 0.03,
        "OOD →",
        fontsize=7.0,
        fontweight="bold",
        color=BOX_OOD_EC,
    )
    _text(
        fig,
        X_OUT,
        OUT_MID_Y - 0.05,
        "abstain / review",
        fontsize=5.5,
        color=BOX_OOD_EC,
        style="italic",
    )

    # ADV
    _fancy_box(fig, X_OUT, OUT_BOT_Y, BOX_W_OUT, BOX_H_OUT, BOX_ADV_FC, BOX_ADV_EC, lw=1.2)
    _text(
        fig,
        X_OUT,
        OUT_BOT_Y + 0.03,
        "ADV ✗",
        fontsize=7.0,
        fontweight="bold",
        color=BOX_ADV_EC,
    )
    _text(
        fig,
        X_OUT,
        OUT_BOT_Y - 0.05,
        "reject / alert",
        fontsize=5.5,
        color=BOX_ADV_EC,
        style="italic",
    )

    # ── Stage labels above each major box ─────────────────────────────────────
    # Only label the columns NOT covered by the "Viyog (this paper)" header,
    # so nothing collides at the top edge.
    TOP_LABEL_Y = 0.955
    for x, lbl, col in [
        (X_INPUT, "Input", "#888888"),
        (X_DNN, "Full model forward", BOX_DNN_EC),
        (X_OUT, "Decision", "#888888"),
    ]:
        _text(fig, x, TOP_LABEL_Y, lbl, fontsize=6.0, fontweight="bold", color=col, zorder=9)

    # ── save ─────────────────────────────────────────────────────────────────
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), bbox_inches="tight", dpi=300)
    plt.close(fig)
    size_bytes = out.stat().st_size
    size_kb = size_bytes // 1024
    print(f"saved → {out}  ({size_kb} KB)")


if __name__ == "__main__":
    main()
