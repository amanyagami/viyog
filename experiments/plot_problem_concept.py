"""Concept diagram: why OOD and ADV are different threats that need different responses.

Left-to-right story, four stages:
  (1) Input space   -- ID cluster, with an OOD point clearly OUTSIDE the data and
      an ADV point INSIDE the cluster (crafted to look in-distribution).
  (2) Classifier    -- to the network's logits both look "non-ID", so output-side
      detectors cannot tell which is which.
  (3) Viyog gate  -- a free first-conv dormant-band read that separates them.
  (4) Response       -- OOD => abstain / flag; ADV => reject / alert (opposite handling).

Run::

    uv run python experiments/plot_problem_concept.py \
        --out figs/rebuttal/fig_problem_concept.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from viyog_plotstyle import C_BASE, C_COMP, C_OURS, apply_style

apply_style()

BG = "#FFFFFF"
ELLIPSE_FC = "#CDE8F7"
ELLIPSE_EC = "#4A90D9"
DNN_FC = "#FFF6E0"
DNN_EC = "#B8860B"
GATE_FC = "#E2F6EE"
OOD_FC = "#E2EFFB"
ADV_FC = "#FCE3E1"


def _textbox(
    ax: plt.Axes,
    cx: float,
    cy: float,
    w: float,
    h: float,
    text: str,
    fc: str,
    ec: str,
    fontsize: float = 7.5,
    fontweight: str = "bold",
    color: str = "#111111",
    lw: float = 1.6,
) -> None:
    """Draw a rounded-rectangle label centred at (cx, cy) in axes coords."""
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (cx - w / 2, cy - h / 2),
            w,
            h,
            boxstyle="round,pad=0.012",
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
            transform=ax.transAxes,
            zorder=4,
        )
    )
    ax.text(
        cx,
        cy,
        text,
        transform=ax.transAxes,
        fontsize=fontsize,
        ha="center",
        va="center",
        fontweight=fontweight,
        color=color,
        linespacing=1.25,
        zorder=5,
    )


def _arrow(
    ax: plt.Axes,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    color: str,
    rad: float = 0.0,
    lw: float = 1.8,
) -> None:
    """Draw a curved arrow between two axes-fraction points."""
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(
            arrowstyle="-|>,head_width=0.28,head_length=0.5",
            color=color,
            lw=lw,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=2,
            shrinkB=2,
        ),
        zorder=6,
    )


def _star(ax: plt.Axes, cx: float, cy: float, r: float, color: str) -> None:
    """Draw a small spiky star marker at (cx, cy) in axes fraction."""
    ang = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, 11)
    rad = np.where(np.arange(11) % 2 == 0, r, r * 0.45)
    # correct aspect so the star is not stretched by the wide axes
    xs = cx + rad * np.cos(ang) * 0.45
    ys = cy + rad * np.sin(ang)
    ax.add_patch(
        mpatches.Polygon(
            np.column_stack([xs, ys]),
            closed=True,
            facecolor=color,
            edgecolor="white",
            linewidth=0.8,
            transform=ax.transAxes,
            zorder=7,
        )
    )


def main() -> None:
    """Build and save fig_problem_concept.pdf."""
    ap = argparse.ArgumentParser(description="Draw the OOD-vs-ADV problem concept diagram.")
    ap.add_argument("--out", default="figs/rebuttal/fig_problem_concept.pdf")
    args = ap.parse_args()
    rng = np.random.default_rng(7)

    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.grid(False)

    # four zones along x
    div = [0.40, 0.58, 0.72]
    zone_titles = [
        (0.20, "1.  Input space"),
        (0.49, "2.  Classifier"),
        (0.65, "3.  Viyog"),
        (0.86, "4.  Response"),
    ]
    zcols = ["#EAF4FC", "#F3F0E6", "#E9F8F1", "#F7EFEF"]
    zbnd = [(0.0, 0.40), (0.40, 0.58), (0.58, 0.72), (0.72, 1.0)]
    for (x0, x1), zc in zip(zbnd, zcols):
        ax.axvspan(x0, x1, color=zc, alpha=0.7, zorder=0)
    for xv in div:
        ax.axvline(xv, color="#CBD3E0", lw=0.8, ls=(0, (4, 3)), zorder=1)
    for x, t in zone_titles:
        ax.text(
            x,
            0.955,
            t,
            transform=ax.transAxes,
            fontsize=7.8,
            fontweight="bold",
            ha="center",
            va="top",
            color="#33414F",
            zorder=5,
        )

    # ── ZONE 1: input space ─────────────────────────────────────────────────
    ex, ey, rx, ry = 0.175, 0.50, 0.135, 0.30
    ax.add_patch(
        mpatches.Ellipse(
            (ex, ey),
            2 * rx,
            2 * ry,
            transform=ax.transAxes,
            facecolor=ELLIPSE_FC,
            edgecolor=ELLIPSE_EC,
            linewidth=1.8,
            alpha=0.6,
            zorder=1,
        )
    )
    ax.text(
        ex,
        ey + ry - 0.045,
        "ID  (training data)",
        transform=ax.transAxes,
        fontsize=6.3,
        ha="center",
        va="center",
        color="#2C6BB0",
        style="italic",
        zorder=3,
    )
    # ID dots in the mid band of the ellipse
    pts: list[tuple[float, float]] = []
    while len(pts) < 9:
        x = rng.uniform(ex - rx, ex + rx)
        y = rng.uniform(ey - 0.02, ey + ry - 0.09)
        if ((x - ex) / rx) ** 2 + ((y - ey) / ry) ** 2 < 0.72:
            pts.append((x, y))
    for px, py in pts:
        ax.plot(px, py, "o", ms=4.2, color=C_OURS, alpha=0.85, transform=ax.transAxes, zorder=3)
    # ADV point: inside the cluster, lower area
    advx, advy = 0.135, 0.32
    _star(ax, advx, advy, 0.045, C_BASE)
    ax.text(
        advx + 0.085,
        advy,
        "ADV\ncrafted, looks ID",
        transform=ax.transAxes,
        fontsize=6.0,
        ha="center",
        va="center",
        color=C_BASE,
        fontweight="bold",
        linespacing=1.15,
        zorder=7,
    )
    # OOD point: outside the cluster, to the right
    oodx, oody = 0.345, 0.66
    ax.plot(
        oodx,
        oody,
        "D",
        ms=8.5,
        color=C_COMP,
        transform=ax.transAxes,
        markeredgecolor="white",
        markeredgewidth=0.8,
        zorder=7,
    )
    ax.text(
        oodx,
        oody + 0.115,
        "OOD\nnovel, outside",
        transform=ax.transAxes,
        fontsize=6.0,
        ha="center",
        va="center",
        color=C_COMP,
        fontweight="bold",
        linespacing=1.15,
        zorder=7,
    )

    # ── ZONE 2: classifier ──────────────────────────────────────────────────
    dnx, dny = 0.49, 0.60
    _textbox(ax, dnx, dny, 0.13, 0.17, "DNN\nclassifier", DNN_FC, DNN_EC, 7.2, color=DNN_EC)
    ax.text(
        dnx,
        0.34,
        "logits look\n“non-ID” for\nboth — cannot\ntell which",
        transform=ax.transAxes,
        fontsize=5.9,
        ha="center",
        va="center",
        color="#7A6512",
        style="italic",
        linespacing=1.2,
        zorder=5,
    )
    # inputs -> classifier
    _arrow(ax, oodx + 0.02, oody - 0.02, dnx - 0.07, dny + 0.04, C_COMP, rad=-0.18)
    _arrow(ax, advx + 0.06, advy + 0.03, dnx - 0.07, dny - 0.05, C_BASE, rad=0.22)

    # ── ZONE 3: Viyog gate ────────────────────────────────────────────────
    gx, gy = 0.65, 0.60
    _textbox(ax, gx, gy, 0.12, 0.19, "Viyog", GATE_FC, C_OURS, 8.2, color=C_OURS, lw=2.2)
    ax.text(
        gx,
        0.355,
        "reads first-conv\ndormant-band\nshape  V(x)",
        transform=ax.transAxes,
        fontsize=5.9,
        ha="center",
        va="center",
        color="#1A7A4A",
        style="italic",
        linespacing=1.2,
        zorder=5,
    )
    _arrow(ax, dnx + 0.07, dny, gx - 0.065, gy, "#555555", rad=0.0)

    # ── ZONE 4: responses ───────────────────────────────────────────────────
    rxc = 0.865
    _textbox(
        ax,
        rxc,
        0.74,
        0.245,
        0.20,
        "OOD → abstain /\nflag for review",
        OOD_FC,
        C_COMP,
        7.0,
        color=C_COMP,
    )
    _textbox(
        ax,
        rxc,
        0.30,
        0.245,
        0.20,
        "ADV → reject /\nsecurity alert",
        ADV_FC,
        C_BASE,
        7.0,
        color=C_BASE,
    )
    # gate -> two different responses
    _arrow(ax, gx + 0.06, gy + 0.03, rxc - 0.125, 0.72, C_COMP, rad=-0.15)
    _arrow(ax, gx + 0.06, gy - 0.05, rxc - 0.125, 0.32, C_BASE, rad=0.18)

    # bottom takeaway strip
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (0.04, 0.025),
            0.92,
            0.085,
            boxstyle="round,pad=0.008",
            facecolor="#E9F8F1",
            edgecolor=C_OURS,
            linewidth=1.1,
            transform=ax.transAxes,
            zorder=2,
        )
    )
    ax.text(
        0.50,
        0.068,
        "Same symptom, opposite cure: Viyog separates OOD from ADV from the "
        "first-conv dormant band — no retraining, no extra forward pass.",
        transform=ax.transAxes,
        fontsize=6.6,
        ha="center",
        va="center",
        color="#16603C",
        fontweight="bold",
        zorder=5,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"saved → {out}  ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
