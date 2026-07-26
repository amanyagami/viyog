"""Fig. 6 — How Viyog works, as a clean single-row 5-step academic flow.

White background, generous whitespace, one small visual per step, minimal text,
no overlaps. Replaces the crowded 3-row v3 diagram.

Steps (left -> right):
  1. Hook the first-conv activation map a = f0(x).
  2. Rank channels by mean ID activation; the quietest 10% form the dormant band B.
  3. Read spatial roughness (TV) on B: clean channels are smooth, adversarial jagged.
  4. Score V(x) = mean normalized TV over B; ID/OOD stay low, ADV spikes past tau.
  5. Route: ID pass / OOD abstain / ADV reject.

Run::

    uv run python experiments/plot_viyog_diagram_v4.py \
        --out figs/rebuttal/fig_viyog_diagram.pdf
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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).parent))
from viyog_plotstyle import C_BASE, C_COMP, C_OURS, apply_style

apply_style()

INK = "#1F2933"
MUTED = "#7A8694"
CARD_EC = "#C9D3DF"
CARD_FC = "#FFFFFF"


def _smooth(rng: np.random.Generator, n: int = 26) -> np.ndarray:
    """A smooth (clean / natural) activation patch via repeated box blur."""
    im = rng.standard_normal((n, n)).astype(np.float32)
    for _ in range(6):
        p = np.pad(im, 1, mode="edge")
        im = sum(p[a : a + n, b : b + n] for a in range(3) for b in range(3)) / 9.0
    im -= im.min()
    return im / (im.max() + 1e-9)


def _jagged(rng: np.random.Generator, n: int = 26) -> np.ndarray:
    """A smooth base with injected high-frequency residue (adversarial-like)."""
    base = _smooth(rng, n)
    hf = rng.standard_normal((n, n)).astype(np.float32)
    im = base + 0.55 * hf
    im -= im.min()
    return im / (im.max() + 1e-9)


def _card(fig, x0, y0, w, h, title, num):
    """Draw a rounded card with a numbered step title; return inner content box."""
    fig.add_artist(
        FancyBboxPatch(
            (x0, y0),
            w,
            h,
            boxstyle="round,pad=0.006",
            facecolor=CARD_FC,
            edgecolor=CARD_EC,
            linewidth=1.1,
            transform=fig.transFigure,
            zorder=2,
            mutation_aspect=0.5,
        )
    )
    # number badge
    fig.add_artist(
        plt.matplotlib.patches.Circle(
            (x0 + 0.018, y0 + h - 0.05),
            0.013,
            facecolor=C_OURS,
            edgecolor="none",
            transform=fig.transFigure,
            zorder=4,
        )
    )
    fig.text(
        x0 + 0.018,
        y0 + h - 0.05,
        str(num),
        ha="center",
        va="center",
        fontsize=7,
        color="white",
        fontweight="bold",
        zorder=5,
    )
    fig.text(
        x0 + 0.038,
        y0 + h - 0.05,
        title,
        ha="left",
        va="center",
        fontsize=7.4,
        color=INK,
        fontweight="bold",
        zorder=5,
    )


def _arrow(fig, x0, x1, y):
    """Short horizontal connector arrow between two cards at height y."""
    fig.add_artist(
        FancyArrowPatch(
            (x0, y),
            (x1, y),
            transform=fig.transFigure,
            arrowstyle="-|>,head_width=2.4,head_length=3.2",
            color=MUTED,
            lw=1.4,
            mutation_scale=1.0,
            zorder=3,
        )
    )


def main() -> None:
    """Build and save the clean 5-step Viyog diagram."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="figs/rebuttal/fig_viyog_diagram.pdf")
    args = ap.parse_args()
    rng = np.random.default_rng(3)

    fig = plt.figure(figsize=(7.16, 2.35))
    fig.patch.set_facecolor("#FFFFFF")
    bg = fig.add_axes([0, 0, 1, 1])
    bg.set_axis_off()

    fig.text(
        0.5,
        0.95,
        "How Viyog works — one forward pass, no extra parameters",
        ha="center",
        va="center",
        fontsize=8.8,
        fontweight="bold",
        color=INK,
    )

    # five evenly spaced cards
    n = 5
    gap = 0.018
    margin = 0.012
    cw = (1.0 - 2 * margin - (n - 1) * gap) / n
    y0, ch = 0.10, 0.70
    xs = [margin + i * (cw + gap) for i in range(n)]
    titles = ["Hook first conv", "Dormant band B", "Read shape (TV)", "Score V(x)", "Route"]
    for x, t, k in zip(xs, titles, range(1, n + 1)):
        _card(fig, x, y0, cw, ch, t, k)
    for i in range(n - 1):
        _arrow(fig, xs[i] + cw + 0.002, xs[i + 1] - 0.002, y0 + ch * 0.5)

    def inset(ci, rx, ry, rw, rh):
        """Axes inside card ci (figure-fraction coords), drawn above the card."""
        iax = fig.add_axes([xs[ci] + rx * cw, y0 + ry * ch, rw * cw, rh * ch])
        iax.set_zorder(6)  # above the white card patch (zorder 2)
        return iax

    # ── Step 1: hook — input thumbnail + activation map ──────────────────────
    a1 = inset(0, 0.16, 0.42, 0.68, 0.40)
    a1.imshow(_smooth(rng), cmap="viridis")
    a1.set_xticks([])
    a1.set_yticks([])
    fig.text(
        xs[0] + cw / 2,
        y0 + 0.10 * ch,
        r"$a=f_0(x)\in\mathbb{R}^{C\times H\times W}$",
        ha="center",
        va="center",
        fontsize=6.0,
        color=MUTED,
    )

    # ── Step 2: channel ranking, bottom-10% = band B ─────────────────────────
    a2 = inset(1, 0.16, 0.16, 0.70, 0.62)
    vals = np.sort(rng.uniform(0.05, 1.0, 22))
    cols = [C_OURS if i < 3 else "#B7C2D0" for i in range(len(vals))]
    a2.barh(range(len(vals)), vals, color=cols, height=0.8)
    a2.set_xticks([])
    a2.set_yticks([])
    for sp in a2.spines.values():
        sp.set_visible(False)
    fig.text(
        xs[1] + cw * 0.5,
        y0 + 0.085 * ch,
        "quietest 10% = B",
        ha="center",
        va="center",
        fontsize=6.0,
        color=C_OURS,
        fontweight="bold",
    )

    # ── Step 3: shape — clean (smooth) vs ADV (jagged) ───────────────────────
    p1 = inset(2, 0.10, 0.40, 0.36, 0.40)
    p1.imshow(_smooth(rng), cmap="magma")
    p1.set_xticks([])
    p1.set_yticks([])
    p2 = inset(2, 0.54, 0.40, 0.36, 0.40)
    p2.imshow(_jagged(rng), cmap="magma")
    p2.set_xticks([])
    p2.set_yticks([])
    fig.text(
        xs[2] + cw * 0.28,
        y0 + 0.30 * ch,
        "clean\nlow TV",
        ha="center",
        va="center",
        fontsize=5.6,
        color=C_OURS,
        linespacing=1.1,
    )
    fig.text(
        xs[2] + cw * 0.72,
        y0 + 0.30 * ch,
        "ADV\nhigh TV",
        ha="center",
        va="center",
        fontsize=5.6,
        color=C_BASE,
        linespacing=1.1,
    )
    fig.text(
        xs[2] + cw / 2,
        y0 + 0.085 * ch,
        "TV = jaggedness",
        ha="center",
        va="center",
        fontsize=6.0,
        color=MUTED,
    )

    # ── Step 4: V(x) formula + tiny bar with threshold ───────────────────────
    fig.text(
        xs[3] + cw / 2,
        y0 + 0.62 * ch,
        r"$V=\frac{1}{|\mathcal{B}|}\sum_{c\in\mathcal{B}}\frac{\mathrm{TV}(a_c)}{\overline{|a_c|}}$",
        ha="center",
        va="center",
        fontsize=7.2,
        color=INK,
    )
    a4 = inset(3, 0.20, 0.16, 0.62, 0.30)
    lab = ["ID", "OOD", "ADV"]
    vv = [0.22, 0.30, 0.86]
    bcol = [C_OURS, C_COMP, C_BASE]
    a4.bar(range(3), vv, color=bcol, width=0.66)
    a4.axhline(0.55, color="#444", ls="--", lw=0.9)
    a4.set_xticks(range(3))
    a4.set_xticklabels(lab, fontsize=5.6)
    a4.set_yticks([])
    a4.set_ylim(0, 1.0)
    for sp in a4.spines.values():
        sp.set_visible(False)
    a4.text(2.5, 0.58, r"$\tau$", fontsize=7, color="#444", va="bottom", ha="right")

    # ── Step 5: route to three outcomes ──────────────────────────────────────
    chips = [
        ("ID  ✓  pass", C_OURS, "#E9F8F1"),
        ("OOD  →  abstain", C_COMP, "#E6F0FB"),
        ("ADV  ✗  reject", C_BASE, "#FCE7E4"),
    ]
    for j, (txt, ec, fc) in enumerate(chips):
        cy = y0 + ch * (0.66 - 0.22 * j)
        fig.add_artist(
            FancyBboxPatch(
                (xs[4] + 0.10 * cw, cy - 0.05 * ch),
                0.80 * cw,
                0.10 * ch,
                boxstyle="round,pad=0.004",
                facecolor=fc,
                edgecolor=ec,
                linewidth=1.2,
                transform=fig.transFigure,
                zorder=4,
                mutation_aspect=0.4,
            )
        )
        fig.text(
            xs[4] + cw / 2,
            cy,
            txt,
            ha="center",
            va="center",
            fontsize=6.2,
            color=ec,
            fontweight="bold",
            zorder=5,
        )

    fig.text(
        0.5,
        0.035,
        "0.3 KB state · gradient-free · no extra forward pass · threshold calibrated on ID only",
        ha="center",
        va="center",
        fontsize=6.2,
        color=MUTED,
        style="italic",
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), bbox_inches="tight", dpi=300, facecolor="#FFFFFF")
    plt.close(fig)
    print(f"saved → {out}  ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
