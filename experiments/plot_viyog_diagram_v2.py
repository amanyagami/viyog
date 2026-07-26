r"""Viyog step-by-step diagram — IEEE double-column format (7.0" × 4.5").

Layout: 2-column × 3-row grid.
  Row 0 col 0: Step 1 — input images + first-conv activations
  Row 0 col 1: Step 2 — channel ranking bar chart
  Row 1 col 0: Step 3 — dormant band comparison (ID vs ADV, 3 channels)
  Row 1 col 1: Step 4 — V(x) formula + per-input score bars
  Row 2 spanning both cols: Step 5 — threshold decision flow

A diagonal arrow from cell (1,0) to cell (1,1) illustrates the column-to-column
connection (dormant band → V(x) computation).

Run:
    uv run python experiments/plot_viyog_diagram_v2.py \
        --out /mnt/data1/asing725/viyog/paper_rev/figs/rebuttal/fig_viyog_diagram.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).parent))
from viyog_plotstyle import C_ACCENT, C_BASE, C_COMP, C_OURS, apply_style

apply_style()

# ── colour constants ────────────────────────────────────────────────────────
BG = "#F7F9FC"
STEP_BG = "#FFFFFF"
DIVIDER = "#D0D8E8"
ARROW_C = "#555555"

RdBu = plt.cm.RdBu_r
GnBu = LinearSegmentedColormap.from_list(
    "GnBu",
    ["#FFFFFF", "#A8D8B9", "#009E73"],
    N=256,
)
Oranges = LinearSegmentedColormap.from_list(
    "Oranges",
    ["#FFFFFF", "#F4C28A", "#D55E00"],
    N=256,
)


# ── synthetic data helpers ──────────────────────────────────────────────────
def _rand_activation(
    rng: np.random.Generator, c: int, h: int, rough: float, scale: float
) -> np.ndarray:
    """Return (c,h,h) activation tensor with given spatial roughness and mean magnitude."""
    a = rng.standard_normal((c, h, h)).astype(np.float32)
    if rough < 0.5:
        for _ in range(2):
            p = np.pad(a, ((0, 0), (1, 1), (1, 1)), mode="edge")
            a = (
                sum(p[:, r : r + h, s : s + h] for r in range(3) for s in range(3)).astype(
                    np.float32
                )
                / 9.0
            )
    channel_scale = np.linspace(0.05, 1.5, c) * scale
    rng.shuffle(channel_scale)
    return a * channel_scale[:, None, None]


def _tv(a: np.ndarray) -> np.ndarray:
    """Return mean TV per channel, shape (c,)."""
    dh = np.abs(np.diff(a, axis=1)).mean(axis=(1, 2))
    dw = np.abs(np.diff(a, axis=2)).mean(axis=(1, 2))
    return 0.5 * (dh + dw)


def _v(a: np.ndarray, dorm: np.ndarray, eps: float = 1e-6) -> float:
    """Return the Viyog score for activation tensor *a* and dormant indices *dorm*."""
    ma = np.abs(a).mean(axis=(1, 2))
    tv = _tv(a)
    return float((tv / (ma + eps))[dorm].mean())


# ── drawing helpers ─────────────────────────────────────────────────────────
def card(ax: plt.Axes, title: str, step_num: int, color: str = C_OURS) -> None:
    """Draw step card: coloured header strip with step number and title."""
    ax.set_facecolor("#F8FAFC")
    for sp in ax.spines.values():
        sp.set_color(color)
        sp.set_linewidth(1.2)
        sp.set_visible(True)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axhspan(
        0.90,
        1.00,
        xmin=0,
        xmax=1,
        transform=ax.transAxes,
        color=color,
        alpha=0.12,
        zorder=0,
    )
    ax.text(
        0.02,
        0.945,
        f"Step {step_num}",
        transform=ax.transAxes,
        fontsize=7.5,
        fontweight="bold",
        color=color,
        va="center",
        zorder=1,
    )
    ax.text(
        0.55,
        0.945,
        title,
        transform=ax.transAxes,
        fontsize=6.0,
        color="#333",
        va="center",
        ha="center",
        style="italic",
        zorder=1,
    )


def show_map(
    parent: plt.Axes,
    bounds: tuple[float, float, float, float],
    data: np.ndarray,
    cmap: str | object,
    border_color: str,
    label: str,
) -> None:
    """Embed an activation heatmap as an inset axes inside *parent*."""
    ins = parent.inset_axes(bounds, transform=parent.transAxes)
    vmax = float(np.abs(data).max()) + 1e-9
    ins.imshow(data, cmap=cmap, aspect="auto", vmin=-vmax, vmax=vmax, interpolation="nearest")
    ins.set_xticks([])
    ins.set_yticks([])
    for sp in ins.spines.values():
        sp.set_color(border_color)
        sp.set_linewidth(1.6)
    if label:
        parent.text(
            bounds[0] + bounds[2] / 2,
            bounds[1] - 0.05,
            label,
            transform=parent.transAxes,
            fontsize=5.5,
            ha="center",
            va="top",
            color=border_color,
            fontweight="bold",
        )


def diag_arrow(fig: plt.Figure, ax_from: plt.Axes, ax_to: plt.Axes) -> None:
    """Draw a diagonal arrow from the bottom-right of *ax_from* to the top-left of *ax_to*."""
    bb_f = ax_from.get_position()
    bb_t = ax_to.get_position()
    x0 = bb_f.x1
    y0 = bb_f.y0
    x1 = bb_t.x0
    y1 = bb_t.y1
    fig.add_artist(
        mpatches.FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            transform=fig.transFigure,
            arrowstyle="->,head_width=0.014,head_length=0.010",
            connectionstyle="arc3,rad=-0.30",
            color=ARROW_C,
            lw=1.6,
            zorder=10,
        )
    )


def step_arrow(
    fig: plt.Figure, ax_from: plt.Axes, ax_to: plt.Axes, horizontal: bool = False
) -> None:
    """Draw a straight arrow between adjacent step panels."""
    bb_f = ax_from.get_position()
    bb_t = ax_to.get_position()
    if horizontal:
        x0 = bb_f.x1 + 0.003
        y0 = (bb_f.y0 + bb_f.y1) / 2
        x1 = bb_t.x0 - 0.003
        y1 = y0
    else:
        x0 = (bb_f.x0 + bb_f.x1) / 2
        y0 = bb_f.y0 - 0.003
        x1 = (bb_t.x0 + bb_t.x1) / 2
        y1 = bb_t.y1 + 0.003
    fig.add_artist(
        mpatches.FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            transform=fig.transFigure,
            arrowstyle="->,head_width=0.012,head_length=0.008",
            color=ARROW_C,
            lw=1.4,
            zorder=10,
        )
    )


def synth_image(rng: np.random.Generator, kind: str) -> np.ndarray:
    """Return a normalised (28,28) synthetic image for the given input *kind*."""
    if kind == "id":
        im = rng.standard_normal((28, 28)).astype(np.float32)
        p = np.pad(im, 2, mode="edge")
        for _ in range(6):
            im = (
                sum(p[r : r + 28, s : s + 28] for r in range(5) for s in range(5)).astype(
                    np.float32
                )
                / 25
            )
            p = np.pad(im, 2, mode="edge")
    elif kind == "ood":
        x = np.linspace(0, 4 * np.pi, 28)
        im = np.outer(np.sin(x), np.cos(x)).astype(np.float32)
        im += 0.15 * rng.standard_normal((28, 28)).astype(np.float32)
    else:
        im = rng.standard_normal((28, 28)).astype(np.float32)
        p = np.pad(im, 2, mode="edge")
        for _ in range(3):
            im = (
                sum(p[r : r + 28, s : s + 28] for r in range(5) for s in range(5)).astype(
                    np.float32
                )
                / 25
            )
            p = np.pad(im, 2, mode="edge")
        noise = rng.choice([-1, 0, 0, 0, 1], size=(28, 28)).astype(np.float32) * 0.8
        im = im + noise
    mn, mx = im.min(), im.max()
    return (im - mn) / (mx - mn + 1e-9)


# ── main ────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default="/mnt/data1/asing725/viyog/paper_rev/figs/rebuttal/fig_viyog_diagram.pdf",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    C, H = 48, 12

    # ── synthetic activation data ──────────────────────────────────────────
    a_id = _rand_activation(rng, C, H, rough=0.10, scale=1.0)
    a_adv = _rand_activation(rng, C, H, rough=0.95, scale=1.0)
    a_ood = _rand_activation(rng, C, H, rough=0.13, scale=0.9)

    mean_id = np.abs(a_id).mean(axis=(1, 2))
    alive = np.where(mean_id > 1e-4)[0]
    k = max(1, round(0.10 * len(alive)))
    dorm = alive[np.argsort(mean_id[alive])][:k]
    active = alive[np.argsort(mean_id[alive])][k:]

    v_id = _v(a_id, dorm)
    v_ood = _v(a_ood, dorm)
    v_adv = _v(a_adv, dorm)
    v_adv = max(v_adv, v_ood * 3.2)
    v_ood = max(v_ood, v_id * 1.4)  # keep OOD distinct from ID
    tau = (v_ood + v_adv) * 0.42

    # ── figure: 7.0" × 4.5" for IEEE double-column ────────────────────────
    # GridSpec: 2 cols × 3 rows; row 2 uses colspan=2 for Step 5
    fig = plt.figure(figsize=(7.0, 4.5))
    fig.patch.set_facecolor(BG)

    # Outer margins: leave top for title, bottom for caption space
    gs = gridspec.GridSpec(
        3,
        2,
        figure=fig,
        left=0.02,
        right=0.98,
        top=0.91,
        bottom=0.03,
        hspace=0.38,
        wspace=0.08,
    )

    ax1 = fig.add_subplot(gs[0, 0])  # Step 1 — inputs + activations
    ax2 = fig.add_subplot(gs[0, 1])  # Step 2 — channel ranking
    ax3 = fig.add_subplot(gs[1, 0])  # Step 3 — dormant band comparison
    ax4 = fig.add_subplot(gs[1, 1])  # Step 4 — V(x) formula + bars
    ax5 = fig.add_subplot(gs[2, :])  # Step 5 — decision flow (full width)

    # ── STEP 1: Input images → first-conv activations ─────────────────────
    card(ax1, "Input  →  first-conv activations", 1, C_OURS)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    imgs = [synth_image(rng, k) for k in ["id", "ood", "adv"]]
    labels_in = ["ID", "OOD", "ADV"]
    colors_in = [C_OURS, C_COMP, C_BASE]
    cmaps_in = ["Greys_r", "YlGnBu", "OrRd"]

    # input images row — top half of the panel
    for j, (im, lbl, col, cm) in enumerate(zip(imgs, labels_in, colors_in, cmaps_in, strict=False)):
        x0 = 0.03 + j * 0.32
        show_map(ax1, (x0, 0.55, 0.27, 0.30), im - 0.5, cm, col, lbl)

    # Separator arrow between the two image rows
    ax1.annotate(
        "",
        xy=(0.50, 0.49),
        xytext=(0.50, 0.53),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->,head_width=0.12,head_length=0.08", color="#888", lw=1.0),
    )

    # activation maps row — bottom half
    act_arrays = [a_id, a_ood, a_adv]
    act_cmaps = [GnBu, GnBu, RdBu]
    for j, (arr, cm, col) in enumerate(zip(act_arrays, act_cmaps, colors_in, strict=False)):
        x0 = 0.03 + j * 0.32
        ch = dorm[0] if len(dorm) > 0 else 0
        show_map(ax1, (x0, 0.19, 0.27, 0.30), arr[ch], cm, col, "")

    ax1.text(
        0.50,
        0.09,
        r"$a = f_0(x)\in\mathbb{R}^{C\times H\times W}$",
        transform=ax1.transAxes,
        ha="center",
        va="bottom",
        fontsize=6.5,
        color="#333",
        bbox=dict(boxstyle="round,pad=0.20", fc="#EEF4FF", ec="#AABBD0", lw=0.8),
    )

    # ── STEP 2: Channel ranking ────────────────────────────────────────────
    card(ax2, "Rank channels  →  find dormant band", 2, C_OURS)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    N_SHOW = 28
    mags_show = np.abs(a_id).mean(axis=(1, 2))[:N_SHOW]
    order = np.argsort(mags_show)
    sorted_mags = mags_show[order] / (mags_show.max() + 1e-9)
    bar_colors = [C_OURS if i < k else "#C0C8D8" for i in range(N_SHOW)]
    bar_colors_sorted = [bar_colors[o] for o in order]

    # bar chart occupies the left 60% of the panel; sample maps on the right
    ax_bar = ax2.inset_axes([0.22, 0.14, 0.54, 0.70], transform=ax2.transAxes)
    y_pos = np.linspace(0, 0.95, N_SHOW)
    ax_bar.barh(y_pos, sorted_mags, height=0.030, color=bar_colors_sorted, alpha=0.90)
    thresh_x = float(sorted_mags[min(int(k), N_SHOW - 1)])
    ax_bar.axvline(thresh_x, color=C_OURS, lw=1.2, ls="--", alpha=0.9)
    ax_bar.set_xlim(0, 1.15)
    ax_bar.set_ylim(-0.02, 1.02)
    ax_bar.set_xlabel("mean |act.|", fontsize=5.5)
    ax_bar.set_yticks([])
    ax_bar.set_xticks([0, 0.5, 1.0])
    ax_bar.tick_params(axis="x", labelsize=5.0)
    for sp in ["top", "right"]:
        ax_bar.spines[sp].set_visible(False)

    # side labels — use axes fraction coords so they never overlap bars
    ax2.text(
        0.02,
        0.33,
        "dormant\n$\\mathcal{B}$\n(10%)",
        transform=ax2.transAxes,
        ha="left",
        va="center",
        fontsize=5.5,
        color=C_OURS,
        fontweight="bold",
    )
    ax2.text(
        0.02,
        0.68,
        "active\n$\\mathcal{A}$",
        transform=ax2.transAxes,
        ha="left",
        va="center",
        fontsize=5.5,
        color="#666",
    )

    # dormant vs active sample maps — right side
    for col_idx, (chan, lbl, col, cm) in enumerate(
        [
            (dorm[0] if len(dorm) > 0 else 0, "dorm.", C_OURS, GnBu),
            (active[-1] if len(active) > 0 else 1, "active", "#888", "Greys_r"),
        ]
    ):
        x0 = 0.78 + col_idx * 0.11
        show_map(ax2, (x0, 0.26, 0.10, 0.46), a_id[chan], cm, col, lbl)

    ax2.text(
        0.50,
        0.06,
        "calibrate once on ID data · no test labels",
        transform=ax2.transAxes,
        ha="center",
        va="bottom",
        fontsize=5.2,
        color="#666",
        style="italic",
    )

    # ── STEP 3: Dormant band comparison (ID vs ADV, 3 channels) ───────────
    card(ax3, "Dormant band: ID vs ADV activations", 3, C_BASE)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)

    # Two rows: ID (clean) and ADV (attacked); 3 channels each
    # Two rows: row labels on far left; images fill the rest; sublabels below each row
    # Layout: row 1 top=0.90, height=0.34; row 2 top=0.52, height=0.34
    # Row label at left margin; sublabel below each row (in clear space)
    rows3 = [
        (a_id, "ID", C_OURS, GnBu, ""),
        (a_adv, "ADV", C_BASE, RdBu, "jagged — high TV  ← adversarial signature"),
    ]
    N_CHAN3 = 3
    ROW_H3 = 0.32  # image height in axes fraction
    ROW_TOPS3 = [0.90, 0.50]  # y-top of each image row

    for row_i, (arr, row_label, col, cm, sublabel) in enumerate(rows3):
        y_top = ROW_TOPS3[row_i]
        # Row label — to the LEFT of the images, vertically centred on the row
        ax3.text(
            0.01,
            y_top - ROW_H3 / 2,
            row_label,
            transform=ax3.transAxes,
            fontsize=7.0,
            fontweight="bold",
            color=col,
            va="center",
            ha="left",
        )
        # Images (start at x=0.12 to leave room for the row label)
        chan_w = (0.88 - 0.12) / N_CHAN3
        for ch_i in range(N_CHAN3):
            ci = dorm[ch_i % len(dorm)] if len(dorm) > 0 else ch_i
            x0 = 0.12 + ch_i * chan_w
            ins = ax3.inset_axes(
                (x0, y_top - ROW_H3, chan_w - 0.015, ROW_H3),
                transform=ax3.transAxes,
            )
            data = arr[ci]
            vmax = float(np.abs(data).max()) + 1e-9
            ins.imshow(data, cmap=cm, aspect="auto", vmin=-vmax, vmax=vmax, interpolation="nearest")
            ins.set_xticks([])
            ins.set_yticks([])
            for sp in ins.spines.values():
                sp.set_color(col)
                sp.set_linewidth(2.0 if row_i == 1 else 1.0)
            # TV score only on the first channel, inside top-left corner
            if ch_i == 0:
                tv_val = float(_tv(arr)[ci]) / (float(np.abs(arr).mean(axis=(1, 2))[ci]) + 1e-6)
                ins.text(
                    0.05,
                    0.97,
                    f"TV={tv_val:.2f}",
                    transform=ins.transAxes,
                    fontsize=4.5,
                    va="top",
                    color=col,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.7, lw=0),
                )

        # Sublabel BELOW the image row — enough gap so it never touches images
        ax3.text(
            0.55,
            y_top - ROW_H3 - 0.04,
            sublabel,
            transform=ax3.transAxes,
            fontsize=5.8,
            ha="center",
            va="top",
            color=col,
            fontweight="bold" if row_i == 1 else "normal",
        )

    # ── STEP 4: Compute V(x) ──────────────────────────────────────────────
    card(ax4, "Shape score V(x)", 4, C_ACCENT)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)

    # Formula 1: per-channel normalised TV
    # Use \frac (not \dfrac) to keep height compact; fontsize 9 still looks good
    ax4.text(
        0.50,
        0.93,
        r"$\widetilde{\mathrm{TV}}(a_c)=\frac{\mathrm{TV}(a_c)}{\overline{|a_c|}+\varepsilon}$",
        transform=ax4.transAxes,
        ha="center",
        va="top",
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.28", fc="#FFFBF0", ec=C_ACCENT, lw=1.5),
    )
    ax4.text(
        0.50,
        0.73,
        "roughness ÷ magnitude  →  scale-invariant",
        transform=ax4.transAxes,
        ha="center",
        va="top",
        fontsize=5.5,
        color="#555",
        style="italic",
    )

    # Formula 2: aggregate V(x) — pushed well below formula 1 + its annotation
    ax4.text(
        0.50,
        0.60,
        r"$V(x)=\frac{1}{|\mathcal{B}|}\sum_{c\in\mathcal{B}}\widetilde{\mathrm{TV}}(a_c)$",
        transform=ax4.transAxes,
        ha="center",
        va="top",
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.28", fc="#F0F8F0", ec=C_OURS, lw=1.5),
    )
    ax4.text(
        0.50,
        0.41,
        "average over dormant band  $\\mathcal{B}$",
        transform=ax4.transAxes,
        ha="center",
        va="top",
        fontsize=5.5,
        color="#555",
        style="italic",
    )

    # per-input V bar chart — bottom 36%, with enough space above for labels
    v_vals = [v_id, v_ood, v_adv]
    v_cols = [C_OURS, C_COMP, C_BASE]
    v_lbls = ["ID", "OOD", "ADV"]
    # inset: leave top margin so score labels don't touch formula box
    ax_vbar = ax4.inset_axes([0.08, 0.03, 0.84, 0.34], transform=ax4.transAxes)
    bar_pos = [0.22, 0.50, 0.78]
    bar_w = 0.16
    vmax_show = v_adv * 1.25  # extra headroom so score labels fit inside
    for x, v, col, lbl in zip(bar_pos, v_vals, v_cols, v_lbls, strict=False):
        ax_vbar.bar(x, v / vmax_show, width=bar_w, color=col, alpha=0.88)
        # score above bar — use a white-backed bbox so it never bleeds onto tau line
        ax_vbar.text(
            x,
            v / vmax_show + 0.04,
            f"{v:.2f}",
            ha="center",
            fontsize=5.8,
            color=col,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.08", fc="white", alpha=0.85, lw=0),
        )
        # label below x-axis
        ax_vbar.text(x, -0.16, lbl, ha="center", fontsize=6.0, color=col, fontweight="bold")

    tau_frac = tau / vmax_show
    ax_vbar.axhline(tau_frac, color="#444", lw=1.5, ls="--")
    ax_vbar.text(
        0.98,
        tau_frac + 0.08,
        "τ",
        ha="right",
        fontsize=7.5,
        color="#444",
        fontweight="bold",
        transform=ax_vbar.transAxes,
        bbox=dict(boxstyle="round,pad=0.10", fc="white", alpha=0.9, lw=0),
    )
    ax_vbar.set_xlim(0, 1)
    ax_vbar.set_ylim(-0.22, 1.30)
    ax_vbar.set_xticks([])
    ax_vbar.set_yticks([])
    for sp in ["top", "right", "bottom", "left"]:
        ax_vbar.spines[sp].set_visible(False)

    # ── STEP 5: Threshold & route (full-width) ────────────────────────────
    card(ax5, "Compare V(x) to threshold τ  →  route to ADV or OOD/ID", 5, C_BASE)
    ax5.set_xlim(0, 1)
    ax5.set_ylim(0, 1)

    # Decision boxes
    for x_box, label, col, action in [
        (0.18, r"$V(x)\leq\tau$", C_OURS, "OOD / ID\n→ abstain / monitor"),
        (0.72, r"$V(x)>\tau$", C_BASE, "Adversarial\n→ reject / alert"),
    ]:
        ax5.add_patch(
            mpatches.FancyBboxPatch(
                (x_box - 0.15, 0.16),
                0.30,
                0.66,
                boxstyle="round,pad=0.02",
                facecolor=col + "22",
                edgecolor=col,
                linewidth=1.8,
                transform=ax5.transAxes,
            )
        )
        ax5.text(
            x_box,
            0.73,
            label,
            transform=ax5.transAxes,
            fontsize=8.5,
            ha="center",
            va="center",
            color=col,
        )
        ax5.text(
            x_box,
            0.42,
            action,
            transform=ax5.transAxes,
            fontsize=7.0,
            ha="center",
            va="center",
            color=col,
            fontweight="bold",
            linespacing=1.4,
        )

    # V(x) label + arrows fanning to both boxes
    ax5.text(
        0.50,
        0.93,
        "V(x)",
        transform=ax5.transAxes,
        fontsize=8.5,
        ha="center",
        va="top",
        color="#333",
        fontweight="bold",
    )
    for tgt_x, col in [(0.32, C_OURS), (0.68, C_BASE)]:
        ax5.annotate(
            "",
            xy=(tgt_x, 0.84),
            xytext=(0.50, 0.91),
            xycoords="axes fraction",
            textcoords="axes fraction",
            arrowprops=dict(arrowstyle="->,head_width=0.16,head_length=0.10", color=col, lw=1.8),
        )

    # key property badges
    badges = [
        ("No extra forward pass", "#EEF4FF"),
        ("0.3 KB state", "#EEF4FF"),
        ("No ADV labels needed", "#FFEEF0"),
    ]
    for i, (txt, fc) in enumerate(badges):
        ax5.text(
            0.38 + i * 0.22,
            0.08,
            txt,
            transform=ax5.transAxes,
            fontsize=5.5,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.20", fc=fc, ec="#AABBCC", lw=0.7),
        )

    ax5.text(
        0.50,
        0.22,
        "τ calibrated at 5% ID false-positive rate · no adversarial labels",
        transform=ax5.transAxes,
        ha="center",
        va="center",
        fontsize=5.5,
        color="#555",
        style="italic",
    )

    # ── figure title ──────────────────────────────────────────────────────
    fig.text(
        0.50,
        0.975,
        "How Viyog detects adversarial inputs — one forward pass, no extra parameters",
        ha="center",
        va="top",
        fontsize=8.5,
        fontweight="bold",
        color="#1A1A2E",
    )

    # ── inter-step arrows (drawn after layout is fixed) ───────────────────
    step_arrow(fig, ax1, ax3)  # Step 1 → Step 3  (down col 0)
    step_arrow(fig, ax2, ax4)  # Step 2 → Step 4  (down col 1)
    step_arrow(fig, ax3, ax5)  # Step 3 → Step 5  (into bottom row, col 0 side)
    step_arrow(fig, ax4, ax5)  # Step 4 → Step 5  (into bottom row, col 1 side)
    # Diagonal arrow: dormant band (3,0) → V(x) computation (3,1)
    diag_arrow(fig, ax3, ax4)

    # ── save ─────────────────────────────────────────────────────────────
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    size_kb = out.stat().st_size // 1024
    print(f"saved → {out}  ({size_kb} KB)")


if __name__ == "__main__":
    main()
