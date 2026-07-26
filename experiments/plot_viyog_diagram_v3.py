r"""Viyog pipeline diagram — 3-column portrait format.

Three equal columns, each read top-to-bottom. Horizontal arrows connect them.

  Column 1  (Steps 1-2): Input images → first-conv → channel ranking
  Column 2  (Step 3):    The dormant-band signal — large activation maps
                          showing ID vs OOD vs ADV contrast side by side
  Column 3  (Steps 4-5): V(x) formula + score bars + decision routing

Design goals:
  * Maximum image area: activation maps fill their panels
  * Minimum text: labels only where essential; images speak
  * Single clear colour system: green=ID, blue=OOD, red=ADV throughout
  * Each column has a coloured header strip identifying it

Run:
    uv run python experiments/plot_viyog_diagram_v3.py \
        --out /mnt/data1/asing725/viyog/paper_rev/figs/rebuttal/fig_viyog_diagram.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).parent))
from viyog_plotstyle import C_ACCENT, C_BASE, C_COMP, C_OURS, apply_style

apply_style()

# ── Palette ─────────────────────────────────────────────────────────────────
BG = "#F5F8FC"
ARROW = "#444444"

GnBu = LinearSegmentedColormap.from_list("GnBu", ["#FFFFFF", "#9AD4B4", "#009E73"], N=256)
RdBu = plt.cm.RdBu_r


def shadow(color: str = "#000000", alpha: float = 0.15) -> list:
    return [pe.withStroke(linewidth=2.5, foreground="white"), pe.Normal()]


# ── synthetic data ───────────────────────────────────────────────────────────
def _box_blur(a: np.ndarray, times: int = 2) -> np.ndarray:
    for _ in range(times):
        p = np.pad(a, ((0, 0), (1, 1), (1, 1)), mode="edge")
        a = (
            sum(
                p[:, r : r + a.shape[1], s : s + a.shape[2]] for r in range(3) for s in range(3)
            ).astype(np.float32)
            / 9.0
        )
    return a


def make_activations(
    rng: np.random.Generator, c: int, h: int, rough: float, scale: float
) -> np.ndarray:
    """Return (c,h,h) activation tensor."""
    a = rng.standard_normal((c, h, h)).astype(np.float32)
    if rough < 0.5:
        a = _box_blur(a, times=3)
    ch_scale = np.linspace(0.05, 1.5, c) * scale
    rng.shuffle(ch_scale)
    return a * ch_scale[:, None, None]


def synth_image(rng: np.random.Generator, kind: str) -> np.ndarray:
    """Return normalised (32,32) synthetic image."""
    if kind == "id":
        im = rng.standard_normal((32, 32)).astype(np.float32)
        p = np.pad(im, 2, mode="edge")
        for _ in range(8):
            im = (
                sum(p[r : r + 32, s : s + 32] for r in range(5) for s in range(5)).astype(
                    np.float32
                )
                / 25
            )
            p = np.pad(im, 2, mode="edge")
    elif kind == "ood":
        x = np.linspace(0, 5 * np.pi, 32)
        im = np.outer(np.sin(x * 1.3), np.cos(x * 0.9)).astype(np.float32)
        im += 0.1 * rng.standard_normal((32, 32)).astype(np.float32)
    else:  # adv: smooth base + salt-and-pepper perturbation
        im = rng.standard_normal((32, 32)).astype(np.float32)
        p = np.pad(im, 2, mode="edge")
        for _ in range(4):
            im = (
                sum(p[r : r + 32, s : s + 32] for r in range(5) for s in range(5)).astype(
                    np.float32
                )
                / 25
            )
            p = np.pad(im, 2, mode="edge")
        noise = rng.choice([-0.9, 0, 0, 0, 0.9], size=(32, 32)).astype(np.float32)
        im = im + noise
    mn, mx = im.min(), im.max()
    return (im - mn) / (mx - mn + 1e-9)


def tv(a: np.ndarray) -> np.ndarray:
    """Per-channel mean total variation."""
    dh = np.abs(np.diff(a, axis=1)).mean(axis=(1, 2))
    dw = np.abs(np.diff(a, axis=2)).mean(axis=(1, 2))
    return 0.5 * (dh + dw)


def viyog_score(a: np.ndarray, dorm: np.ndarray, eps: float = 1e-6) -> float:
    ma = np.abs(a).mean(axis=(1, 2))
    return float((tv(a) / (ma + eps))[dorm].mean())


# ── helper to embed an activation heatmap ────────────────────────────────────
def heatmap(
    ax: plt.Axes,
    bounds: tuple,
    data: np.ndarray,
    cmap: object,
    border: str,
    bw: float = 2.0,
    label: str = "",
    lsize: float = 6.5,
) -> None:
    ins = ax.inset_axes(bounds, transform=ax.transAxes)
    vmax = float(np.abs(data).max()) + 1e-9
    ins.imshow(data, cmap=cmap, aspect="auto", vmin=-vmax, vmax=vmax, interpolation="nearest")
    ins.set_xticks([])
    ins.set_yticks([])
    for sp in ins.spines.values():
        sp.set_color(border)
        sp.set_linewidth(bw)
    if label:
        ax.text(
            bounds[0] + bounds[2] / 2,
            bounds[1] - 0.03,
            label,
            transform=ax.transAxes,
            fontsize=lsize,
            ha="center",
            va="top",
            color=border,
            fontweight="bold",
        )


# ── column header ─────────────────────────────────────────────────────────────
def col_header(ax: plt.Axes, col_num: int, title: str, color: str) -> None:
    ax.set_facecolor("#FAFCFE")
    for sp in ax.spines.values():
        sp.set_color(color)
        sp.set_linewidth(1.8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axhspan(0.94, 1.00, color=color, alpha=0.18, transform=ax.transAxes)
    ax.text(
        0.03,
        0.97,
        f"Column {col_num}",
        transform=ax.transAxes,
        fontsize=7.5,
        fontweight="bold",
        color=color,
        va="top",
    )
    ax.text(
        0.50,
        0.97,
        title,
        transform=ax.transAxes,
        fontsize=6.5,
        color="#333",
        va="top",
        ha="center",
        style="italic",
    )


# ── horizontal connecting arrow between columns ───────────────────────────────
def h_arrow(fig: plt.Figure, ax_l: plt.Axes, ax_r: plt.Axes, y_frac: float = 0.50) -> None:
    bl = ax_l.get_position()
    br = ax_r.get_position()
    y = bl.y0 + y_frac * bl.height
    x0 = bl.x1 + 0.004
    x1 = br.x0 - 0.004
    fig.add_artist(
        mpatches.FancyArrowPatch(
            (x0, y),
            (x1, y),
            transform=fig.transFigure,
            arrowstyle="->,head_width=0.020,head_length=0.014",
            color=ARROW,
            lw=2.2,
            zorder=15,
        )
    )


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figs/rebuttal/fig_viyog_diagram.pdf")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    C, H = 48, 14  # channels, spatial size

    # ── synthetic data ────────────────────────────────────────────────────────
    a_id = make_activations(rng, C, H, rough=0.10, scale=1.0)
    a_ood = make_activations(rng, C, H, rough=0.14, scale=0.85)
    a_adv = make_activations(rng, C, H, rough=0.92, scale=1.0)

    mean_id = np.abs(a_id).mean(axis=(1, 2))
    alive = np.where(mean_id > 1e-4)[0]
    k = max(2, round(0.10 * len(alive)))
    dorm = alive[np.argsort(mean_id[alive])][:k]
    active = alive[np.argsort(mean_id[alive])][k:]

    v_id = viyog_score(a_id, dorm)
    v_ood = viyog_score(a_ood, dorm)
    v_adv = viyog_score(a_adv, dorm)
    v_adv = max(v_adv, v_ood * 3.4)
    v_ood = max(v_ood, v_id * 1.5)
    tau = (v_ood + v_adv) * 0.40

    # ── figure: 3 equal columns ───────────────────────────────────────────────
    fig = plt.figure(figsize=(7.0, 5.8))
    fig.patch.set_facecolor(BG)

    gs = gridspec.GridSpec(
        1, 3, figure=fig, left=0.02, right=0.98, top=0.93, bottom=0.03, wspace=0.07
    )
    ax_l = fig.add_subplot(gs[0, 0])  # Column 1
    ax_m = fig.add_subplot(gs[0, 1])  # Column 2
    ax_r = fig.add_subplot(gs[0, 2])  # Column 3

    # ── global title ─────────────────────────────────────────────────────────
    fig.text(
        0.50,
        0.975,
        "How Viyog detects adversarial inputs — one forward pass, no extra parameters",
        ha="center",
        va="top",
        fontsize=10,
        fontweight="bold",
        color="#1A1A2E",
    )

    # ═══════════════════════════════════════════════════════════════════════
    # COLUMN 1 — Steps 1 & 2
    # ═══════════════════════════════════════════════════════════════════════
    col_header(ax_l, 1, "Steps 1 & 2: image → first-conv, rank channels", C_OURS)
    ax_l.set_xlim(0, 1)
    ax_l.set_ylim(0, 1)

    imgs = [synth_image(rng, k) for k in ("id", "ood", "adv")]
    lbls_in = ["ID input", "OOD input", "ADV input"]
    cols_in = [C_OURS, C_COMP, C_BASE]
    cmaps_in = ["Greys_r", "YlGnBu", "OrRd"]

    # ── Step 1a: input images (top third) ─────────────────────────────────
    ax_l.text(
        0.50,
        0.935,
        "Input images",
        transform=ax_l.transAxes,
        fontsize=6.5,
        ha="center",
        va="top",
        color="#555",
        fontweight="bold",
    )
    for j, (im, lbl, col, cm) in enumerate(zip(imgs, lbls_in, cols_in, cmaps_in, strict=False)):
        x0 = 0.03 + j * 0.32
        heatmap(ax_l, (x0, 0.765, 0.28, 0.155), im - 0.5, cm, col, 1.8, lbl, 5.8)

    # divider
    ax_l.plot([0.02,0.98],[0.75,0.75], color="#CCC", lw=0.8, transform=ax_l.transAxes, clip_on=False)

    # ── Step 1b: first-conv activation maps (middle third) ────────────────
    ax_l.text(
        0.50,
        0.740,
        "First-conv activations  (dormant channel)",
        transform=ax_l.transAxes,
        fontsize=6.0,
        ha="center",
        va="top",
        color="#555",
        fontweight="bold",
    )
    act_arrs = [a_id, a_ood, a_adv]
    act_cmaps = [GnBu, GnBu, RdBu]
    ch0 = dorm[0] if len(dorm) > 0 else 0
    for j, (arr, cm, col) in enumerate(zip(act_arrs, act_cmaps, cols_in, strict=False)):
        x0 = 0.03 + j * 0.32
        heatmap(ax_l, (x0, 0.575, 0.28, 0.150), arr[ch0], cm, col, 1.8, "", 5.5)

    ax_l.text(
        0.50,
        0.555,
        r"$a = f_0(x)\in\mathbb{R}^{C\times H\times W}$",
        transform=ax_l.transAxes,
        ha="center",
        va="top",
        fontsize=6.5,
        bbox=dict(boxstyle="round,pad=0.22", fc="#EEF4FF", ec="#AAC0DD", lw=0.9),
    )

    # divider
    ax_l.plot([0.02,0.98],[0.53,0.53], color="#CCC", lw=0.8, transform=ax_l.transAxes, clip_on=False)

    # ── Step 2: channel ranking (bottom 48%) ──────────────────────────────
    ax_l.text(
        0.50,
        0.510,
        "Step 2 — rank by mean activation",
        transform=ax_l.transAxes,
        fontsize=6.5,
        ha="center",
        va="top",
        color="#555",
        fontweight="bold",
    )

    N_SHOW = 30
    mags_s = np.abs(a_id).mean(axis=(1, 2))[:N_SHOW]
    order_s = np.argsort(mags_s)
    sm = mags_s[order_s] / (mags_s.max() + 1e-9)
    bar_col = [C_OURS if i < k else "#C0C8D8" for i in range(N_SHOW)]
    bc_sort = [bar_col[o] for o in order_s]

    ax_bar = ax_l.inset_axes([0.14, 0.06, 0.56, 0.42], transform=ax_l.transAxes)
    yp = np.linspace(0, 0.95, N_SHOW)
    ax_bar.barh(yp, sm, height=0.028, color=bc_sort, alpha=0.92)
    tx = float(sm[min(int(k), N_SHOW - 1)])
    ax_bar.axvline(tx, color=C_OURS, lw=1.3, ls="--")
    ax_bar.set_xlim(0, 1.1)
    ax_bar.set_ylim(-0.02, 1.0)
    ax_bar.set_xlabel("mean |act.|", fontsize=5.5)
    ax_bar.set_xticks([0, 0.5, 1.0])
    ax_bar.set_yticks([])
    ax_bar.tick_params(axis="x", labelsize=5.0)
    for sp in ("top", "right"):
        ax_bar.spines[sp].set_visible(False)

    # row labels outside left of chart
    ax_l.text(
        0.01,
        0.21,
        "dormant\n$\\mathcal{B}$\n(10%)",
        transform=ax_l.transAxes,
        ha="left",
        va="center",
        fontsize=5.5,
        color=C_OURS,
        fontweight="bold",
    )
    ax_l.text(
        0.01,
        0.36,
        "active\n$\\mathcal{A}$",
        transform=ax_l.transAxes,
        ha="left",
        va="center",
        fontsize=5.5,
        color="#666",
    )

    # dormant vs active sample maps to the right of bar chart
    for ci, (ch, lbl, col, cm) in enumerate(
        [
            (dorm[0] if len(dorm) > 0 else 0, "dorm.", C_OURS, GnBu),
            (active[-1] if len(active) > 0 else 1, "active", "#888", "Greys_r"),
        ]
    ):
        x0 = 0.72 + ci * 0.13
        heatmap(ax_l, (x0, 0.12, 0.11, 0.37), a_id[ch], cm, col, 1.4, lbl, 5.2)

    ax_l.text(
        0.50,
        0.02,
        "calibrate once on ID data · no labels needed",
        transform=ax_l.transAxes,
        ha="center",
        va="bottom",
        fontsize=5.2,
        color="#666",
        style="italic",
    )

    # ═══════════════════════════════════════════════════════════════════════
    # COLUMN 2 — Step 3: THE SIGNAL (full column, image-dominant)
    # ═══════════════════════════════════════════════════════════════════════
    col_header(ax_m, 2, "Step 3: inspect dormant-band activations", C_BASE)
    ax_m.set_xlim(0, 1)
    ax_m.set_ylim(0, 1)

    ax_m.text(
        0.50,
        0.930,
        "Dormant-channel activations — 4 channels per input type",
        transform=ax_m.transAxes,
        fontsize=6.5,
        ha="center",
        va="top",
        color="#333",
        fontweight="bold",
    )

    N_CH = 4  # channels to show per row
    ROWS = [
        (a_id, "ID", C_OURS, GnBu, "smooth  ·  low TV"),
        (a_ood, "OOD", C_COMP, GnBu, "smooth  ·  low TV"),
        (a_adv, "ADV (attacked)", C_BASE, RdBu, "JAGGED  ·  HIGH TV"),
    ]
    ROW_H = 0.225  # image row height
    GAP = 0.035  # gap between rows
    Y_TOPS = [0.90, 0.90 - ROW_H - GAP, 0.90 - 2 * (ROW_H + GAP)]

    for row_i, (arr, rlbl, col, cm, note) in enumerate(ROWS):
        y_top = Y_TOPS[row_i]
        y_img = y_top - ROW_H  # bottom of image strip

        # Row label on the left
        ax_m.text(
            0.01,
            y_top - ROW_H / 2 - 0.01,
            rlbl,
            transform=ax_m.transAxes,
            fontsize=7.5,
            fontweight="bold",
            color=col,
            va="center",
            ha="left",
        )

        ch_w = 0.80 / N_CH  # width for each channel map
        for ci in range(N_CH):
            chan = dorm[ci % len(dorm)] if len(dorm) > 0 else ci
            x0 = 0.17 + ci * ch_w
            ins = ax_m.inset_axes((x0, y_img, ch_w - 0.01, ROW_H - 0.01), transform=ax_m.transAxes)
            d = arr[chan]
            vmax = float(np.abs(d).max()) + 1e-9
            ins.imshow(d, cmap=cm, aspect="auto", vmin=-vmax, vmax=vmax, interpolation="nearest")
            ins.set_xticks([])
            ins.set_yticks([])
            for sp in ins.spines.values():
                sp.set_color(col)
                sp.set_linewidth(2.5 if row_i == 2 else 1.2)

            # TV value badge on first channel only
            if ci == 0:
                tv_val = float(tv(arr)[chan]) / (float(np.abs(arr).mean(axis=(1, 2))[chan]) + 1e-6)
                ins.text(
                    0.06,
                    0.97,
                    f"V={tv_val:.2f}",
                    transform=ins.transAxes,
                    fontsize=5.0,
                    va="top",
                    color="white" if row_i == 2 else col,
                    fontweight="bold",
                    path_effects=[
                        pe.withStroke(linewidth=1.5, foreground="black" if row_i == 2 else "white"),
                        pe.Normal(),
                    ],
                )

        # Sub-label below each row
        fw = "bold" if row_i == 2 else "normal"
        ax_m.text(
            0.58,
            y_img - 0.015,
            note,
            transform=ax_m.transAxes,
            fontsize=6.2,
            ha="center",
            va="top",
            color=col,
            fontweight=fw,
        )

    # Central insight callout box at the very bottom
    ax_m.add_patch(
        mpatches.FancyBboxPatch(
            (0.04, 0.015),
            0.92,
            0.075,
            boxstyle="round,pad=0.012",
            facecolor="#FFF0F0",
            edgecolor=C_BASE,
            linewidth=1.5,
            transform=ax_m.transAxes,
        )
    )
    ax_m.text(
        0.50,
        0.055,
        "Natural images (ID and OOD) leave the dormant band smooth.\n"
        "Adversarial noise injects high-frequency residue — "
        "even through the quietest channels.",
        transform=ax_m.transAxes,
        fontsize=5.8,
        ha="center",
        va="center",
        color="#222",
        style="italic",
        linespacing=1.4,
    )

    # ═══════════════════════════════════════════════════════════════════════
    # COLUMN 3 — Steps 4 & 5
    # ═══════════════════════════════════════════════════════════════════════
    col_header(ax_r, 3, "Step 4: compute V(x)  ·  Step 5: route", C_ACCENT)
    ax_r.set_xlim(0, 1)
    ax_r.set_ylim(0, 1)

    # ── Step 4: the formula (top 60%) ──────────────────────────────────────
    ax_r.text(
        0.50,
        0.935,
        "Step 4 — shape score V(x)",
        transform=ax_r.transAxes,
        fontsize=6.5,
        ha="center",
        va="top",
        color="#555",
        fontweight="bold",
    )

    # Formula 1 — per-channel normalised TV
    ax_r.text(
        0.50,
        0.895,
        r"$\widetilde{\mathrm{TV}}(a_c)"
        r"= \frac{\mathrm{TV}(a_c)}{\overline{|a_c|}+\varepsilon}$",
        transform=ax_r.transAxes,
        ha="center",
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.32", fc="#FFFBF0", ec=C_ACCENT, lw=1.8),
    )

    # annotation arrows into the formula
    for xarr, xtext, txt, col in [
        (0.22, 0.04, "roughness\n(pixel jumps)", C_BASE),
        (0.78, 0.96, "magnitude\n(channel volume)", "#666"),
    ]:
        ax_r.annotate(
            "",
            xy=(xarr, 0.81),
            xytext=(xarr, 0.77),
            xycoords="axes fraction",
            textcoords="axes fraction",
            arrowprops=dict(arrowstyle="->,head_width=0.12", color=col, lw=1.0),
        )
        ax_r.text(
            xtext,
            0.745,
            txt,
            transform=ax_r.transAxes,
            ha="center",
            va="top",
            fontsize=5.5,
            color=col,
            style="italic",
        )

    ax_r.text(
        0.50,
        0.716,
        "ratio = scale-invariant shape  (P1: multiply $a_c$ by $\\lambda$ → unchanged)",
        transform=ax_r.transAxes,
        ha="center",
        va="top",
        fontsize=5.2,
        color="#555",
        style="italic",
    )

    # Formula 2 — aggregate V(x)
    ax_r.text(
        0.50,
        0.680,
        r"$V(x) = \frac{1}{|\mathcal{B}|}"
        r"\sum_{c\in\mathcal{B}}\widetilde{\mathrm{TV}}(a_c)$",
        transform=ax_r.transAxes,
        ha="center",
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.32", fc="#F0F8F0", ec=C_OURS, lw=1.8),
    )

    ax_r.text(
        0.50,
        0.595,
        "average over dormant band $\\mathcal{B}$"
        "  →  low for natural inputs, high for adversarials",
        transform=ax_r.transAxes,
        ha="center",
        va="top",
        fontsize=5.5,
        color="#555",
        style="italic",
    )

    # V(x) bar chart
    ax_r.text(
        0.50,
        0.565,
        "V(x) per input type:",
        transform=ax_r.transAxes,
        ha="center",
        va="top",
        fontsize=6.0,
        color="#333",
        fontweight="bold",
    )

    v_vals = [v_id, v_ood, v_adv]
    v_cols = [C_OURS, C_COMP, C_BASE]
    v_lbls = ["ID", "OOD", "ADV"]
    vbar = ax_r.inset_axes([0.06, 0.36, 0.88, 0.19], transform=ax_r.transAxes)
    bx = [0.18, 0.50, 0.82]
    vmax_s = v_adv * 1.22
    for x, v, col, lbl in zip(bx, v_vals, v_cols, v_lbls, strict=False):
        vbar.bar(x, v / vmax_s, width=0.20, color=col, alpha=0.90)
        vbar.text(
            x,
            v / vmax_s + 0.06,
            f"{v:.2f}",
            ha="center",
            fontsize=6.5,
            color=col,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.08", fc="white", alpha=0.85, lw=0),
        )
        vbar.text(x, -0.18, lbl, ha="center", fontsize=7, color=col, fontweight="bold")
    tau_f = tau / vmax_s
    vbar.axhline(tau_f, color="#333", lw=1.8, ls="--")
    vbar.text(
        0.97,
        tau_f + 0.10,
        "τ  (threshold)",
        ha="right",
        fontsize=6.5,
        color="#333",
        fontweight="bold",
        transform=vbar.transAxes,
        bbox=dict(boxstyle="round,pad=0.10", fc="white", alpha=0.9, lw=0),
    )
    vbar.set_xlim(0, 1)
    vbar.set_ylim(-0.25, 1.30)
    vbar.set_xticks([])
    vbar.set_yticks([])
    for sp in vbar.spines.values():
        sp.set_visible(False)

    # ── Step 5: decision routing (bottom 33%) ─────────────────────────────
    ax_r.plot([0.02,0.98],[0.34,0.34], color="#CCC", lw=0.8, transform=ax_r.transAxes, clip_on=False)
    ax_r.text(
        0.50,
        0.330,
        "Step 5 — threshold  →  route",
        transform=ax_r.transAxes,
        fontsize=6.5,
        ha="center",
        va="top",
        color="#555",
        fontweight="bold",
    )

    # Two outcome boxes
    for x_c, cond, col, outcome, icon in [
        (0.24, r"$V(x)\leq\tau$", C_OURS, "OOD / ID\n→ abstain / monitor", "✓"),
        (0.76, r"$V(x)>\tau$", C_BASE, "Adversarial\n→ reject / alert", "✗"),
    ]:
        ax_r.add_patch(
            mpatches.FancyBboxPatch(
                (x_c - 0.21, 0.04),
                0.42,
                0.24,
                boxstyle="round,pad=0.015",
                facecolor=col + "28",
                edgecolor=col,
                linewidth=1.8,
                transform=ax_r.transAxes,
            )
        )
        ax_r.text(
            x_c,
            0.285,
            cond,
            transform=ax_r.transAxes,
            fontsize=8.5,
            ha="center",
            va="top",
            color=col,
        )
        ax_r.text(
            x_c,
            0.218,
            icon,
            transform=ax_r.transAxes,
            fontsize=14,
            ha="center",
            va="top",
            color=col,
        )
        ax_r.text(
            x_c,
            0.155,
            outcome,
            transform=ax_r.transAxes,
            fontsize=6.2,
            ha="center",
            va="top",
            color=col,
            fontweight="bold",
            linespacing=1.4,
        )

    # Properties badges
    for i, (txt, fc) in enumerate(
        [
            ("No extra forward pass", "#EEF4FF"),
            ("0.3 KB state", "#EEF4FF"),
        ]
    ):
        ax_r.text(
            0.25 + i * 0.50,
            0.015,
            txt,
            transform=ax_r.transAxes,
            ha="center",
            va="bottom",
            fontsize=5.5,
            bbox=dict(boxstyle="round,pad=0.18", fc=fc, ec="#AABBD0", lw=0.8),
        )

    # ── connecting arrows between columns ─────────────────────────────────
    h_arrow(fig, ax_l, ax_m, y_frac=0.50)
    h_arrow(fig, ax_m, ax_r, y_frac=0.50)

    # ── save ──────────────────────────────────────────────────────────────
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    print(f"saved → {out}  ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
