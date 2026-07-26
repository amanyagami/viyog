"""Step-by-step visual diagram of how Viyog works.

Produces a single landscape figure (fig_viyog_diagram.pdf) that walks through
every conceptual step of V(x) from raw image to detection decision, so that a
reader with no prior knowledge can follow the formula.

Five horizontal lanes, left to right:
  Step 1  Input x → first conv layer  (image → activation grid)
  Step 2  Channel ranking: which channels are "dormant"?
  Step 3  What does the dormant band look like for clean vs adversarial?
  Step 4  Compute TV and normalise (the V formula)
  Step 5  Compare V to threshold τ → ADV or OOD
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

# ── add experiments dir to path so we can import the style module ──────────
sys.path.insert(0, str(Path(__file__).parent))
from viyog_plotstyle import C_ACCENT, C_BASE, C_COMP, C_OURS, C_REF, apply_style

apply_style()


def _grid_activations(
    rng: np.random.Generator, roughness: float, c: int = 64, h: int = 8
) -> np.ndarray:
    """Simulate activation maps (c, h, h) with given spatial roughness."""
    a = rng.standard_normal((c, h, h)).astype(np.float32)
    if roughness < 0.5:  # smooth — simple box blur via cumulative sum trick
        # 3x3 box average without scipy
        p = np.pad(a, ((0, 0), (1, 1), (1, 1)), mode="edge")
        a = (
            p[:, :-2, :-2]
            + p[:, 1:-1, :-2]
            + p[:, 2:, :-2]
            + p[:, :-2, 1:-1]
            + p[:, 1:-1, 1:-1]
            + p[:, 2:, 1:-1]
            + p[:, :-2, 2:]
            + p[:, 1:-1, 2:]
            + p[:, 2:, 2:]
        ) / 9.0
    # set channel magnitudes: most channels large, bottom-10% small (dormant)
    scale = np.sort(np.abs(rng.standard_normal(c)) + 0.2)
    a = a * scale[:, None, None]
    return a


def _tv_per_channel(a: np.ndarray) -> np.ndarray:
    """Mean |Δh| + |Δw| per channel."""
    dh = np.abs(np.diff(a, axis=1)).mean(axis=(1, 2))
    dw = np.abs(np.diff(a, axis=2)).mean(axis=(1, 2))
    return 0.5 * (dh + dw)


def _v_score(a: np.ndarray, dorm_idx: np.ndarray, eps: float = 1e-6) -> float:
    mean_abs = np.abs(a).mean(axis=(1, 2))
    tv = _tv_per_channel(a)
    shape = tv / (mean_abs + eps)
    return float(shape[dorm_idx].mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figs/rebuttal/fig_viyog_diagram.pdf")
    args = ap.parse_args()

    rng = np.random.default_rng(42)
    C, H = 64, 8

    # synthetic activation maps
    a_id = _grid_activations(rng, roughness=0.1, c=C, h=H)
    a_ood = _grid_activations(rng, roughness=0.12, c=C, h=H)
    a_adv = _grid_activations(rng, roughness=0.9, c=C, h=H)

    # channel magnitudes
    mean_abs_id = np.abs(a_id).mean(axis=(1, 2))
    alive = np.where(mean_abs_id > 1e-4)[0]
    k = max(1, int(round(0.10 * len(alive))))
    dorm_idx = alive[np.argsort(mean_abs_id[alive])][:k]  # bottom-10%

    v_id = _v_score(a_id, dorm_idx)
    v_ood = _v_score(a_ood, dorm_idx)
    v_adv = _v_score(a_adv, dorm_idx)
    # push adv clearly above threshold for illustration
    v_adv = max(v_adv, v_ood * 2.6)

    # ─── figure layout ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14.0, 4.6))
    fig.patch.set_facecolor("white")

    # 5 main panels + 4 arrow gaps
    panel_w = [1.15, 0.22, 1.0, 0.22, 1.0, 0.22, 1.2, 0.22, 0.85]
    gs = gridspec.GridSpec(
        1,
        9,
        width_ratios=panel_w,
        left=0.03,
        right=0.99,
        bottom=0.10,
        top=0.88,
        wspace=0.0,
    )

    ax1 = fig.add_subplot(gs[0, 0])  # Step 1 — image → conv
    ax2 = fig.add_subplot(gs[0, 2])  # Step 2 — channel ranking
    ax3 = fig.add_subplot(gs[0, 4])  # Step 3 — dormant band heatmaps
    ax4 = fig.add_subplot(gs[0, 6])  # Step 4 — V formula
    ax5 = fig.add_subplot(gs[0, 8])  # Step 5 — threshold

    arrow_kw = dict(
        arrowstyle="->,head_width=0.25,head_length=0.15",
        color=C_REF,
        lw=1.6,
        transform=fig.transFigure,
        clip_on=False,
    )

    def mid_x(col: int) -> float:
        """Figure-fraction x-centre of a gridspec column."""
        rights = np.cumsum(panel_w) / sum(panel_w)
        lefts = np.r_[0, rights[:-1]]
        span = rights[col] - lefts[col]
        return float(lefts[col] + span * 0.03 + (rights[col] - lefts[col]) * 0.85)

    # arrow centres (between panels)
    for arrow_col, ax_col in [(1, 0), (3, 2), (5, 4), (7, 6)]:
        x_start = mid_x(ax_col) + 0.005
        x_end = mid_x(arrow_col + 1) - 0.005 if arrow_col < 8 else mid_x(arrow_col) + 0.015
        fig.add_artist(
            mpatches.FancyArrowPatch(
                (x_start, 0.50),
                (x_end, 0.50),
                **arrow_kw,
            )
        )

    # ─── Step 1 ─────────────────────────────────────────────────────────────
    ax1.set_axis_off()
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.text(
        0.5, 0.97, "Step 1", ha="center", va="top", fontsize=9, fontweight="bold", color="#333"
    )
    ax1.text(
        0.5,
        0.89,
        "Input image →\nfirst conv layer",
        ha="center",
        va="top",
        fontsize=7.5,
        color="#555",
    )

    # draw three small channel maps
    for i, (col, label) in enumerate([(C_OURS, "ID"), (C_COMP, "OOD"), (C_BASE, "ADV")]):
        y0 = 0.62 - i * 0.20
        img = (
            np.abs(a_id[dorm_idx[0]])
            if label == "ID"
            else (np.abs(a_ood[dorm_idx[0]]) if label == "OOD" else np.abs(a_adv[dorm_idx[0]]))
        )
        ins = ax1.inset_axes([0.05, y0, 0.38, 0.17])
        ins.imshow(img, cmap="Greys_r", aspect="auto", vmin=0, vmax=img.max() + 1e-6)
        ins.set_xticks([])
        ins.set_yticks([])
        for spine in ins.spines.values():
            spine.set_color(col)
            spine.set_linewidth(1.5)
        ax1.text(
            0.55,
            y0 + 0.08,
            f"{label}\ninput",
            ha="left",
            va="center",
            fontsize=6.5,
            color=col,
            fontweight="bold",
        )

    ax1.text(
        0.5,
        0.10,
        r"$a = f_0(x) \in \mathbb{R}^{C \times H \times W}$",
        ha="center",
        va="bottom",
        fontsize=7,
        color="#333",
        bbox=dict(boxstyle="round,pad=0.25", fc="#f5f5f5", ec="#ccc", lw=0.8),
    )

    # ─── Step 2 ─────────────────────────────────────────────────────────────
    ax2.set_axis_off()
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.text(
        0.5, 0.97, "Step 2", ha="center", va="top", fontsize=9, fontweight="bold", color="#333"
    )
    ax2.text(
        0.5, 0.89, "Find\n'dormant'\nchannels", ha="center", va="top", fontsize=7.5, color="#555"
    )

    # bar chart of mean |a_c| for 20 channels
    n_show = 20
    mags = np.sort(mean_abs_id[:n_show])[::-1]
    bar_colors = [C_OURS if i >= n_show - k else "#cccccc" for i in range(n_show)]
    bar_colors = bar_colors[::-1]  # sorted ascending → top channels = active
    y_pos = np.arange(n_show)
    ax2.barh(y_pos / n_show, mags / mags.max(), height=0.04, color=bar_colors[::-1], left=0)
    ax2.axvline(mags[-(k + 1)] / mags.max(), color=C_OURS, lw=1.2, ls="--")
    ax2.text(
        0.05,
        0.18,
        f"Bottom {k * 100 // n_show}%\n= dormant\nband $\\mathcal{{B}}$",
        ha="left",
        va="bottom",
        fontsize=6,
        color=C_OURS,
        fontweight="bold",
    )
    ax2.text(
        0.5, 0.06, "Mean activation per channel", ha="center", va="bottom", fontsize=6, color="#555"
    )

    # ─── Step 3 ─────────────────────────────────────────────────────────────
    ax3.set_axis_off()
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.text(
        0.5, 0.97, "Step 3", ha="center", va="top", fontsize=9, fontweight="bold", color="#333"
    )
    ax3.text(
        0.5,
        0.89,
        "Dormant channel activations\n(clean vs adversarial)",
        ha="center",
        va="top",
        fontsize=7.5,
        color="#555",
    )

    for row, (a_arr, col, lbl, rough) in enumerate(
        [
            (a_id, C_OURS, "Clean (ID/OOD)\nsmooth — low TV", "smooth"),
            (a_adv, C_BASE, "Adversarial\njagged — high TV", "jagged"),
        ]
    ):
        y0 = 0.60 - row * 0.30
        # show 3 dormant channels side by side
        for j, ci in enumerate(dorm_idx[:3]):
            ins = ax3.inset_axes([0.04 + j * 0.32, y0, 0.28, 0.22])
            ch = a_arr[ci]
            ins.imshow(
                ch,
                cmap="RdBu_r" if lbl.startswith("Adv") else "Greys_r",
                aspect="auto",
                vmin=-np.abs(ch).max(),
                vmax=np.abs(ch).max(),
            )
            ins.set_xticks([])
            ins.set_yticks([])
            for sp in ins.spines.values():
                sp.set_color(col)
                sp.set_linewidth(1.5)
        ax3.text(
            0.5, y0 - 0.04, lbl, ha="center", va="top", fontsize=6.5, color=col, fontweight="bold"
        )

    ax3.annotate(
        "",
        xy=(0.85, 0.56),
        xytext=(0.85, 0.34),
        arrowprops=dict(arrowstyle="<->", color="#888", lw=1.2),
    )
    ax3.text(0.87, 0.45, "TV\ngap", ha="left", va="center", fontsize=6, color="#888")

    # ─── Step 4 ─────────────────────────────────────────────────────────────
    ax4.set_axis_off()
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.text(
        0.5, 0.97, "Step 4", ha="center", va="top", fontsize=9, fontweight="bold", color="#333"
    )
    ax4.text(
        0.5,
        0.89,
        "Compute $V(x)$\n(the Viyog score)",
        ha="center",
        va="top",
        fontsize=7.5,
        color="#555",
    )

    # formula breakdown — two annotated boxes: numerator + denominator
    ax4.text(
        0.50,
        0.80,
        r"$\widetilde{\mathrm{TV}}(a_c) = \frac{\mathrm{TV}(a_c)}{\overline{|a_c|} + \varepsilon}$",
        ha="center",
        va="center",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", fc="#f0f8f0", ec=C_OURS, lw=1.4),
    )
    ax4.text(
        0.50,
        0.66,
        r"$\uparrow$ roughness$\quad /\quad$magnitude $\downarrow$",
        ha="center",
        va="center",
        fontsize=7,
        color="#555",
    )

    ax4.text(
        0.50,
        0.44,
        r"$V(x) = \frac{1}{|\mathcal{B}|} \sum_{c \in \mathcal{B}} \widetilde{\mathrm{TV}}(a_c)$",
        ha="center",
        va="center",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", fc="#fff8f0", ec=C_ACCENT, lw=1.4),
    )

    ax4.text(0.15, 0.74, "per\nchannel", ha="center", va="center", fontsize=6, color=C_OURS)
    ax4.text(
        0.15,
        0.44,
        "average\nover $\\mathcal{B}$",
        ha="center",
        va="center",
        fontsize=6,
        color=C_ACCENT,
    )

    ax4.text(
        0.50,
        0.18,
        "High $V$ = rough dormant channels\n→ adversarial residue present",
        ha="center",
        va="center",
        fontsize=6.5,
        color=C_BASE,
        style="italic",
        bbox=dict(boxstyle="round,pad=0.2", fc="#fff0f0", ec=C_BASE, lw=0.8),
    )

    # ─── Step 5 ─────────────────────────────────────────────────────────────
    ax5.set_axis_off()
    ax5.set_xlim(0, 1)
    ax5.set_ylim(0, 1)
    ax5.text(
        0.5, 0.97, "Step 5", ha="center", va="top", fontsize=9, fontweight="bold", color="#333"
    )
    ax5.text(
        0.5, 0.89, "Compare to\nthreshold τ", ha="center", va="top", fontsize=7.5, color="#555"
    )

    # score distribution chart
    scores = {"ID": v_id, "OOD": v_ood, "ADV": v_adv}
    tau = (v_ood + v_adv) * 0.50
    colors_map = {"ID": C_OURS, "OOD": C_COMP, "ADV": C_BASE}
    y0s = {"ID": 0.72, "OOD": 0.56, "ADV": 0.40}
    max_v = v_adv * 1.1
    for lbl, score in scores.items():
        y0 = y0s[lbl]
        col = colors_map[lbl]
        bar_len = score / max_v * 0.80
        ax5.barh([y0], [bar_len], height=0.10, color=col, alpha=0.85, left=0.10)
        ax5.text(
            0.10 + bar_len + 0.02,
            y0,
            f"{score:.2f}",
            ha="left",
            va="center",
            fontsize=7,
            color=col,
            fontweight="bold",
        )
        ax5.text(
            0.10, y0 - 0.07, lbl, ha="left", va="top", fontsize=6.5, color=col, fontweight="bold"
        )

    tau_x = 0.10 + (tau / max_v) * 0.80
    ax5.axvline(tau_x, ymin=0.30, ymax=0.95, color="#333", lw=1.6, ls="--")
    ax5.text(
        tau_x + 0.01,
        0.88,
        "τ\n(threshold)",
        ha="left",
        va="top",
        fontsize=7,
        color="#333",
        fontweight="bold",
    )
    ax5.annotate(
        "ADV\n(reject)",
        xy=(tau_x + 0.04, 0.38),
        xytext=(tau_x + 0.04, 0.26),
        ha="center",
        fontsize=6.5,
        color=C_BASE,
        arrowprops=dict(arrowstyle="->", color=C_BASE, lw=1.0),
    )
    ax5.annotate(
        "OOD /ID\n(safe)",
        xy=(tau_x - 0.12, 0.62),
        xytext=(tau_x - 0.12, 0.26),
        ha="center",
        fontsize=6.5,
        color=C_COMP,
        arrowprops=dict(arrowstyle="->", color=C_COMP, lw=1.0),
    )

    ax5.text(
        0.50,
        0.10,
        r"$V(x) > \tau \Rightarrow$ \textbf{ADV}",
        ha="center",
        va="center",
        fontsize=7,
        color=C_BASE,
        bbox=dict(boxstyle="round,pad=0.25", fc="#fff0f0", ec=C_BASE, lw=0.8),
    )

    # ─── title + step labels ─────────────────────────────────────────────────
    fig.text(
        0.50,
        0.995,
        "How Viyog works: from image to adversarial detection in one forward pass",
        ha="center",
        va="top",
        fontsize=10,
        fontweight="bold",
        color="#222",
    )

    step_labels = [
        ("Run the\nfirst conv", 0),
        ("Rank channels\nby quietness", 2),
        ("Inspect the\ndormant band", 4),
        ("Compute the\nshape score V(x)", 6),
        ("Threshold\nand route", 8),
    ]
    col_rights = np.cumsum(panel_w) / sum(panel_w)
    col_lefts = np.r_[0, col_rights[:-1]]
    for label, col_idx in step_labels:
        cx = float(col_lefts[col_idx] + (col_rights[col_idx] - col_lefts[col_idx]) * 0.5)
        fig.text(
            cx, 0.04, label, ha="center", va="bottom", fontsize=6.5, color="#444", style="italic"
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
