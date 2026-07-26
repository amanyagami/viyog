"""Generate the figure set for the latest experiments (measured cost, EOT, non-vision).

Three figures, each a distinct, easy-to-read chart type:
  * fig_baselines_cost : measured per-detector latency bars + a latency-vs-memory
    Pareto scatter (Viyog dominates the lower-left).
  * fig_eot            : stochastic-vs-fixed dorm-band AUROC under an EOT attacker,
    as a y=x scatter over 18 architectures (points above the line = stochastic wins).
  * fig_nonvision      : the 1D-signal sanity check -- shape vs raw L-inf AUROC.

    python experiments/plot_latest_results.py --out-dir <paper>/figs/rebuttal
"""

from __future__ import annotations

import argparse

import config
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

A = config.ANALYSIS_DIR


def fig_baselines_cost(out: str) -> None:
    d = pd.read_csv(A / "baseline_latency_cifar100.csv")
    g = (
        d.groupby("detector")
        .agg(
            lat=("lat_ms_per_img", "mean"), state=("state_KB", "mean"), x=("lat_vs_viyogd", "mean")
        )
        .reset_index()
        .sort_values("lat")
    )
    # Drop the raw-norm baseline: the deployed method is Viyog only.
    g = g[g.detector != "Viyog-Linf"].reset_index(drop=True)
    g["detector"] = g["detector"].replace({"Viyog-D": "Viyog (ours)", "Viyog": "Viyog (ours)"})

    def is_ours(name: str) -> bool:
        return name.startswith("Viyog")

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.1), gridspec_kw={"width_ratios": [1.15, 1]})
    # (a) latency bars (log x) — Viyog highlighted in green, baselines in red
    ax = axes[0]
    colors = ["#2ca02c" if is_ours(n) else "#d62728" for n in g.detector]
    ax.barh(g.detector, g.lat, color=colors, alpha=0.9)
    # bold the y-tick label for our method
    for tick in ax.get_yticklabels():
        if is_ours(tick.get_text()):
            tick.set_fontweight("bold")
            tick.set_color("#1a7a44")
    ax.set_xscale("log")
    for y, (lat, mult, name) in enumerate(zip(g.lat, g.x, g.detector, strict=False)):
        txt = f"{lat:.2f} ms" if is_ours(name) else f"{lat:.2f} ms ({mult:.0f}x slower)"
        ax.text(lat * 1.15, y, txt, va="center", fontsize=6,
                fontweight="bold" if is_ours(name) else "normal",
                color="#1a7a44" if is_ours(name) else "#333")
    ax.set_xlabel("detection latency (ms/img, log) — lower is better")
    ax.set_title("(a) Measured detection latency (H200)", fontsize=8.5)
    ax.set_xlim(g.lat.min() * 0.5, g.lat.max() * 9)
    ax.grid(axis="x", alpha=0.25)

    # (b) Pareto scatter latency vs state memory (log-log)
    # Per-point text offsets to prevent overlap (tuned for typical layout)
    OFFSETS = {
        "MSP": (-4, -14), "Energy": (-4, 6), "KNN": (6, -10),
        "ODIN": (6, 4), "MCD": (6, 4),
    }
    ax = axes[1]
    for _, r in g.iterrows():
        ours = is_ours(r.detector)
        ax.scatter(
            r.lat, max(r.state, 1e-3),
            s=170 if ours else 45,
            marker="*" if ours else "o",
            color="#2ca02c" if ours else "#d62728",
            zorder=5 if ours else 3, edgecolor="k", linewidth=0.6,
        )
        off = OFFSETS.get(r.detector, (6, 4))
        ax.annotate(
            r.detector, (r.lat, max(r.state, 1e-3)),
            fontsize=7 if ours else 6,
            fontweight="bold" if ours else "normal",
            color="#1a7a44" if ours else "#333",
            xytext=off, textcoords="offset points",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("latency (ms/img, log)")
    ax.set_ylabel("state memory (KB, log)")
    ax.set_title("(b) Cost Pareto — Viyog alone in the cheap corner", fontsize=8.5)
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out}")


def fig_eot(out: str) -> None:
    rows = []
    for f in sorted(A.glob("eot_stochastic_cifar100_*.csv")):
        m = f.name.replace("eot_stochastic_cifar100_", "").replace(".csv", "")
        df = pd.read_csv(f)
        atk = df[df["mode"] != "pgd"] if (df["mode"] != "pgd").any() else df
        r = atk.loc[atk["lambda"].idxmax()]
        rows.append(dict(model=m, fixed=r.T2_fixed, stoch=r.T2_stoch))
    e = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.1), gridspec_kw={"width_ratios": [1, 0.8]})
    ax = axes[0]
    ax.plot([0.45, 1.02], [0.45, 1.02], "k--", lw=1, alpha=0.6)
    ax.fill_between([0.45, 1.02], [0.45, 1.02], 1.02, color="#2ca02c", alpha=0.07)
    ax.scatter(e.fixed, e.stoch, s=40, color="#1f77b4", edgecolor="k", linewidth=0.4, zorder=3)
    win = int((e.stoch > e.fixed).sum())
    ax.text(0.52, 0.97, f"{win}/{len(e)} above the line\n(stochastic wins)", fontsize=7, va="top")
    ax.set_xlabel("fixed dorm-band AUROC (under EOT)")
    ax.set_ylabel("stochastic dorm-band AUROC")
    ax.set_title("(a) Stochastic vs fixed band, 18 archs", fontsize=8.5)
    ax.set_xlim(0.45, 1.02)
    ax.set_ylim(0.45, 1.02)
    ax.grid(alpha=0.25)

    ax = axes[1]
    means = [e.fixed.mean(), e.stoch.mean()]
    errs = [e.fixed.std(), e.stoch.std()]
    ax.bar(
        ["fixed", "stochastic"],
        means,
        yerr=errs,
        capsize=4,
        color=["#d62728", "#2ca02c"],
        alpha=0.9,
    )
    for i, v in enumerate(means):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_ylabel("mean ID-vs-ADV AUROC (EOT)")
    ax.set_title("(b) Mean ± sd over 18 archs", fontsize=8.5)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out}")


def fig_nonvision(out: str) -> None:
    d = pd.read_csv(A / "nonvision_1d.csv")
    d["cond"] = d.attack.str.upper() + "\nε=" + d.eps.astype(str)
    x = np.arange(len(d))
    w = 0.38
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.bar(x - w / 2, d.T2_shape, w, label="dorm-band shape $V$", color="#2ca02c")
    ax.bar(x + w / 2, d.T2_linf, w, label=r"raw $L_\infty$", color="#d62728", alpha=0.85)
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    for xi, (s, suc) in enumerate(zip(d.T2_shape, d.succ, strict=False)):
        ax.text(xi - w / 2, s + 0.01, f"{s:.2f}", ha="center", fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(d.cond, fontsize=7)
    ax.set_ylabel("ID-vs-ADV AUROC (T2)")
    ax.set_title("Non-vision (1D signals): shape transfers, raw norm does not", fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="figs/rebuttal")
    args = ap.parse_args()
    fig_baselines_cost(f"{args.out_dir}/fig_baselines_cost.pdf")
    fig_eot(f"{args.out_dir}/fig_eot.pdf")
    fig_nonvision(f"{args.out_dir}/fig_nonvision.pdf")


if __name__ == "__main__":
    main()
