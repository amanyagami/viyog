"""Plots for the embedded / edge-deployment evidence (A-w5/A-d7, B-1).

Reads the three edge CSVs and renders a 2x2 summary figure (plus individual panels):

  A. detector STATE memory per method (log scale) — Viyog's O(C) vs O(D^2)/O(N*D)
  B. per-routed-sample COMPUTE (GMACs, log) — Viyog (first conv) vs full-forward baselines
  C. edge-CPU latency: first-conv fraction of full forward, per model (ONNX-RT, 1 thread)
  D. accelerator first-conv ENERGY fraction (ZigZag Edge-TPU / Tesla-NPU), if available

    python experiments/plot_edge_metrics.py
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
VIYOG_C = "#c0392b"  # highlight Viyog
BASE_C = "#5b6770"  # baselines
ACC_C = "#2c7fb8"


def _bar_colors(labels: list[str]) -> list[str]:
    return [VIYOG_C if "viyog" in l.lower() else BASE_C for l in labels]


def panel_state(ax) -> None:
    c = pd.read_csv(A / "detector_cost_compute.csv")
    g = c.groupby("detector")["state_KB"].mean().sort_values()
    g = g.clip(lower=5e-3)  # floor for log display (state-free methods)
    labels = list(g.index)
    ax.barh(labels, g.values, color=_bar_colors(labels))
    ax.set_xscale("log")
    ax.set_xlabel("detector state (KB, log)")
    ax.set_title("A. Detector state memory (lower = better)")
    for y, v in enumerate(g.values):
        ax.text(v * 1.3, y, f"{v:.2g}" if v < 1 else f"{v:,.0f}", va="center", fontsize=7)
    ax.margins(x=0.25)


def panel_compute(ax) -> None:
    c = pd.read_csv(A / "detector_cost_compute.csv")
    g = (
        c.groupby("detector")
        .agg(macs=("macs_per_img_G", "mean"), fwd=("fwd_passes", "mean"))
        .sort_values("macs")
    )
    labels = list(g.index)
    ax.barh(labels, g["macs"].values, color=_bar_colors(labels))
    ax.set_xscale("log")
    ax.set_xlabel("compute (GMACs / routed sample, log)")
    ax.set_title("B. Per-sample compute (lower = better)")
    for y, (m, f) in enumerate(zip(g["macs"].values, g["fwd"].values)):
        ax.text(m * 1.3, y, f"{f:.2g}x fwd", va="center", fontsize=7)
    ax.margins(x=0.30)


def panel_latency(ax) -> None:
    e = pd.read_csv(A / "edge_latency.csv")
    e = e[e.model != "densenet121"]  # FP32-ONNX concat pathology (see EDGE_METRICS.md)
    x = np.arange(len(e))
    ax.bar(x, e["firstconv_lat_ratio_%"], color=VIYOG_C)
    ax.set_xticks(x)
    ax.set_xticklabels(e["model"], rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("first-conv latency (% of full)")
    ax.set_title("C. Edge-CPU: Viyog stage cost (ONNX-RT, 1 thread)")
    for xi, v in zip(x, e["firstconv_lat_ratio_%"]):
        ax.text(xi, v + 0.1, f"{v:.1f}%", ha="center", fontsize=7)
    ax.set_ylim(0, max(e["firstconv_lat_ratio_%"]) * 1.35)


def panel_accel(ax) -> None:
    f = A / "accelerator_energy.csv"
    if not f.exists() or not len(pd.read_csv(f)):
        ax.text(
            0.5,
            0.5,
            "accelerator_energy.csv\nnot available",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=9,
            color="gray",
        )
        ax.set_title("D. Accelerator first-conv energy")
        ax.axis("off")
        return
    d = pd.read_csv(f)
    piv = d.pivot_table(
        index="workload", columns="accelerator", values="E_firstconv_%", aggfunc="mean"
    )
    piv.plot(
        kind="bar",
        ax=ax,
        color=[ACC_C, "#7fcdbb", "#edf8b1"][: piv.shape[1]],
        width=0.75,
        legend=True,
    )
    ax.set_ylabel("first-conv energy (% of full)")
    ax.set_xlabel("")
    ax.set_title("D. Accelerator: first-conv energy share (ZigZag)")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(fontsize=7, title=None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25})

    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    panel_state(axs[0, 0])
    panel_compute(axs[0, 1])
    panel_latency(axs[1, 0])
    panel_accel(axs[1, 1])
    fig.suptitle(
        "Viyog embedded cost: state, compute, edge-CPU latency, accelerator energy",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = args.out or str(config.PLOTS_DIR / "edge_summary.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"saved -> {out}")

    # individual panels too
    for name, fn in [
        ("edge_state_memory", panel_state),
        ("edge_compute", panel_compute),
        ("edge_cpu_latency", panel_latency),
        ("edge_accel_energy", panel_accel),
    ]:
        f1, a1 = plt.subplots(figsize=(6, 4))
        fn(a1)
        f1.tight_layout()
        p = str(config.PLOTS_DIR / f"{name}.png")
        f1.savefig(p, dpi=140, bbox_inches="tight")
        plt.close(f1)
        print(f"saved -> {p}")


if __name__ == "__main__":
    main()
