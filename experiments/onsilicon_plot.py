"""Plot the measured on-silicon detector cost + roofline edge projection (A7 figure).

Panel (a): per-architecture detector cost as a fraction of full inference -- the
exact MAC fraction and the *measured* (H200) batch-1 latency fraction side by
side, showing the deployed V(x) is a single-digit-% tax, now measured on silicon.
Panel (b): a compute-bound roofline projection of the detector's per-image energy
on real edge devices (Jetson Orin Nano, RPi5+Hailo-8L) -- the "easy simulation"
stand-in for a physical board.

    python experiments/onsilicon_plot.py --dataset cifar100
"""

from __future__ import annotations

import argparse

import config
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SHORT = {
    "mobilenetv3_l": "MobNetV3",
    "effnet_lite0": "EffLite0",
    "fastvit_sa12": "FastViT",
    "resnet50": "ResNet50",
    "densenet121": "DenseNet",
    "convnextv2_base": "ConvNeXt",
}
DEV_SHORT = {
    "JetsonOrinNano_67TOPS": "Orin Nano\n(67 TOPS, 25 W)",
    "JetsonOrinNano_7W": "Orin Nano\n(7 W mode)",
    "RPi5_Hailo8L_13TOPS": "RPi5+Hailo-8L\n(13 TOPS, 2.5 W)",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar100")
    ap.add_argument("--out", default="figs/rebuttal/fig_onsilicon.pdf")
    args = ap.parse_args()
    m = pd.read_csv(config.ANALYSIS_DIR / f"onsilicon_measured_{args.dataset}.csv")
    r = pd.read_csv(config.ANALYSIS_DIR / f"onsilicon_roofline_{args.dataset}.csv")

    fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.0), gridspec_kw={"width_ratios": [1.5, 1]})

    # (a) per-model detector fraction: exact MACs vs measured H200 batch-1 latency
    ax = axes[0]
    labels = [SHORT.get(x, x) for x in m.model]
    xs = range(len(m))
    w = 0.38
    ax.bar([x - w / 2 for x in xs], m.mac_ratio_pct, w, label="MAC fraction (exact)", color="C0")
    ax.bar(
        [x + w / 2 for x in xs],
        m.lat_ratio_b1_pct,
        w,
        label="latency (measured, H200, b=1)",
        color="C2",
    )
    ax.axhline(m.mac_ratio_pct.mean(), color="C0", ls=":", lw=1, alpha=0.7)
    ax.axhline(m.lat_ratio_b1_pct.mean(), color="C2", ls=":", lw=1, alpha=0.7)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("detector cost (% of full inference)")
    ax.set_title("(a) Cost is single-digit %, measured on silicon", fontsize=8.5)
    ax.set_ylim(0, max(m.lat_ratio_b1_pct.max(), m.mac_ratio_pct.max()) * 1.35)
    ax.legend(fontsize=6.5, loc="upper right")
    ax.grid(axis="y", alpha=0.25)

    # (b) roofline detector energy (uJ/img), mean across models, per edge device
    ax = axes[1]
    devs = [d for d in DEV_SHORT if d in set(r.device)]
    uj = [1e3 * r[r.device == d].det_mJ.mean() for d in devs]  # mJ -> uJ
    full_mj = [r[r.device == d].full_mJ.mean() for d in devs]
    ax.bar(range(len(devs)), uj, color="C3", alpha=0.85)
    for i, (u, f) in enumerate(zip(uj, full_mj, strict=False)):
        ax.text(i, u, f"{u:.0f} uJ\n({f:.1f} mJ full)", ha="center", va="bottom", fontsize=6)
    ax.set_xticks(range(len(devs)))
    ax.set_xticklabels([DEV_SHORT[d] for d in devs], fontsize=6.5)
    ax.set_ylabel("detector energy (uJ / img)")
    ax.set_title("(b) Roofline edge projection", fontsize=8.5)
    ax.set_ylim(0, max(uj) * 1.45)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print(f"saved -> {args.out}")
    print(f"\nMEAN MAC fraction      = {m.mac_ratio_pct.mean():.2f}%")
    print(f"MEAN measured latency  = {m.lat_ratio_b1_pct.mean():.2f}% (H200, batch 1)")
    for d in devs:
        dd = r[r.device == d]
        uj, us = 1e3 * dd.det_mJ.mean(), dd.det_ms.mean() * 1e3
        print(f"{d:22} det {uj:.1f} uJ / {us:.1f} us per img")


if __name__ == "__main__":
    main()
