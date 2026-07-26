"""Detector-state memory + per-detector inference cost (closes A-w5/A-d7/B-1).

The headline embedded claim is that Viyog is a *resource-constrained* second-stage
detector: it stores almost no state and runs only the first conv, whereas the
strong distance baselines (Mahalanobis / KNN / ViM) must store a covariance / a
feature bank and run a *full* forward. This script makes that claim exact and
quotable, per model, with NO weights, NO GPU and NO dataset (architecture-only):

  * detector STATE memory (float32 bytes) that must live on the edge device:
        Viyog (L-inf) ........ O(1)          running mean inf-norm + threshold
        Viyog* ............. O(C)          per-filter ID-mean profile + band idx
        Mahalanobis .......... O(K*D + D^2)  class means + shared precision
        KNN .................. O(N*D)        fit-set feature bank
        ViM .................. O(D*(D-d)+KD) residual subspace + class means
        MCD / ODIN / Energy .. O(1)          temperature only (state-free)
  * per-image inference COMPUTE (MACs + forward passes):
        Viyog ................ first-conv MACs + an O(C*H*W) reduction  (1 partial fwd)
        Maha/KNN/ViM/Energy .. full forward + a distance/logit reduction (1 fwd)
        ODIN ................. full forward + backward + 2nd forward     (~3 fwd)
        MCD .................. 30 full forwards (MC dropout)

C = first-conv out-channels, D = penultimate (classifier-input) dim, K = #classes,
N = KNN/Mahalanobis fit-set size. Reports the Viyog-vs-baseline memory and compute
ratios that back the paper's "~6.8x less memory" figure.

    python experiments/eval_detector_cost.py --num-classes 100 --knn-n 5000 \
        --models mobilenetv3_l effnet_lite0 resnet50 densenet121 convnextv2_base
"""

from __future__ import annotations

import argparse
import math

import config
import torch
import torch.nn as nn
from model_utils import find_first_conv, load_model

F32 = 4  # bytes per float32 parameter of detector state
I64 = 8  # bytes per int64 band index


def penultimate_dim(backbone: nn.Module) -> int:
    """Return the classifier input dim D (penultimate feature width)."""
    clf = backbone.get_classifier() if hasattr(backbone, "get_classifier") else None
    if isinstance(clf, nn.Linear):
        return clf.in_features
    last = None
    for m in backbone.modules():
        if isinstance(m, nn.Linear):
            last = m
    if last is None:
        raise RuntimeError("no Linear classifier found")
    return last.in_features


def full_macs(model: nn.Module, x: torch.Tensor) -> int:
    """Sum conv+linear MACs for one forward via hooks (CPU, weight-agnostic)."""
    total = {"v": 0}
    handles = []

    def hook(mod, inp, out):
        if isinstance(mod, nn.Conv2d):
            oh, ow = out.shape[2], out.shape[3]
            cin = mod.in_channels // mod.groups
            total["v"] += mod.out_channels * cin * mod.kernel_size[0] * mod.kernel_size[1] * oh * ow
        elif isinstance(mod, nn.Linear):
            total["v"] += mod.in_features * mod.out_features

    for mod in model.modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            handles.append(mod.register_forward_hook(hook))
    with torch.no_grad():
        model(x)
    for h in handles:
        h.remove()
    return total["v"]


def firstconv_macs(
    model: nn.Module, first: nn.Conv2d, x: torch.Tensor
) -> tuple[int, tuple[int, int]]:
    """Return (first-conv MACs, output H/W) for one forward."""
    out_hw = {}
    h = first.register_forward_hook(lambda m, i, o: out_hw.update(hw=(o.shape[2], o.shape[3])))
    with torch.no_grad():
        model(x)
    h.remove()
    oh, ow = out_hw["hw"]
    cin = first.in_channels // first.groups
    macs = first.out_channels * cin * first.kernel_size[0] * first.kernel_size[1] * oh * ow
    return macs, (oh, ow)


def detector_states(C: int, D: int, K: int, N: int, band_pct: float = 5.0) -> dict[str, int]:
    """State memory in bytes for each detector, given the architecture dims."""
    band_k = max(1, math.ceil(band_pct / 100.0 * C))
    return {
        "Viyog_Linf": F32 * 2,  # running mean inf-norm + threshold
        "Viyog_D*": F32 * (C + 2) + I64 * band_k,  # ID-mean profile + band stats + band idx
        "Mahalanobis": F32 * (K * D + D * D),  # class means + shared precision
        "KNN": F32 * (N * D),  # feature bank
        "ViM": F32 * (D * (D - min(64, D - 1)) + K * D),  # residual subspace + class means
        "MCD": F32 * 1,  # state-free (MC dropout at inference)
        "Energy/MSP": F32 * 1,  # temperature only
        "ODIN": F32 * 2,  # temperature + eps
    }


def detector_compute(fc_macs: int, full: int) -> dict[str, tuple[int, float]]:
    """(MACs per image, effective forward passes) for each detector."""
    return {
        "Viyog_Linf": (fc_macs, fc_macs / full),
        "Viyog_D*": (fc_macs, fc_macs / full),
        "Mahalanobis": (full, 1.0),
        "KNN": (full, 1.0),
        "ViM": (full, 1.0),
        "Energy/MSP": (full, 1.0),
        "ODIN": (3 * full, 3.0),  # fwd + backward + perturbed fwd
        "MCD": (30 * full, 30.0),  # 30 MC-dropout forwards
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-classes", type=int, default=config.NUM_CLASSES)
    ap.add_argument("--knn-n", type=int, default=5000, help="Mahalanobis/KNN fit-set size N")
    ap.add_argument(
        "--models",
        nargs="+",
        default=[
            "mobilenetv3_l",
            "effnet_lite0",
            "fastvit_sa12",
            "resnet50",
            "densenet121",
            "convnextv2_base",
        ],
    )
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    K, N = args.num_classes, args.knn_n
    import pandas as pd

    x = torch.randn(1, 3, 224, 224)
    rows, comp_rows = [], []
    print(f"=== detector cost  (K={K} classes, N={N} fit-set)  arch-only, CPU ===\n")
    print(
        f"{'model':16} {'C':>4} {'D':>5} {'fcMAC%':>7}  "
        f"{'Maha/ViyogD':>12} {'KNN/ViyogD':>11}  state(ViyogD)"
    )
    for model in args.models:
        arch = config.MODEL_ARCHS.get(model)
        if arch is None:
            print(f"[skip] {model}: not in MODEL_ARCHS")
            continue
        backbone = load_model(arch, None, num_classes=K, device="cpu")  # no weights needed
        name, first = find_first_conv(backbone)
        C = first.out_channels
        D = penultimate_dim(backbone)
        full = full_macs(backbone, x)
        fc_macs, (oh, ow) = firstconv_macs(backbone, first, x)

        st = detector_states(C, D, K, N)
        cp = detector_compute(fc_macs, full)
        vd = st["Viyog_D*"]
        rows.append(
            {
                "model": model,
                "arch": arch,
                "C_firstconv": C,
                "D_penult": D,
                "K": K,
                "macs_full_G": round(full / 1e9, 3),
                "macs_firstconv_M": round(fc_macs / 1e6, 2),
                "firstconv_mac_ratio_%": round(100 * fc_macs / max(full, 1), 4),
                **{f"state_{k}_B": v for k, v in st.items()},
                "ratio_Maha_over_ViyogD": round(st["Mahalanobis"] / vd, 1),
                "ratio_KNN_over_ViyogD": round(st["KNN"] / vd, 1),
                "ratio_ViM_over_ViyogD": round(st["ViM"] / vd, 1),
            }
        )
        for det, (macs, fwd) in cp.items():
            comp_rows.append(
                {
                    "model": model,
                    "detector": det,
                    "macs_per_img_G": round(macs / 1e9, 4),
                    "fwd_passes": fwd,
                    "state_bytes": st[det],
                    "state_KB": round(st[det] / 1024, 2),
                }
            )
        print(
            f"{model:16} {C:>4} {D:>5} {100 * fc_macs / max(full, 1):>6.3f}%  "
            f"{st['Mahalanobis'] / vd:>11.1f}x {st['KNN'] / vd:>10.1f}x  "
            f"{vd / 1024:.2f} KB",
            flush=True,
        )
        del backbone

    df = pd.DataFrame(rows)
    cdf = pd.DataFrame(comp_rows)
    out = args.csv or str(config.ANALYSIS_DIR / "detector_cost.csv")
    df.to_csv(out, index=False)
    cdf.to_csv(out.replace(".csv", "_compute.csv"), index=False)

    print("\n=== detector STATE memory (KB) per detector (mean over models) ===")
    piv = cdf.groupby("detector")["state_KB"].mean().sort_values()
    print(piv.round(3).to_string())
    print("\n=== mean Viyog-vs-baseline ratios over models ===")
    print(f"  Mahalanobis / Viyog* state : {df['ratio_Maha_over_ViyogD'].mean():.1f}x")
    print(f"  KNN         / Viyog* state : {df['ratio_KNN_over_ViyogD'].mean():.1f}x")
    print(f"  ViM         / Viyog* state : {df['ratio_ViM_over_ViyogD'].mean():.1f}x")
    print(
        f"  first-conv MAC ratio (Viyog compute vs full fwd): "
        f"{df['firstconv_mac_ratio_%'].mean():.3f}%  "
        f"(={100 / df['firstconv_mac_ratio_%'].mean():.1f}x less compute)"
    )
    print(f"\n  saved -> {out}\n          {out.replace('.csv', '_compute.csv')}")


if __name__ == "__main__":
    main()
