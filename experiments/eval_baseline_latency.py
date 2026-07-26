"""Measured end-to-end latency + memory of every pytorch_ood baseline detector.

The paper argues Viyog's *time* advantage by forward-pass count and complexity
order (O(1)/O(D^2)/O(ND)); reviewers want it *measured*. This times each detector's
real detection pipeline on the same GPU -- the forward pass(es) it needs PLUS its
actual scoring head -- with CUDA events, and records its fitted state memory and
peak activation memory. Detectors:

  * Viyog / Viyog-Linf  : first conv only + an O(C) reduction (partial forward)
  * MSP / Energy          : 1 full forward + a logit reduction
  * ODIN                  : forward + backward + a second forward (input preproc.)
  * MCD (MC-Dropout)      : 30 stochastic full forwards
  * Mahalanobis/KNN/ViM   : 1 full forward + a feature-space distance/search head

Latency/memory are weight-value independent for a fixed architecture, so random
ID features are fine to FIT the distance heads (we measure cost, not AUROC); we
still load the real finetuned model so the timed forward is the deployed one.

    CUDA_VISIBLE_DEVICES=4 uv run --with nvidia-ml-py python \
        experiments/eval_baseline_latency.py --gpu 4 \
        --models resnet50 mobilenetv3_l densenet121
"""

from __future__ import annotations

import argparse
import os
import time

import config
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_utils import find_first_conv_in_normalized, load_normalized_model
from pytorch_ood.detector import KNN, Mahalanobis, ViM

EPS = 1e-6
BAND_FRAC = 0.10
F32 = 4


class Penult:
    """Hook the classifier input to read penultimate features D."""

    def __init__(self, norm_model: nn.Module) -> None:
        self.feat: torch.Tensor | None = None
        clf = self._find_classifier(norm_model.model)
        self.h = clf.register_forward_hook(self._cap)

    def _cap(self, _m: nn.Module, inp: tuple, _o: torch.Tensor) -> None:
        self.feat = inp[0].detach()

    @staticmethod
    def _find_classifier(backbone: nn.Module) -> nn.Linear:
        last = None
        for m in backbone.modules():
            if isinstance(m, nn.Linear):
                last = m
        if last is None:
            raise RuntimeError("no Linear classifier")
        return last

    def close(self) -> None:
        self.h.remove()


class ViyogD(nn.Module):
    """Deployed detector path: normalize -> first conv -> dorm-band roughness V."""

    def __init__(self, first: nn.Conv2d, mean: tuple, std: tuple) -> None:
        super().__init__()
        self.first = first
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))
        k = max(1, round(BAND_FRAC * first.out_channels))
        self.register_buffer("band", torch.arange(k))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.first((x - self.mean) / self.std)
        ma = a.abs().mean(dim=(2, 3))
        dh = (a[:, :, 1:, :] - a[:, :, :-1, :]).abs().mean(dim=(2, 3))
        dw = (a[:, :, :, 1:] - a[:, :, :, :-1]).abs().mean(dim=(2, 3))
        return ((0.5 * (dh + dw)) / (ma + EPS)).index_select(1, self.band).mean(dim=1)


@torch.no_grad()
def t_ms(fn, x, iters: int = 50, warmup: int = 10) -> float:  # noqa: ANN001
    """Wall-clock ms/batch with CUDA sync (captures both GPU and CPU-host work)."""
    for _ in range(warmup):
        fn(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(x)
    torch.cuda.synchronize()
    return 1e3 * (time.perf_counter() - t0) / iters


def t_ms_grad(fn, x, iters: int = 20, warmup: int = 5) -> float:  # noqa: ANN001
    """Wall-clock timing that allows autograd (for ODIN)."""
    for _ in range(warmup):
        fn(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(x)
    torch.cuda.synchronize()
    return 1e3 * (time.perf_counter() - t0) / iters


def state_bytes(det: object) -> int:
    """Sum bytes of all tensors/arrays stored on a fitted detector (measured state).

    Walks torch tensors and numpy arrays (the KNN bank lives in a sklearn estimator
    as a numpy array, not a torch tensor).
    """
    import numpy as np

    total = 0
    seen = set()

    def walk(o: object, depth: int = 0) -> None:
        nonlocal total
        if depth > 5 or id(o) in seen:
            return
        seen.add(id(o))
        if torch.is_tensor(o):
            total += o.numel() * o.element_size()
        elif isinstance(o, np.ndarray):
            total += o.nbytes
        elif isinstance(o, dict):
            for v in o.values():
                walk(v, depth + 1)
        elif isinstance(o, (list, tuple, set)):
            for v in o:
                walk(v, depth + 1)
        elif hasattr(o, "__dict__"):
            for v in vars(o).values():
                walk(v, depth + 1)

    walk(det)
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=4)
    ap.add_argument("--dataset", default="cifar100")
    ap.add_argument("--models", nargs="+", default=["resnet50", "mobilenetv3_l", "densenet121"])
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--n-fit", type=int, default=5000)
    ap.add_argument("--mcd-passes", type=int, default=30)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    config.set_dataset(args.dataset)
    # Map a physical GPU index to torch's index (CUDA_VISIBLE_DEVICES-aware).
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    vis = [int(g) for g in cvd.split(",") if g.strip().isdigit()] if cvd else []
    tidx = vis.index(args.gpu) if args.gpu in vis else args.gpu
    dev = f"cuda:{tidx}"
    config.DEVICE = dev
    torch.cuda.set_device(tidx)
    mean, std = config.IMAGENET_MEAN, config.IMAGENET_STD
    sz, b = config.IMAGE_SIZE, args.batch

    rows = []
    for model in args.models:
        arch = config.MODEL_ARCHS[model]
        wp = config.weight_path(args.dataset, model)
        if not wp.exists():
            print(f"[skip] {model}: no weights")
            continue
        nm = load_normalized_model(arch, wp, num_classes=config.NUM_CLASSES, device=dev).eval()
        for p in nm.parameters():
            p.requires_grad_(False)
        _, first = find_first_conv_in_normalized(nm)
        viyog = ViyogD(first, mean, std).to(dev).eval()
        pen = Penult(nm)
        print(f"\n=== {model} ({arch}) ===", flush=True)

        x = torch.rand(b, 3, sz, sz, device=dev)
        with torch.no_grad():
            nm(x)  # populate pen.feat via the classifier hook
            D = pen.feat.shape[1]
            K = config.NUM_CLASSES
        # synthetic ID fit features (cost is value-independent). The distance heads
        # (KNN sklearn, Maha/ViM) run on the CPU host, as deployed -- fit on CPU.
        ztr = torch.randn(args.n_fit, D)
        ytr = torch.randint(0, K, (args.n_fit,))
        z = pen.feat.clone()  # GPU features straight off the forward pass
        wclf = (
            nm.model.get_classifier().weight.detach().cpu()
            if hasattr(nm.model, "get_classifier")
            else torch.randn(K, D)
        )

        # fit feature-distance heads
        maha = Mahalanobis(None).fit_features(ztr, ytr)
        knn = KNN(None).fit_features(ztr, ytr)
        vim = ViM(None, d=min(64, D - 1), w=wclf, b=torch.zeros(K)).fit_features(ztr, ytr)

        # latency components (ms / image). Logit detectors re-run the forward in
        # their timed op; feature detectors add the forward (t_fwd) to the head.
        # Loop vars bound as default args so the closures capture this iteration's.
        def lin_inf(xx, v=viyog):  # noqa: ANN001, ANN202
            return v.first((xx - v.mean) / v.std).abs().amax(dim=(1, 2, 3))

        t_fwd = t_ms(nm, x) / b
        lat = {
            "Viyog": t_ms(viyog, x) / b,
            "Viyog-Linf": t_ms(lin_inf, x) / b,
            "MSP": t_ms(lambda xx, m=nm: -m(xx).softmax(1).max(1).values, x) / b,
            "Energy": t_ms(lambda xx, m=nm: -torch.logsumexp(m(xx), 1), x) / b,
            # forward (GPU) + transfer features to host + distance/search head (CPU)
            "Mahalanobis": t_fwd + t_ms(lambda zz, d=maha: d.predict_features(zz.cpu()), z) / b,
            "KNN": t_fwd + t_ms(lambda zz, d=knn: d.predict_features(zz.cpu()), z) / b,
            "ViM": t_fwd + t_ms(lambda zz, d=vim: d.predict_features(zz.cpu()), z) / b,
        }

        # ODIN: forward + backward + 2nd forward (measured, autograd on)
        def odin(xx, m=nm):  # noqa: ANN001, ANN202
            xx = xx.clone().requires_grad_(True)
            lo = m(xx)
            loss = F.cross_entropy(lo, lo.argmax(1))
            (g,) = torch.autograd.grad(loss, xx)
            return m(xx - 1e-3 * g.sign())

        lat["ODIN"] = t_ms_grad(odin, x) / b

        # MCD: enable dropout, N stochastic forwards
        def mcd(xx, m=nm, passes=args.mcd_passes):  # noqa: ANN001, ANN202
            m.train()
            out = sum(m(xx).softmax(1) for _ in range(passes)) / passes
            m.eval()
            return out

        lat["MCD"] = t_ms(mcd, x, iters=5, warmup=2) / b

        # state memory (measured for distance heads; analytical for the rest)
        C = first.out_channels
        state = {
            "Viyog": F32 * (C + 2) + 8 * max(1, round(BAND_FRAC * C)),
            "Viyog-Linf": F32 * 2,
            "MSP": F32,
            "Energy": F32,
            "ODIN": F32 * 2,
            "MCD": F32,
            "Mahalanobis": state_bytes(maha),
            "KNN": state_bytes(knn),
            "ViM": state_bytes(vim),
        }
        n_fwd = {
            "Viyog": 0,
            "Viyog-Linf": 0,
            "MSP": 1,
            "Energy": 1,
            "ODIN": 3,
            "MCD": args.mcd_passes,
            "Mahalanobis": 1,
            "KNN": 1,
            "ViM": 1,
        }
        for d in lat:
            rows.append(
                {
                    "model": model,
                    "detector": d,
                    "lat_ms_per_img": round(max(lat[d], 0), 5),
                    "state_KB": round(state[d] / 1024, 3),
                    "n_fwd": n_fwd[d],
                    "lat_vs_viyogd": round(lat[d] / max(lat["Viyog"], 1e-9), 1),
                }
            )
            print(
                f"  {d:13} lat={lat[d]:.4f} ms/img  state={state[d] / 1024:8.2f} KB  "
                f"({lat[d] / max(lat['Viyog'], 1e-9):.1f}x Viyog)",
                flush=True,
            )
        pen.close()
        del nm, viyog
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    out = args.csv or str(config.ANALYSIS_DIR / f"baseline_latency_{args.dataset}.csv")
    df.to_csv(out, index=False)
    print("\n=== mean across models (ms/img, state KB) ===")
    g = df.groupby("detector")[["lat_ms_per_img", "state_KB", "lat_vs_viyogd"]].mean().round(3)
    print(g.sort_values("lat_ms_per_img").to_string())
    print(f"\n  saved -> {out}")


if __name__ == "__main__":
    main()
