"""Systems / embedded-deployment metrics per model (closes A-w5/A-d7, B-1).

Reviewers A and B ask for the embedded-hardware story to be quantified and
front-loaded. For every accepted model this reports the numbers that matter for
a resource-constrained second-stage detector:

  * total params, first-conv params,
  * total MACs and first-conv MACs for a 224×224 input → first-conv MAC RATIO
    (Viyog only needs the network up to the first conv + an L∞ reduction, so this
    ratio is an upper bound on Viyog's compute relative to a full forward),
  * peak GPU memory for a single forward at batch 1 and batch 64,
  * detection latency (ms/img) and throughput (img/s) for (a) the FULL forward and
    (b) FIRST-CONV-ONLY (what Viyog actually costs), warm-timed with CUDA sync.

CPU-light; one GPU briefly. Run on a free GPU:
    CUDA_VISIBLE_DEVICES=1 python experiments/eval_systems.py --dataset cifar100 \
        --models mobilenetv3_l effnet_lite0 resnet50 [--csv out.csv]
"""
from __future__ import annotations

import argparse
import time

import config
import torch
import torch.nn as nn
from config import DEVICE
from model_utils import find_first_conv_in_normalized, load_normalized_model


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def conv_macs(layer: nn.Conv2d, out_hw: tuple[int, int]) -> int:
    cout, cin = layer.out_channels, layer.in_channels // layer.groups
    kh, kw = layer.kernel_size
    oh, ow = out_hw
    return cout * cin * kh * kw * oh * ow


def total_macs(model: nn.Module, x: torch.Tensor) -> tuple[int, dict]:
    """Sum conv+linear MACs via forward hooks for one input."""
    macs = {"total": 0}
    handles = []

    def hook(mod, inp, out):
        if isinstance(mod, nn.Conv2d):
            oh, ow = out.shape[2], out.shape[3]
            cin = mod.in_channels // mod.groups
            macs["total"] += mod.out_channels * cin * mod.kernel_size[0] * mod.kernel_size[1] * oh * ow
        elif isinstance(mod, nn.Linear):
            macs["total"] += mod.in_features * mod.out_features

    for mod in model.modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            handles.append(mod.register_forward_hook(hook))
    with torch.no_grad():
        model(x)
    for h in handles:
        h.remove()
    return macs["total"], macs


@torch.no_grad()
def time_forward(model: nn.Module, x: torch.Tensor, iters: int = 50) -> float:
    """Return ms/batch, warm-timed with CUDA sync."""
    for _ in range(5):
        model(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        model(x)
    torch.cuda.synchronize()
    return 1e3 * (time.perf_counter() - t0) / iters


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar100", choices=list(config.DATASET_SPECS))
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    config.set_dataset(args.dataset)

    models = args.models or [m for m in config.MODEL_ARCHS
                             if config.weight_path(args.dataset, m).exists()]
    import pandas as pd
    rows = []
    print(f"=== systems metrics [{args.dataset}] models={models} ===")
    for model in models:
        arch = config.MODEL_ARCHS[model]
        wp = config.weight_path(args.dataset, model)
        if not wp.exists():
            print(f"[skip] {model}: no weights")
            continue
        nm = load_normalized_model(arch, wp, num_classes=config.NUM_CLASSES, device=DEVICE).eval()
        _, first = find_first_conv_in_normalized(nm)

        x1 = torch.randn(1, 3, 224, 224, device=DEVICE)
        x64 = torch.randn(64, 3, 224, 224, device=DEVICE)

        # MACs
        full_macs, _ = total_macs(nm, x1)
        with torch.no_grad():
            from model_utils import FirstLayerHook
            with FirstLayerHook(nm) as hk:
                nm(x1)
                oh, ow = hk.features.shape[2], hk.features.shape[3]
        fc_macs = conv_macs(first, (oh, ow))

        # latency / throughput (batch 64)
        ms_full = time_forward(nm, x64)

        # first-conv-only latency: time the first conv layer in isolation (Viyog's
        # forward cost is bounded by normalize + this conv + an L∞ reduction).
        try:
            ms_fc = time_forward(first, x64)
        except Exception:  # noqa: BLE001
            ms_fc = float("nan")

        # peak memory at batch 1 and 64
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            nm(x1)
        mem1 = torch.cuda.max_memory_allocated() / 1e6
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            nm(x64)
        mem64 = torch.cuda.max_memory_allocated() / 1e6

        row = {
            "model": model, "arch": arch,
            "params_M": round(count_params(nm) / 1e6, 2),
            "firstconv_params_K": round(count_params(first) / 1e3, 2),
            "macs_full_G": round(full_macs / 1e9, 3),
            "macs_firstconv_M": round(fc_macs / 1e6, 2),
            "firstconv_mac_ratio_%": round(100 * fc_macs / max(full_macs, 1), 4),
            "lat_full_ms_per_img": round(ms_full / 64, 4),
            "lat_firstconv_ms_per_img": round(ms_fc / 64, 4),
            "throughput_full_img_s": round(64 * 1e3 / ms_full, 1),
            "peak_mem_b1_MB": round(mem1, 1),
            "peak_mem_b64_MB": round(mem64, 1),
        }
        rows.append(row)
        print(f"  {model:16} params={row['params_M']}M  fc_MAC%={row['firstconv_mac_ratio_%']}  "
              f"lat_full={row['lat_full_ms_per_img']}ms  lat_fc={row['lat_firstconv_ms_per_img']}ms  "
              f"mem_b1={row['peak_mem_b1_MB']}MB")
        del nm
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    out = args.csv or str(config.ANALYSIS_DIR / f"systems_{args.dataset}.csv")
    df.to_csv(out, index=False)
    print(f"\n  saved → {out}")


if __name__ == "__main__":
    main()
