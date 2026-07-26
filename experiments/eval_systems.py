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

The "full forward" MAC count is computed with `fvcore.nn.FlopCountAnalysis` (a
maintained, trace-based counter with handlers for matmul/bmm/einsum/attention,
not just Conv2d/Linear) — this is the correct denominator for
firstconv_mac_ratio_%. An earlier version of this script used a hand-rolled
Conv2d/Linear-only forward-hook (`total_macs`, kept below for the audit trail)
that silently undercounts any architecture whose forward pass isn't pure
conv+linear: it missed attention matmuls entirely (verified: it captured only
~1% of true MACs on ViT-base and Swin-tiny) and missed the H×W spatial
multiplier for ConvNeXtV2's per-position Linear "convs" (~4% of true MACs).
Both `macs_full_G` (fvcore, correct) and `macs_full_hookonly_G` (the old
hand-rolled count, for transparency/comparison) are always reported.

The GPU latency numbers above are wall-clock on whatever GPU happens to run
this script, so they aren't reproducible across reviewer hardware. --onnx-cpu
adds a second, portable latency anchor: export each model (and the first-conv
layer alone) to ONNX and time it under onnxruntime's CPUExecutionProvider —
every reviewer regenerates the same *relative* full-vs-first-conv cost on
whatever CPU they have, independent of GPU model. Best-effort per architecture
(falls back to NaN + a warning if an export fails, never aborts the run).

CPU-light; one GPU briefly. Run on a free GPU:
    CUDA_VISIBLE_DEVICES=1 python experiments/eval_systems.py --dataset cifar100 \
        --models mobilenetv3_l effnet_lite0 resnet50 [--csv out.csv] [--onnx-cpu]
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


def total_macs_hookonly(model: nn.Module, x: torch.Tensor) -> tuple[int, dict]:
    """Sum conv+linear MACs via forward hooks for one input.

    Kept only as an audit-trail comparison against fvcore_macs() (below), which
    is the correct full-model MAC count used for firstconv_mac_ratio_%. This
    hook undercounts any architecture using attention (misses matmul/bmm
    entirely) or per-position Linear "convs" (misses the H×W multiplier) — do
    not use it as the ratio denominator.
    """
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


def fvcore_macs(model: nn.Module, x: torch.Tensor) -> float:
    """Full-model MAC count via fvcore's trace-based FlopCountAnalysis.

    This is the correct denominator for firstconv_mac_ratio_%: fvcore has
    handlers for matmul/bmm/einsum/addmm (attention) in addition to conv/linear,
    and its "flop" convention counts one fused multiply-add as one unit — the
    same convention conv_macs()/total_macs_hookonly() use, so values are
    directly comparable with no factor-of-2 conversion needed.
    """
    from fvcore.nn import FlopCountAnalysis

    analysis = FlopCountAnalysis(model, x)
    analysis.unsupported_ops_warnings(False)
    analysis.uncalled_modules_warnings(False)
    return float(analysis.total())


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


def onnx_cpu_latency_ms(model: nn.Module, x_shape: tuple[int, ...], iters: int = 20) -> float:
    """Export `model` to ONNX and time a warm CPU forward pass (ms/batch).

    A hardware-portable companion to time_forward()'s GPU wall-clock: every
    reviewer gets the same execution provider (ONNX Runtime, CPUExecutionProvider)
    regardless of which GPU (or none) they have, so the *ratio* between full and
    first-conv-only latency is reproducible across machines even though absolute
    GPU latency isn't.
    """
    import io

    import onnxruntime as ort

    x_cpu = torch.randn(*x_shape)
    buf = io.BytesIO()
    torch.onnx.export(
        model.cpu().eval(),
        x_cpu,
        buf,
        input_names=["x"],
        output_names=["y"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,  # legacy TorchScript-based exporter; avoids an onnxscript dependency
    )
    sess = ort.InferenceSession(buf.getvalue(), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    feed = {in_name: x_cpu.numpy()}

    for _ in range(3):
        sess.run(None, feed)
    t0 = time.perf_counter()
    for _ in range(iters):
        sess.run(None, feed)
    return 1e3 * (time.perf_counter() - t0) / iters


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar100", choices=list(config.DATASET_SPECS))
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--csv", default=None)
    ap.add_argument(
        "--onnx-cpu",
        action="store_true",
        help="Add a hardware-portable ONNX Runtime CPU latency anchor (full model + "
        "first-conv-only) alongside the GPU wall-clock numbers. Best-effort per "
        "architecture; slower (exports + times on CPU), so opt-in.",
    )
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

        # MACs: fvcore is the correct full-model count (handles attention);
        # the hand-rolled hook is kept alongside only as an audit-trail column.
        full_macs = fvcore_macs(nm, x1)
        hookonly_macs, _ = total_macs_hookonly(nm, x1)
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
            "macs_full_hookonly_G": round(hookonly_macs / 1e9, 3),
            "hookonly_coverage_%": round(100 * hookonly_macs / max(full_macs, 1), 2),
            "macs_firstconv_M": round(fc_macs / 1e6, 2),
            "firstconv_mac_ratio_%": round(100 * fc_macs / max(full_macs, 1), 4),
            "lat_full_ms_per_img": round(ms_full / 64, 4),
            "lat_firstconv_ms_per_img": round(ms_fc / 64, 4),
            "throughput_full_img_s": round(64 * 1e3 / ms_full, 1),
            "peak_mem_b1_MB": round(mem1, 1),
            "peak_mem_b64_MB": round(mem64, 1),
        }
        if args.onnx_cpu:
            # Moves nm to CPU in-place — must run last, after every GPU-based
            # measurement above, since nothing GPU-side needs `nm` afterward.
            in_ch = first.in_channels
            try:
                row["lat_full_onnxcpu_ms"] = round(onnx_cpu_latency_ms(nm, (1, 3, 224, 224)), 4)
            except Exception as e:  # noqa: BLE001
                print(f"  [onnx-cpu] {model}: full-model export/timing failed ({e})")
                row["lat_full_onnxcpu_ms"] = float("nan")
            try:
                row["lat_firstconv_onnxcpu_ms"] = round(
                    onnx_cpu_latency_ms(first, (1, in_ch, 224, 224)), 4
                )
            except Exception as e:  # noqa: BLE001
                print(f"  [onnx-cpu] {model}: first-conv export/timing failed ({e})")
                row["lat_firstconv_onnxcpu_ms"] = float("nan")

        rows.append(row)
        msg = (f"  {model:16} params={row['params_M']}M  fc_MAC%={row['firstconv_mac_ratio_%']}  "
               f"(hookonly covered {row['hookonly_coverage_%']}% of true MACs)  "
               f"lat_full={row['lat_full_ms_per_img']}ms  lat_fc={row['lat_firstconv_ms_per_img']}ms  "
               f"mem_b1={row['peak_mem_b1_MB']}MB")
        if args.onnx_cpu:
            msg += (f"  onnxcpu_full={row['lat_full_onnxcpu_ms']}ms  "
                    f"onnxcpu_fc={row['lat_firstconv_onnxcpu_ms']}ms")
        print(msg)
        del nm
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    out = args.csv or str(config.ANALYSIS_DIR / f"systems_{args.dataset}.csv")
    df.to_csv(out, index=False)
    print(f"\n  saved → {out}")


if __name__ == "__main__":
    main()
