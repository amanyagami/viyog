"""Embedded-CPU latency + deployable size via ONNX Runtime (closes A-w5/A-d7/B-1).

`eval_systems.py` measures latency on the H200 GPU, which is not an edge target.
This measures the numbers an embedded reviewer actually wants, on CPU with ONNX
Runtime, single-threaded (a Cortex-A-class proxy):

  * FULL forward latency (ms/img) vs FIRST-CONV-ONLY latency (what Viyog costs:
    normalize + first conv + an L-inf/TV reduction) -> Viyog's fraction of full cost,
  * INT8 (static-quantized) full-forward latency -> the quantized-edge feasibility,
  * deployable model size on disk (FP32 vs INT8 .onnx) -> the storage footprint.

Architecture-only (random weights): latency and size are weight-value independent,
so no checkpoints / GPU / dataset are needed. INT8 is static-quantized with a small
random calibration set (we measure SIZE and LATENCY here, not accuracy; a deployment
would calibrate on real ID data).

Run pinned to idle cores on the shared box so the GPU-side finetunes don't skew the
single-thread timing:
    taskset -c 180-191 python experiments/eval_edge.py \
        --models mobilenetv3_l effnet_lite0 fastvit_sa12 resnet50 densenet121 [--iters 50]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import config
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from model_utils import find_first_conv, load_model
from onnxruntime.quantization import CalibrationDataReader, QuantType, quantize_static

ort.set_default_logger_severity(3)  # errors only


class NormFirstConv(nn.Module):
    """Viyog's compute path: normalize [0,1] input then run only the first conv."""

    def __init__(
        self, first: nn.Conv2d, mean: tuple[float, float, float], std: tuple[float, float, float]
    ) -> None:
        super().__init__()
        self.first = first
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.first((x - self.mean) / self.std)


class NormFull(nn.Module):
    """Full forward with internal [0,1] normalization (matches NormalizedModel)."""

    def __init__(
        self, backbone: nn.Module, mean: tuple[float, float, float], std: tuple[float, float, float]
    ) -> None:
        super().__init__()
        self.model = backbone
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model((x - self.mean) / self.std)


class RandReader(CalibrationDataReader):
    """Yield random [0,1] inputs for static-quant calibration (size/latency only)."""

    def __init__(self, input_name: str, n: int = 16) -> None:
        self.data = iter(
            [{input_name: np.random.rand(1, 3, 224, 224).astype(np.float32)} for _ in range(n)]
        )

    def get_next(self):
        return next(self.data, None)


def export(module: nn.Module, path: Path) -> None:
    x = torch.randn(1, 3, 224, 224)
    module.eval()
    torch.onnx.export(
        module, x, str(path), opset_version=17, input_names=["x"], output_names=["y"], dynamo=False
    )


def session(path: Path) -> ort.InferenceSession:
    """Single-thread CPU session (edge proxy)."""
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def bench(sess: ort.InferenceSession, iters: int) -> float:
    """Return ms/img at batch 1, warm-timed."""
    name = sess.get_inputs()[0].name
    x = np.random.rand(1, 3, 224, 224).astype(np.float32)
    for _ in range(10):
        sess.run(None, {name: x})
    t0 = time.perf_counter()
    for _ in range(iters):
        sess.run(None, {name: x})
    return 1e3 * (time.perf_counter() - t0) / iters


def mb(path: Path) -> float:
    return path.stat().st_size / 1e6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-classes", type=int, default=config.NUM_CLASSES)
    ap.add_argument(
        "--models",
        nargs="+",
        default=["mobilenetv3_l", "effnet_lite0", "fastvit_sa12", "resnet50", "densenet121"],
    )
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    import pandas as pd

    tmp = config.ANALYSIS_DIR / "onnx_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    mean, std = config.IMAGENET_MEAN, config.IMAGENET_STD
    rows = []
    print(f"=== ONNX-Runtime CPU edge latency (1 thread, batch 1, {args.iters} iters) ===\n")
    print(
        f"{'model':16} {'full_fp32':>10} {'firstconv':>10} {'fc/full':>8} "
        f"{'full_int8':>10} {'int8 speedup':>12} {'onnx MB fp32->int8':>20}"
    )
    for model in args.models:
        arch = config.MODEL_ARCHS.get(model)
        if arch is None:
            print(f"[skip] {model}: not in MODEL_ARCHS")
            continue
        try:
            backbone = load_model(arch, None, num_classes=args.num_classes, device="cpu")
            _, first = find_first_conv(backbone)
            full_mod = NormFull(backbone, mean, std)
            fc_mod = NormFirstConv(first, mean, std)

            p_full = tmp / f"{model}_full.onnx"
            p_fc = tmp / f"{model}_fc.onnx"
            p_int8 = tmp / f"{model}_full_int8.onnx"
            export(full_mod, p_full)
            export(fc_mod, p_fc)

            s_full, s_fc = session(p_full), session(p_fc)
            lat_full = bench(s_full, args.iters)
            lat_fc = bench(s_fc, args.iters)

            # static INT8 (quantizes conv+linear) — random calibration (size/latency only)
            lat_int8, size_int8 = float("nan"), float("nan")
            try:
                quantize_static(
                    str(p_full),
                    str(p_int8),
                    RandReader(s_full.get_inputs()[0].name, n=16),
                    weight_type=QuantType.QInt8,
                )
                s_int8 = session(p_int8)
                lat_int8 = bench(s_int8, args.iters)
                size_int8 = mb(p_int8)
            except Exception as e:
                print(f"   [warn] int8 {model}: {type(e).__name__}: {e}")

            size_full = mb(p_full)
            row = {
                "model": model,
                "arch": arch,
                "lat_full_fp32_ms": round(lat_full, 4),
                "lat_firstconv_fp32_ms": round(lat_fc, 4),
                "firstconv_lat_ratio_%": round(100 * lat_fc / lat_full, 3),
                "lat_full_int8_ms": round(lat_int8, 4),
                "int8_speedup_x": round(lat_full / lat_int8, 2)
                if lat_int8 == lat_int8
                else float("nan"),
                "onnx_fp32_MB": round(size_full, 2),
                "onnx_int8_MB": round(size_int8, 2),
                "size_reduction_x": round(size_full / size_int8, 2)
                if size_int8 == size_int8
                else float("nan"),
            }
            rows.append(row)
            print(
                f"{model:16} {row['lat_full_fp32_ms']:>9.3f}m {row['lat_firstconv_fp32_ms']:>9.3f}m "
                f"{row['firstconv_lat_ratio_%']:>7.2f}% {row['lat_full_int8_ms']:>9.3f}m "
                f"{row['int8_speedup_x']:>11.2f}x {size_full:>8.1f}->{size_int8:<8.1f}MB",
                flush=True,
            )
            for p in (p_full, p_fc, p_int8):
                p.unlink(missing_ok=True)
            del backbone, full_mod, fc_mod
        except Exception as e:
            print(f"[fail] {model}: {type(e).__name__}: {e}", flush=True)

    df = pd.DataFrame(rows)
    out = args.csv or str(config.ANALYSIS_DIR / "edge_latency.csv")
    df.to_csv(out, index=False)
    if len(df):
        print("\n=== means ===")
        print(
            f"  first-conv latency fraction of full : {df['firstconv_lat_ratio_%'].mean():.2f}% "
            f"(={100 / df['firstconv_lat_ratio_%'].mean():.1f}x faster than full forward)"
        )
        print(f"  INT8 full-forward speedup           : {df['int8_speedup_x'].mean():.2f}x")
        print(f"  INT8 size reduction                 : {df['size_reduction_x'].mean():.2f}x")
    print(f"\n  saved -> {out}")


if __name__ == "__main__":
    main()
