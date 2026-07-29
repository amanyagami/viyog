"""Measured on-silicon cost of the deployed detector V(x) (closes A7 / Limitation vi).

Reviewer A7 asked for an *on-device* number: the paper so far has analytical MACs
(``eval_detector_cost.py``), a ZigZag *modeled* accelerator energy, and an ONNX-CPU
*latency* proxy (``eval_edge.py``) -- but **no measured energy on real silicon**.
Buying a Jetson/Pi is camera-ready work; meanwhile we can measure the real thing on
the accelerator we have. This script reports, per model, on a *real* GPU:

  * measured LATENCY (CUDA events) of the deployed detector path -- normalize +
    first conv + the dormant-band roughness reduction V(x) -- vs the full forward,
    across a batch sweep; and
  * measured ENERGY (NVML power integration) of the detector path vs the full
    forward at a throughput batch, with the idle floor subtracted (dynamic mJ/img).

It then projects to two real edge targets with a compute-bound roofline (the
"easy simulation" alternative to physical hardware): Jetson Orin Nano (67 INT8
TOPS) and a Raspberry Pi 5 + Hailo-8L AI kit (13 INT8 TOPS). The detector FRACTION
is device-independent (= the MAC ratio); the roofline supplies the absolute edge
per-image latency/energy.

Latency and energy are weight-VALUE independent for a fixed architecture, so the
measurement is exact regardless of the checkpoint; we still load the real
finetuned weights so the reported model is the deployed one.

    CUDA_VISIBLE_DEVICES="" uv run --with nvidia-ml-py python experiments/exp_onsilicon.py \
        --gpu 7 --models mobilenetv3_l effnet_lite0 fastvit_sa12 resnet50 \
        densenet121 convnextv2_base

NB: run with ``uv run --with nvidia-ml-py`` so pynvml is available without
mutating the shared project venv.
"""

from __future__ import annotations

import argparse
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import pairwise

import config
import pandas as pd
import torch
import torch.nn as nn

EPS = 1e-6
BAND_FRAC = 0.10  # bottom-10% dormant band (cost depends only on the band SIZE)

# Datasheet peak INT8 throughput / typical inference power for the roofline.
# (compute-bound lower bound; clearly labelled as a projection, not a measurement)
EDGE_DEVICES = {
    "JetsonOrinNano_67TOPS": {"ops_s": 67e12, "watt": 25.0},
    "JetsonOrinNano_7W": {"ops_s": 20e12, "watt": 7.0},
    "RPi5_Hailo8L_13TOPS": {"ops_s": 13e12, "watt": 2.5},
}


# --------------------------------------------------------------------------- #
# Detector compute path == the exact deployed V(x)
# --------------------------------------------------------------------------- #
class DetectorPath(nn.Module):
    """Normalize -> first conv -> dormant-band roughness V(x).

    V(x) = mean_{c in B} TV(a_c) / (mean|a_c| + eps),  TV = 1/2(mean|d_h|+mean|d_w|).
    The band B is a fixed index set of size ceil(BAND_FRAC*C): cost depends only on
    |B|, not on which channels (latency/energy are value-independent).
    """

    def __init__(
        self,
        first: nn.Conv2d,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
    ) -> None:
        super().__init__()
        self.first = first
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))
        c = first.out_channels
        k = max(1, round(BAND_FRAC * c))
        self.register_buffer("band", torch.arange(k))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.first((x - self.mean) / self.std)  # (B,C,H,W)
        mean_abs = a.abs().mean(dim=(2, 3))  # (B,C)
        dh = (a[:, :, 1:, :] - a[:, :, :-1, :]).abs().mean(dim=(2, 3))
        dw = (a[:, :, :, 1:] - a[:, :, :, :-1]).abs().mean(dim=(2, 3))
        shape = (0.5 * (dh + dw)) / (mean_abs + EPS)  # (B,C)
        return shape.index_select(1, self.band).mean(dim=1)  # (B,)


# --------------------------------------------------------------------------- #
# NVML power sampler
# --------------------------------------------------------------------------- #
@dataclass(eq=False)
class PowerSampler(threading.Thread):
    """Background NVML power sampler; trapezoidal energy over a window."""

    gpu_index: int
    period: float = 0.004
    samples: list[tuple[float, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        super().__init__(daemon=True)
        import pynvml

        self._nvml = pynvml
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
        self._stop_evt = threading.Event()

    def run(self) -> None:
        while not self._stop_evt.is_set():
            w = self._nvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
            self.samples.append((time.perf_counter(), w))
            time.sleep(self.period)

    def stop(self) -> None:
        self._stop_evt.set()
        self.join()

    def energy_joules(self, t0: float, t1: float) -> tuple[float, float]:
        """Return (energy_J, mean_W) over [t0, t1] by trapezoid integration."""
        pts = [(t, w) for t, w in self.samples if t0 <= t <= t1]
        if len(pts) < 2:
            return 0.0, 0.0
        e = 0.0
        for (ta, wa), (tb, wb) in pairwise(pts):
            e += 0.5 * (wa + wb) * (tb - ta)
        dur = pts[-1][0] - pts[0][0]
        return e, (e / dur if dur > 0 else 0.0)


def _torch_index_for_phys(phys: int) -> int:
    """Map a physical/NVML GPU index to the torch index (CUDA_VISIBLE_DEVICES-aware)."""
    import pynvml

    pynvml.nvmlInit()
    nvml_uuid = pynvml.nvmlDeviceGetUUID(pynvml.nvmlDeviceGetHandleByIndex(phys))
    if isinstance(nvml_uuid, bytes):
        nvml_uuid = nvml_uuid.decode()
    for i in range(torch.cuda.device_count()):
        if str(torch.cuda.get_device_properties(i).uuid) in nvml_uuid:
            return i
    raise RuntimeError(f"physical GPU {phys} not visible to torch (check CUDA_VISIBLE_DEVICES)")


def idle_power(gpu_index: int, seconds: float = 2.0) -> float:
    """Mean GPU power (W) with no work, as the static floor."""
    s = PowerSampler(gpu_index)
    s.start()
    time.sleep(seconds)
    t0 = s.samples[0][0]
    t1 = s.samples[-1][0]
    s.stop()
    _, w = s.energy_joules(t0, t1)
    return w


# --------------------------------------------------------------------------- #
# measurement primitives
# --------------------------------------------------------------------------- #
@torch.no_grad()
def latency_ms(
    fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, iters: int = 100, warmup: int = 20
) -> float:
    """Mean per-batch latency (ms) via CUDA events."""
    for _ in range(warmup):
        fn(x)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(x)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


@torch.no_grad()
def energy_per_img(
    fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    gpu_index: int,
    seconds: float = 4.0,
) -> tuple[float, float]:
    """Return (mJ/img, mean_W) over a fixed wall-time window via NVML."""
    b = x.shape[0]
    for _ in range(20):  # warmup (not sampled)
        fn(x)
    torch.cuda.synchronize()
    sampler = PowerSampler(gpu_index)
    sampler.start()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    n = 0
    while time.perf_counter() - t0 < seconds:
        fn(x)
        n += 1
        if n % 8 == 0:
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    sampler.stop()
    e_j, w_mean = sampler.energy_joules(t0, t1)
    imgs = n * b
    return (1e3 * e_j / imgs if imgs else 0.0), w_mean


@torch.no_grad()
def count_macs(model: nn.Module, first: nn.Conv2d, x: torch.Tensor) -> tuple[int, int]:
    """Return (full_macs, first_conv_macs).

    full_macs is computed via fvcore.nn.FlopCountAnalysis, not a hand-rolled
    Conv2d/Linear hook -- that hook undercounts any architecture using
    attention (misses matmul/bmm entirely) or per-position Linear "convs"
    (misses the H x W multiplier), verified to be off by 10-100x on
    ViT/Swin/ConvNeXtV2/EdgeNeXt (see eval_systems.py). first_conv_macs is a
    single Conv2d layer, computed directly (exact regardless of method).
    """
    from fvcore.nn import FlopCountAnalysis

    analysis = FlopCountAnalysis(model, x)
    analysis.unsupported_ops_warnings(False)
    analysis.uncalled_modules_warnings(False)
    full = int(analysis.total())

    fc = {"v": 0}

    def hook(mod: nn.Module, _inp: object, out: torch.Tensor) -> None:
        oh, ow = out.shape[2], out.shape[3]
        cin = mod.in_channels // mod.groups
        fc["v"] = mod.out_channels * cin * mod.kernel_size[0] * mod.kernel_size[1] * oh * ow

    handle = first.register_forward_hook(hook)
    model(x)
    handle.remove()
    return full, fc["v"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=7)
    ap.add_argument("--dataset", default="cifar100")
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
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 16, 128])
    ap.add_argument("--energy-batch", type=int, default=128)
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument(
        "--skip-energy",
        action="store_true",
        help="skip the NVML energy step (use on a co-tenanted GPU; energy frac ~= latency frac)",
    )
    ap.add_argument("--csv", default=None)
    ap.add_argument("--roofline-csv", default=None)
    args = ap.parse_args()

    from model_utils import find_first_conv_in_normalized, load_normalized_model

    config.set_dataset(args.dataset)
    # --gpu is the PHYSICAL/NVML index (what nvidia-smi shows). torch may see a
    # remapped subset via CUDA_VISIBLE_DEVICES, so resolve the torch index by UUID.
    nvml_idx = args.gpu
    torch_idx = _torch_index_for_phys(nvml_idx)
    dev = f"cuda:{torch_idx}"
    config.DEVICE = dev
    torch.cuda.set_device(torch_idx)
    mean, std = config.IMAGENET_MEAN, config.IMAGENET_STD
    sz = config.IMAGE_SIZE

    p_idle = idle_power(nvml_idx)
    print(f"[idle] phys GPU{nvml_idx} (torch {dev}) floor = {p_idle:.1f} W", flush=True)

    rows: list[dict] = []
    roof: list[dict] = []
    for model in args.models:
        arch = config.MODEL_ARCHS[model]
        wp = config.weight_path(args.dataset, model)
        if not wp.exists():
            print(f"[skip] {model}: no weights")
            continue
        out = args.csv or str(config.ANALYSIS_DIR / f"onsilicon_measured_{args.dataset}.csv")
        rout = args.roofline_csv or str(
            config.ANALYSIS_DIR / f"onsilicon_roofline_{args.dataset}.csv"
        )
        try:
            full = load_normalized_model(
                arch, wp, num_classes=config.NUM_CLASSES, device=dev
            ).eval()
            _, first = find_first_conv_in_normalized(full)
            det = DetectorPath(first, mean, std).to(dev).eval()
            print(f"\n=== {model} ({arch}) | first conv out={first.out_channels} ===", flush=True)

            # analytical MACs (exact, contention-immune)
            x1 = torch.rand(1, 3, sz, sz, device=dev)
            m_full, m_first = count_macs(full, first, x1)
            row: dict = {
                "model": model,
                "arch": arch,
                "macs_full": m_full,
                "macs_first": m_first,
                "mac_ratio_pct": round(100.0 * m_first / m_full, 3),
            }

            # latency sweep (batch 1 is launch-bound -> stable even under co-tenancy)
            for b in args.batches:
                x = torch.rand(b, 3, sz, sz, device=dev)
                lf = latency_ms(full, x) / b
                ld = latency_ms(det, x) / b
                row[f"lat_full_b{b}_ms"] = round(lf, 4)
                row[f"lat_det_b{b}_ms"] = round(ld, 5)
                row[f"lat_ratio_b{b}_pct"] = round(100.0 * ld / lf, 3)
                print(
                    f"   b={b:4d}  full={lf:.4f}  det={ld:.5f} ms/img  ratio={100 * ld / lf:.2f}%",
                    flush=True,
                )
                del x
            torch.cuda.empty_cache()

            if not args.skip_energy:
                # Re-measure the floor right before, so on a co-tenanted GPU the
                # marginal (dynamic) energy subtracts a current baseline.
                xe = torch.rand(args.energy_batch, 3, sz, sz, device=dev)
                p_base = idle_power(nvml_idx, seconds=1.5)
                ef, wf = energy_per_img(full, xe, nvml_idx, args.seconds)
                ed, wd = energy_per_img(det, xe, nvml_idx, args.seconds)
                lf_b = latency_ms(full, xe) / args.energy_batch  # ms/img
                ld_b = latency_ms(det, xe) / args.energy_batch
                dyn_f = max(0.0, (wf - p_base)) * (lf_b * 1e-3) * 1e3  # mJ/img dynamic
                dyn_d = max(0.0, (wd - p_base)) * (ld_b * 1e-3) * 1e3
                row.update(
                    {
                        "pwr_base_W": round(p_base, 1),
                        "energy_full_mJ": round(ef, 4),
                        "pwr_full_W": round(wf, 1),
                        "energy_det_mJ": round(ed, 5),
                        "pwr_det_W": round(wd, 1),
                        "energy_ratio_pct": round(100.0 * ed / ef, 3) if ef else float("nan"),
                        "energy_full_dyn_mJ": round(dyn_f, 4),
                        "energy_det_dyn_mJ": round(dyn_d, 5),
                        "energy_ratio_dyn_pct": (
                            round(100.0 * dyn_d / dyn_f, 3) if dyn_f else float("nan")
                        ),
                    }
                )
                print(
                    f"   energy(b={args.energy_batch}) full={ef:.3f} mJ/img ({wf:.0f} W) | "
                    f"det={ed:.4f} mJ/img ({wd:.0f} W) | ratio={100 * ed / ef:.2f}%",
                    flush=True,
                )
                del xe
                torch.cuda.empty_cache()

            # roofline edge projection (compute-bound lower bound, exact)
            for dname, spec in EDGE_DEVICES.items():
                t_full = 2 * m_full / spec["ops_s"]  # s
                t_det = 2 * m_first / spec["ops_s"]
                roof.append(
                    {
                        "model": model,
                        "device": dname,
                        "full_ms": round(1e3 * t_full, 4),
                        "det_ms": round(1e3 * t_det, 5),
                        "full_mJ": round(1e3 * spec["watt"] * t_full, 4),
                        "det_mJ": round(1e3 * spec["watt"] * t_det, 5),
                        "ratio_pct": round(100.0 * m_first / m_full, 3),
                    }
                )
            rows.append(row)
        except torch.cuda.OutOfMemoryError:
            print(f"   [OOM] {model}: skipped (co-tenant memory pressure)", flush=True)
        finally:
            full = det = None
            torch.cuda.empty_cache()

        # incremental save so a later crash never loses completed models
        pd.DataFrame(rows).to_csv(out, index=False)
        pd.DataFrame(roof).to_csv(rout, index=False)

    df = pd.DataFrame(rows)
    out = args.csv or str(config.ANALYSIS_DIR / f"onsilicon_measured_{args.dataset}.csv")
    rout = args.roofline_csv or str(config.ANALYSIS_DIR / f"onsilicon_roofline_{args.dataset}.csv")
    print("\n=== measured detector vs full (mean across models) ===")
    if len(df):
        cols = [
            c
            for c in df.columns
            if c.startswith("lat_ratio") or c.startswith("energy_ratio") or c == "mac_ratio_pct"
        ]
        print(df[["model", *cols]].to_string(index=False))
        print("\nMEAN ratios:", {c: round(float(df[c].mean()), 2) for c in cols})
    print(f"\n  measured -> {out}\n  roofline -> {rout}")


if __name__ == "__main__":
    main()
