"""Accelerator energy / latency for the first-conv stage (CODES+ISSS codesign angle).

`eval_edge.py` gives CPU latency; this gives the *hardware-codesign* number reviewers
at CODES+ISSS care about: on a real embedded accelerator, how much energy and latency
does Viyog's compute (the first conv) cost relative to the full network? We use the
ZigZag design-space-exploration cost model with two shipped accelerator templates:

  * Edge_TPU_like  — a systolic edge-TPU (the embedded-deployment target),
  * Tesla_NPU_like — an automotive NPU (ties to the paper's autonomous-vehicle motivation).

ZigZag parses an ONNX graph, finds the optimal temporal/spatial mapping per layer, and
returns per-layer energy (pJ) and latency (cycles). We sum the stem (first conv) vs the
whole network -> first-conv energy/latency fraction = an upper bound on Viyog's
accelerator cost as a second-stage detector.

Workloads: ZigZag's reference ONNX (resnet18 — in our model family; mobilenetv2 — an
edge net) are parse-guaranteed; extra ONNX paths can be passed with --onnx.

    python experiments/eval_accelerator.py [--onnx /path/model.onnx ...]
"""

from __future__ import annotations

import argparse
import os
import pathlib

import config
import zigzag
from zigzag.api import get_hardware_performance_zigzag

ZZ = os.path.dirname(zigzag.__file__)
BUNDLED = {
    "resnet18": os.path.join(ZZ, "inputs/examples/workload/resnet18.onnx"),
    "mobilenetv2": os.path.join(ZZ, "inputs/examples/workload/mobilenetv2.onnx"),
}
ACCEL = [  # (display, accelerator module, mapping module)
    (
        "Edge-TPU",
        "zigzag.inputs.examples.hardware.Edge_TPU_like",
        "zigzag.inputs.examples.mapping.edge_tpu_like",
    ),
    (
        "Tesla-NPU",
        "zigzag.inputs.examples.hardware.Tesla_NPU_like",
        "zigzag.inputs.examples.mapping.tesla_npu_like",
    ),
]


def cme_list(rest) -> list:
    """Flatten ZigZag's nested return into a flat list of CostModelEvaluations."""
    out = []

    def walk(x):
        if hasattr(x, "energy_total") and hasattr(x, "layer"):
            out.append(x)
        elif isinstance(x, (list, tuple)):
            for e in x:
                walk(e)

    walk(rest)
    return out


def first_conv_cme(cmes: list):
    """First convolution in topological order (the stem = Viyog's compute)."""
    for c in cmes:
        nm = str(getattr(c.layer, "name", "")).lower()
        ld = getattr(c.layer, "type", "") or getattr(c.layer, "operator_type", "")
        if "conv" in nm or "conv" in str(ld).lower():
            return c
    return cmes[0] if cmes else None


def lat_of(c) -> float:
    """Overall per-layer latency in cycles (prefer total2 = compute+stalls+loading)."""
    for a in ("latency_total2", "latency_total1", "latency_total0"):
        if hasattr(c, a):
            return float(getattr(c, a))
    return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", nargs="*", default=[], help="extra ONNX workload paths")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    import pandas as pd

    workloads = dict(BUNDLED)
    for p in args.onnx:
        workloads[os.path.basename(p).replace(".onnx", "")] = p

    rows = []
    print("=== ZigZag accelerator energy/latency: first-conv (Viyog) vs full network ===\n")
    print(
        f"{'workload':14} {'accel':10} {'E_full(uJ)':>11} {'E_fc(uJ)':>9} {'E_fc%':>7} "
        f"{'L_full(cyc)':>12} {'L_fc%':>7}"
    )
    for wname, wpath in workloads.items():
        if not pathlib.Path(wpath).exists():
            print(f"[skip] {wname}: {wpath} missing")
            continue
        for aname, accel, mapping in ACCEL:
            try:
                en, lat, rest = get_hardware_performance_zigzag(
                    workload=wpath,
                    accelerator=accel,
                    mapping=mapping,
                    opt="latency",
                    dump_filename_pattern="/tmp/zz_out/{}.json",
                    pickle_filename="/tmp/zz_out/cmes.pickle",
                )
                cmes = cme_list(rest)
                fc = first_conv_cme(cmes)
                fc_e, fc_l = fc.energy_total, lat_of(fc)
                row = {
                    "workload": wname,
                    "accelerator": aname,
                    "n_layers": len(cmes),
                    "E_full_uJ": round(en / 1e6, 3),
                    "E_firstconv_uJ": round(fc_e / 1e6, 4),
                    "E_firstconv_%": round(100 * fc_e / en, 3),
                    "L_full_cycles": int(lat),
                    "L_firstconv_cycles": int(fc_l),
                    "L_firstconv_%": round(100 * fc_l / lat, 3),
                    "firstconv_layer": str(getattr(fc.layer, "name", "?")),
                }
                rows.append(row)
                print(
                    f"{wname:14} {aname:10} {row['E_full_uJ']:>11.2f} {row['E_firstconv_uJ']:>9.3f} "
                    f"{row['E_firstconv_%']:>6.3f}% {row['L_full_cycles']:>12} {row['L_firstconv_%']:>6.3f}%",
                    flush=True,
                )
            except Exception as e:
                print(f"[fail] {wname}/{aname}: {type(e).__name__}: {str(e)[:120]}", flush=True)

    df = pd.DataFrame(rows)
    out = args.csv or str(config.ANALYSIS_DIR / "accelerator_energy.csv")
    df.to_csv(out, index=False)
    if len(df):
        print("\n=== means ===")
        print(
            f"  first-conv energy fraction : {df['E_firstconv_%'].mean():.3f}%  "
            f"(={100 / df['E_firstconv_%'].mean():.0f}x less energy than full network)"
        )
        print(f"  first-conv latency fraction: {df['L_firstconv_%'].mean():.3f}%")
    print(f"\n  saved -> {out}")


if __name__ == "__main__":
    main()
