"""Master comparison: Viyog vs ALL pytorch-ood baselines, accuracy + recall + cost.

Merges the already-computed evaluation CSVs into one ranked table that puts every
detector on the same row for the three tasks (T1 ID-vs-OOD, T2 ID-vs-ADV, T3 OOD-vs-ADV,
the headline stage-2 routing) alongside its *deployment cost* — detector-state memory,
compute (% of a full forward), CPU latency (% of full), and accelerator energy (% of
full). This is the single table reviewers asked for: "compare with all pytorch-ood
baselines on compute, memory, time and AUROC/recall".

Sources (all pre-computed, CPU only, no re-extract):
  - full_eval_<ds>_summary.csv     held-out, calibration-oriented T1/T2/T3 + operating metrics
  - baselines_feature_<ds>.csv     Mahalanobis / KNN / ViM T1/T2/T3 (per model -> averaged)
  - COST (below)                   per-family deployment cost from the edge measurements

    python experiments/exp_master_table.py --dataset cifar100
"""

from __future__ import annotations

import argparse

import config
import numpy as np
import pandas as pd
from eval_protocol import PROTOCOL_NAME

# Per-family deployment cost. Viyog runs ONLY the first conv; logit + distance baselines
# need a full forward (+ their stored state). Numbers from the edge measurements:
#   detector_cost.csv (state mem), detector_cost_compute.csv (2.284% MACs --
#   corrected 2026-07-26: detector_cost_compute.csv's full-model MAC count was
#   a hand-rolled Conv2d/Linear-only hook that missed attention entirely and
#   undercounted convnextv2_base's per-position Linear "convs"; recomputed via
#   fvcore.nn.FlopCountAnalysis, see eval_detector_cost.py full_macs()),
#   edge_latency.csv (3.5% CPU), accelerator_energy.csv (5.1% energy).
COST = {
    # method:        (state_mem_KB, compute_%fwd, cpu_lat_%fwd, accel_energy_%fwd)
    "ViyogD_tv_dorm": (0.30, 2.284, 3.5, 5.1),
    "ViyogD_hf_dorm": (0.30, 2.284, 3.5, 5.1),
    "Viyog_Linf": (0.01, 2.284, 3.5, 5.1),
    "Energy": (0.004, 100.0, 100.0, 100.0),
    "Entropy": (0.004, 100.0, 100.0, 100.0),
    "GEN": (0.004, 100.0, 100.0, 100.0),
    "KLMatching": (0.05, 100.0, 100.0, 100.0),
    "MSP": (0.004, 100.0, 100.0, 100.0),
    "MaxLogit": (0.004, 100.0, 100.0, 100.0),
    "Mahalanobis": (7588.0, 100.0, 100.0, 100.0),
    "KNN": (25600.0, 100.0, 100.0, 100.0),
    "ViM": (7300.0, 100.0, 100.0, 100.0),
}
FAMILY = {
    "ViyogD_tv_dorm": "Viyog (first-conv)",
    "ViyogD_hf_dorm": "Viyog (first-conv)",
    "Viyog_Linf": "Viyog (first-conv)",
    "Energy": "logit",
    "Entropy": "logit",
    "GEN": "logit",
    "KLMatching": "logit",
    "MSP": "logit",
    "MaxLogit": "logit",
    "Mahalanobis": "distance (feature)",
    "KNN": "distance (feature)",
    "ViM": "distance (feature)",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar100", choices=list(config.DATASET_SPECS))
    args = ap.parse_args()
    config.set_dataset(args.dataset)
    A = config.ANALYSIS_DIR  # per-dataset path — must be read AFTER set_dataset

    fe = pd.read_csv(A / f"full_eval_{args.dataset}_summary.csv")
    required = {"protocol", "heldout_protocol", "mean_oriented", "mean_tpr_at_cal_fpr"}
    missing = required.difference(fe.columns)
    if missing or not (fe["protocol"] == PROTOCOL_NAME).all():
        raise RuntimeError(
            "full_eval summary predates the held-out protocol; rerun experiments/full_eval.py "
            f"(missing columns: {sorted(missing)})"
        )
    # full_eval is long (one row per method x task); all metrics are held-out.
    piv = fe.pivot_table(index="method", columns="task", values="mean_oriented")
    recall = fe.pivot_table(index="method", columns="task", values="mean_tpr_at_cal_fpr")
    dep = fe.groupby("method")["heldout_protocol"].all()
    dc = fe.groupby("method")["dir_consistent"].all()

    rows = []
    for m in piv.index:
        rows.append(
            {
                "method": m,
                "T1_ID_OOD": round(piv.loc[m].get("T1", np.nan), 3),
                "T2_ID_ADV": round(piv.loc[m].get("T2", np.nan), 3),
                "T3_OOD_ADV": round(piv.loc[m].get("T3", np.nan), 3),
                "recall@5%FPR_T3": round(recall.loc[m].get("T3", np.nan), 3),
                "deployable": bool(dep.get(m, False)),
                "dir_consistent": bool(dc.get(m, False)),
            }
        )

    # Distance baselines must use the same held-out direction/threshold protocol.
    bf_path = A / f"baselines_feature_{args.dataset}.csv"
    if bf_path.exists():
        bf = pd.read_csv(bf_path)
        bf_required = {
            "protocol",
            "T1_oriented",
            "T2_oriented",
            "T3_oriented",
            "T3_recall_at_cal5fpr",
            "T1_sign",
            "T2_sign",
            "T3_sign",
        }
        bf_missing = bf_required.difference(bf.columns)
        if bf_missing or not (bf["protocol"] == PROTOCOL_NAME).all():
            raise RuntimeError(
                "distance-baseline CSV predates the held-out protocol; rerun "
                f"experiments/baselines_feature.py (missing columns: {sorted(bf_missing)})"
            )
        bf_consistent = (
            bf.groupby("method")[["T1_sign", "T2_sign", "T3_sign"]]
            .nunique(dropna=False)
            .le(1)
            .all(axis=1)
        )
        g = (
            bf.groupby("method")
            .agg(
                T1=("T1_oriented", "mean"),
                T2=("T2_oriented", "mean"),
                T3=("T3_oriented", "mean"),
                recall=("T3_recall_at_cal5fpr", "mean"),
            )
            .round(3)
        )
        for m in g.index:
            rows.append(
                {
                    "method": m,
                    "T1_ID_OOD": g.loc[m, "T1"],
                    "T2_ID_ADV": g.loc[m, "T2"],
                    "T3_OOD_ADV": g.loc[m, "T3"],
                    "recall@5%FPR_T3": g.loc[m, "recall"],
                    "deployable": True,
                    "dir_consistent": bool(bf_consistent.get(m, False)),
                }
            )

    df = pd.DataFrame(rows)
    # attach cost
    cost = pd.DataFrame.from_dict(
        COST,
        orient="index",
        columns=["state_mem_KB", "compute_%fwd", "cpu_lat_%fwd", "accel_energy_%fwd"],
    )
    df = df.merge(cost, left_on="method", right_index=True, how="left")
    df["family"] = df["method"].map(FAMILY).fillna("other")
    # Viyog memory advantage vs the median full-forward baseline
    df = df.sort_values("T3_OOD_ADV", ascending=False)
    order = [
        "method",
        "family",
        "T1_ID_OOD",
        "T2_ID_ADV",
        "T3_OOD_ADV",
        "recall@5%FPR_T3",
        "deployable",
        "dir_consistent",
        "state_mem_KB",
        "compute_%fwd",
        "cpu_lat_%fwd",
        "accel_energy_%fwd",
    ]
    df = df[order]

    out_csv = str(A / f"master_comparison_{args.dataset}.csv")
    df.to_csv(out_csv, index=False)

    # markdown
    md = [
        f"# Master comparison — Viyog vs all pytorch-ood baselines ({args.dataset})",
        "",
        "All AUROCs use score directions fixed on disjoint calibration data and are evaluated "
        "on held-out test data. Recall uses a calibration-set 5% FPR threshold. Cost columns: "
        "detector-state memory (KB), and compute / CPU-latency / accelerator-energy as **% of "
        "a full forward** (Viyog = first-conv only).",
        "",
        "| Method | Family | T1 ID-OOD | T2 ID-ADV | T3 OOD-ADV | rec@5%FPR | Deploy | DirCons | State (KB) | Compute % | CPU-lat % | Energy % |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():

        def f(x, p=3):
            return "—" if pd.isna(x) else (f"{x:.{p}f}" if isinstance(x, float) else str(x))

        md.append(
            f"| {r.method} | {r.family} | {f(r.T1_ID_OOD)} | {f(r.T2_ID_ADV)} | "
            f"**{f(r.T3_OOD_ADV)}** | {f(r['recall@5%FPR_T3'])} | {'✓' if r.deployable else '✗'} | "
            f"{'✓' if r.dir_consistent else '✗'} | {f(r.state_mem_KB, 2)} | {f(r['compute_%fwd'], 1)} | "
            f"{f(r['cpu_lat_%fwd'], 1)} | {f(r['accel_energy_%fwd'], 1)} |"
        )
    # headline ratios
    viyog_mem = COST["ViyogD_tv_dorm"][0]
    md += [
        "",
        "## Headline",
        f"- **T2 (ID-vs-ADV)** — Viyog-tv {df.set_index('method').loc['ViyogD_tv_dorm', 'T2_ID_ADV']:.3f} "
        f"vs best logit {df[df.family == 'logit']['T2_ID_ADV'].max():.3f}.",
        f"- **T3 (OOD-vs-ADV)** — Viyog-tv {df.set_index('method').loc['ViyogD_tv_dorm', 'T3_OOD_ADV']:.3f}; "
        f"best logit {df[df.family == 'logit']['T3_OOD_ADV'].max():.3f}; distance ViM "
        f"{df.set_index('method').loc['ViM', 'T3_OOD_ADV'] if 'ViM' in df.method.values else float('nan'):.3f} "
        f"at {COST['ViM'][0] / 1024:.1f} MB state.",
        f"- **State memory** — Viyog {viyog_mem} KB vs Mahalanobis {COST['Mahalanobis'][0] / 1024:.1f} MB "
        f"({COST['Mahalanobis'][0] / viyog_mem:,.0f}×), KNN {COST['KNN'][0] / 1024:.1f} MB "
        f"({COST['KNN'][0] / viyog_mem:,.0f}×), ViM {COST['ViM'][0] / 1024:.1f} MB.",
        f"- **Compute / latency / energy** — Viyog first-conv "
        f"{COST['ViyogD_tv_dorm'][1]}% / {COST['ViyogD_tv_dorm'][2]}% / {COST['ViyogD_tv_dorm'][3]}% "
        f"of a full forward; every logit & distance baseline pays 100% (full forward) on top of its state.",
    ]
    out_md = str(A / f"MASTER_COMPARISON_{args.dataset}.md")  # alongside the CSV, not root
    with open(out_md, "w") as fh:
        fh.write("\n".join(md) + "\n")

    print(df.to_string(index=False))
    print(f"\n  saved → {out_csv}\n  saved → {out_md}")


if __name__ == "__main__":
    main()
