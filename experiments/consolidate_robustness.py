"""Consolidate the robustness pack for the rebuttal:
  (1) cross-dataset generalization of complementarity 3-way (bootstrap CIs over models),
  (2) per-dataset best Viyog vs best logit on T2/T3,
  (3) signature-aware adaptive summary (norm-preserving / single-aware / both-aware),
      per dataset, plus the honest *combined-detector floor while the attack still
      succeeds*: across all-aware rows that retain attack success >= 0.8, the weakest
      the OR-detector max(dorm, HF) ever gets — the number that answers C-w1 / D-d3.
Outputs results/analysis/robustness_pack.csv + prints. CPU.

In-progress adaptive sweeps are skipped: a CSV is only counted if it has the full
mode x lambda grid (>= MIN_ADAPTIVE_ROWS), so a half-written running file can never
leak partial numbers into the pack.
"""
from __future__ import annotations
import glob
import numpy as np
import pandas as pd
import config

AD = config.ANALYSIS_DIR  # cifar100 (flat) — only used as the output location
MIN_ADAPTIVE_ROWS = 25  # full grid: pgd(1) + 4 modes x 6 lambda
ADAPTIVE_DATASETS = ["cifar100", "gtsrb"]


def boot_ci(vals: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean and bootstrap 95% CI of ``vals`` (NaNs already excluded by caller)."""
    vals = np.asarray(vals, float)
    if len(vals) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.RandomState(seed)
    means = [rng.choice(vals, len(vals), replace=True).mean() for _ in range(n)]
    return float(vals.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def complete_adaptive_files(analysis_dir, dataset: str, tag: str) -> list[str]:
    """Adaptive CSVs for ``tag`` on ``dataset`` that hold the full mode x lambda grid.

    Files still being written by a running sweep have fewer rows and are dropped, so
    the pack only ever reports finished sweeps.
    """
    out: list[str] = []
    for f in sorted(glob.glob(str(analysis_dir / f"{tag}_{dataset}_*.csv"))):
        try:
            if len(pd.read_csv(f)) >= MIN_ADAPTIVE_ROWS:
                out.append(f)
        except Exception:
            pass
    return out


def main() -> None:
    rows: list[dict] = []
    # (1)+(2) per dataset — read each dataset's own analysis dir (cifar100=flat, others namespaced)
    for ds in ["cifar100", "gtsrb", "cifar10"]:
        ad_ds = config.dataset_dirs(ds)["analysis"]
        cf = ad_ds / f"complementarity_{ds}.csv"
        ff = ad_ds / f"full_eval_{ds}_summary.csv"
        if not cf.exists():
            continue
        comp = pd.read_csv(cf)
        n_models = comp["model"].nunique()
        for fs in ["Energy only (logit)", "Viyog only", "Energy + Viyog", "Full panel"]:
            sub = comp[comp.feature_set == fs]["bal_acc"]
            m, lo, hi = boot_ci(sub.values)
            rows.append(dict(dataset=ds, n_models=n_models, item=f"3way::{fs}",
                             mean=round(m, 3), ci_lo=round(lo, 3), ci_hi=round(hi, 3)))
        if ff.exists():
            fe = pd.read_csv(ff)
            for task in ["T2", "T3"]:
                vd = fe[(fe.task == task) & (fe.method == "ViyogD_tv_dorm")]["deployable"]
                lg = fe[(fe.task == task) & (fe.method.isin(["Energy", "GEN", "MSP", "MaxLogit"]))]
                vdv = float(vd.iloc[0]) if len(vd) else np.nan
                rows.append(dict(dataset=ds, n_models=n_models, item=f"{task}::Viyog-D_tv_dorm",
                                 mean=round(vdv, 3), ci_lo=np.nan, ci_hi=np.nan))
                rows.append(dict(dataset=ds, n_models=n_models, item=f"{task}::best_logit",
                                 mean=round(float(lg["deployable"].max()), 3), ci_lo=np.nan, ci_hi=np.nan))

    # (3) signature-aware adaptive summary, per dataset. Prefer the strong (n=2000)
    # sweep only once >= 3 of its model files are complete; else use the base sweep.
    for ds in ADAPTIVE_DATASETS:
        ad_ds = config.dataset_dirs(ds)["analysis"]
        strong = complete_adaptive_files(ad_ds, ds, "adaptive_strong")
        base = complete_adaptive_files(ad_ds, ds, "adaptive")
        use_strong = len(strong) >= 3
        files = strong if use_strong else base
        tagname = "strong" if use_strong else "base"
        if not files:
            continue
        agg: dict[str, dict[str, list[float]]] = {}
        combined_floor: list[float] = []  # per model: min over succeeding all-aware rows of max(dorm,hf)
        for f in files:
            d = pd.read_csv(f)
            for mode in ["pgd", "normpresv", "dormaware", "hfaware", "allaware"]:
                sub = d[d["mode"] == mode]
                if len(sub) == 0:
                    continue
                r = sub.iloc[-1]  # max lambda
                agg.setdefault(mode, {"succ": [], "dorm": [], "hf": []})
                agg[mode]["succ"].append(r.attack_success)
                agg[mode]["dorm"].append(r.auroc_dorm)
                agg[mode]["hf"].append(r.auroc_hf)
            aa = d[(d["mode"] == "allaware") & (d["attack_success"] >= 0.8)]
            if len(aa):
                combined_floor.append(float(np.maximum(aa["auroc_dorm"], aa["auroc_hf"]).min()))
        for mode, v in agg.items():
            rows.append(dict(dataset=ds, n_models=len(v["succ"]),
                             item=f"adaptive[{tagname}]::{mode}",
                             mean=round(float(np.mean(v["succ"])), 3),
                             ci_lo=round(float(np.mean(v["dorm"])), 3),
                             ci_hi=round(float(np.mean(v["hf"])), 3)))
        if combined_floor:
            rows.append(dict(dataset=ds, n_models=len(combined_floor),
                             item=f"adaptive[{tagname}]::combined_floor@succ>=0.8",
                             mean=round(float(np.min(combined_floor)), 3),
                             ci_lo=np.nan, ci_hi=np.nan))

    df = pd.DataFrame(rows)
    df.to_csv(AD / "robustness_pack.csv", index=False)
    pd.set_option("display.width", 160)
    print("\n################  ROBUSTNESS PACK  ################\n")
    print("--- (1) Cross-dataset 3-way complementarity (bootstrap 95% CI over models) ---")
    print(df[df.item.str.startswith("3way")].to_string(index=False))
    print("\n--- (2) Per-dataset Viyog vs best logit (deployable AUROC) ---")
    print(df[df.item.str.contains("::Viyog-D_tv_dorm|::best_logit")].to_string(index=False))
    print("\n--- (3) Signature-aware adaptive (succ at max lambda; ci_lo=mean dorm AUROC, "
          "ci_hi=mean hf AUROC; combined_floor=min OR-detector while succ>=0.8) ---")
    print(df[df.item.str.startswith("adaptive")].to_string(index=False))
    print(f"\n  saved -> {AD}/robustness_pack.csv")


if __name__ == "__main__":
    main()
