"""export_leaderboard.py --- build the small, self-contained data files the
Viyog demo/leaderboard serves.

It reads the aggregated analysis CSVs already produced by the experiment
pipeline (``results/analysis/...`` and ``results/<ds>/analysis/...``) and writes
a handful of tidy CSVs (a few KB total) into ``demo/data/``. The hosted app
(``app.py``) depends ONLY on those small files --- never on the multi-GB
``data/``/``weights/``/``results/`` trees --- so the Space is free to host and
instant to load.

Run from anywhere::

    uv run --frozen python demo/export_leaderboard.py
    # or, if VIYOG lives elsewhere:
    VIYOG_ROOT=/path/to/viyog uv run --frozen python demo/export_leaderboard.py

Outputs (demo/data/):
    leaderboard.csv     detector x dataset, ranked, with T1/T2/T3 + cost/memory
    permodel_t3.csv     per-architecture OOD-vs-ADV (T3) for every detector
    ood_difficulty.csv  T3 split by OOD difficulty (far/near/texture)
    meta.json           model/detector/dataset inventory + provenance
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("VIYOG_ROOT", "/mnt/data1/asing725/viyog"))
RES = ROOT / "results"
OUT = Path(__file__).resolve().parent / "data"

# dataset -> analysis dir (cifar100 lives at the top, the others under results/<ds>/)
DATASETS = {
    "CIFAR-100": RES / "analysis",
    "CIFAR-10": RES / "cifar10" / "analysis",
    "GTSRB": RES / "gtsrb" / "analysis",
}
_DS_SLUG = {"CIFAR-100": "cifar100", "CIFAR-10": "cifar10", "GTSRB": "gtsrb"}

# friendly detector labels + the family used for colouring
_METHOD_LABEL = {
    "ViyogD_tv_dorm": "Viyog-D (TV)",
    "ViyogD_hf_dorm": "Viyog-HF",
    "Viyog_Linf": "Viyog-L∞ (norm baseline)",
    "GEN": "GEN",
    "Energy": "Energy",
    "MSP": "MSP",
    "MaxLogit": "MaxLogit",
    "Entropy": "Entropy",
    "KLMatching": "KL-Matching",
    "Mahalanobis": "Mahalanobis",
    "KNN": "KNN",
    "ViM": "ViM",
    "ODIN": "ODIN",
}
_SIG_LABEL = {
    "Viyog_D*(tv|p5|adapt)": "Viyog-D (TV)",
    "G_hf_mean": "Viyog-HF",
    "G_tv_mean": "TV (mean)",
    "A_inf_norm": "raw L∞",
}


def _is_viyog(family: str) -> bool:
    return "viyog" in str(family).lower()


def build_leaderboard() -> pd.DataFrame:
    """Concatenate the per-dataset master_comparison tables into one tidy frame."""
    frames = []
    for ds, adir in DATASETS.items():
        f = adir / f"master_comparison_{_DS_SLUG[ds]}.csv"
        if not f.exists():
            print(f"  ! skip {ds}: {f} not found")
            continue
        df = pd.read_csv(f)
        df.insert(0, "dataset", ds)
        df["detector"] = df["method"].map(lambda m: _METHOD_LABEL.get(m, m))
        df["is_viyog"] = df["family"].map(_is_viyog)
        frames.append(df)
        print(f"  + {ds}: {len(df)} detectors")
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["dataset", "T3_OOD_ADV"], ascending=[True, False])
    return out


def build_permodel() -> pd.DataFrame:
    """Per-architecture OOD-vs-ADV (T3, directionless) for every detector."""
    frames = []
    for ds, adir in DATASETS.items():
        f = adir / f"full_eval_{_DS_SLUG[ds]}_permodel.csv"
        if not f.exists():
            print(f"  ! skip permodel {ds}: not found")
            continue
        df = pd.read_csv(f)
        df = df[df["model"] != "model"].copy()           # drop any header-repeat
        df["T3"] = pd.to_numeric(df["T3_dl"], errors="coerce")
        df["detector"] = df["method"].map(lambda m: _METHOD_LABEL.get(m, m))
        df["is_viyog"] = df["method"].str.lower().str.contains("viyog")
        df["dataset"] = ds
        frames.append(df[["dataset", "model", "method", "detector", "is_viyog", "T3"]])
        print(f"  + permodel {ds}: {df['model'].nunique()} models x {df['method'].nunique()} detectors")
    return pd.concat(frames, ignore_index=True).dropna(subset=["T3"])


def build_ood_difficulty() -> pd.DataFrame:
    """T3 split by OOD difficulty (far / near / texture), long form."""
    frames = []
    for ds, adir in DATASETS.items():
        f = adir / f"near_ood_breakdown_{_DS_SLUG[ds]}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        df = df[df["signature"] != "signature"].copy()
        long = df.melt(
            id_vars=["model", "signature"],
            value_vars=["T3_far", "T3_near", "T3_texture"],
            var_name="kind", value_name="T3",
        )
        long["kind"] = long["kind"].str.replace("T3_", "", regex=False).str.title()
        long["detector"] = long["signature"].map(lambda s: _SIG_LABEL.get(s, s))
        long["is_viyog"] = long["detector"].str.lower().str.contains("viyog")
        long["T3"] = pd.to_numeric(long["T3"], errors="coerce")
        long["dataset"] = ds
        frames.append(long.dropna(subset=["T3"]))
        print(f"  + ood-difficulty {ds}: {df['model'].nunique()} models")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("Building leaderboard ...")
    lb = build_leaderboard()
    print("Building per-model T3 ...")
    pm = build_permodel()
    print("Building OOD-difficulty split ...")
    od = build_ood_difficulty()

    lb.to_csv(OUT / "leaderboard.csv", index=False)
    pm.to_csv(OUT / "permodel_t3.csv", index=False)
    od.to_csv(OUT / "ood_difficulty.csv", index=False)

    meta = {
        "datasets": sorted(lb["dataset"].unique().tolist()),
        "n_detectors": int(lb["detector"].nunique()),
        "detectors": sorted(lb["detector"].unique().tolist()),
        "n_models": int(pm["model"].nunique()) if len(pm) else 0,
        "models": sorted(pm["model"].unique().tolist()) if len(pm) else [],
        "headline_metric": "T3 = OOD-vs-ADV separation AUROC (1.0 perfect, 0.5 chance)",
        "source": "results/analysis/*.csv (Viyog CODES+ISSS 2026, paper #215)",
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))

    total_kb = sum((OUT / n).stat().st_size for n in
                   ["leaderboard.csv", "permodel_t3.csv", "ood_difficulty.csv", "meta.json"]) / 1024
    print(f"\nOK -> {OUT}  ({total_kb:.1f} KB total)")
    print(f"   leaderboard rows : {len(lb)}  ({lb['dataset'].nunique()} datasets)")
    print(f"   per-model rows   : {len(pm)}  ({meta['n_models']} models)")
    print(f"   ood-difficulty   : {len(od)} rows")


if __name__ == "__main__":
    main()
