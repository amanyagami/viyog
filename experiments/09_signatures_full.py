"""Step 9 – Full signature battery over the rich (06b) features.

Recomputes every signature from the one-pass rich feature files:
  * the original 28 (families A–F) via step 08's compute_signatures, and
  * a NEW spatial/frequency family (G) enabled by the rich features:
      total-variation, high-frequency-energy ratio, spatial std, and the
      cross-filter co-activation (Gram off-diagonal) scalar — globally and
      restricted to the dormant filter band.
  * a few strong profile-manifold metrics (H/J) from the exploration.

Outputs results/<dataset>/analysis/signature_auroc_full_<model>.csv and prints
the best signature per task. Directionless AUROC, OOD pooled / ADV pooled, as
in step 08.

Usage:
    python experiments/09_signatures_full.py --dataset cifar100
    python experiments/09_signatures_full.py --dataset cifar10 --models vit_base
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import config
import numpy as np
import pandas as pd
from numpy.linalg import svd

HERE = Path(__file__).resolve().parent

# Reuse step 08's families + helpers (module name starts with a digit → importlib).
_spec = importlib.util.spec_from_file_location("sig08", str(HERE / "08_signatures.py"))
sig08 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sig08)

EPS = 1e-8


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full signature battery (rich features)")
    p.add_argument("--dataset", default="cifar100", choices=list(config.DATASET_SPECS))
    p.add_argument("--models", nargs="+", default=list(config.MODEL_ARCHS),
                   choices=list(config.MODEL_ARCHS), metavar="MODEL")
    return p.parse_args()


def _load(path: Path) -> dict | None:
    import h5py
    if not path.exists():
        return None
    with h5py.File(path, "r") as f:
        d = {k: f[k][:].astype(np.float64) for k in f}
    d["logits"] = d.get("logits")
    return d


def base_and_spatial(d: dict, groups: dict, mu, prec) -> dict[str, np.ndarray]:
    """All signatures for one split: families A–F + new spatial G + manifold."""
    m, Mx, l2 = d["filter_means"], d["filter_maxs"], d["filter_l2"]
    infn = d["inf_norms"]
    logits = d.get("logits")
    sig = sig08.compute_signatures(m, Mx, l2, infn, logits, groups, mu, prec)

    L, H = groups["low"], groups["large"]
    tv, hf, std, gram = d["filter_tv"], d["filter_hf"], d["filter_std"], d["gram_offdiag"]

    def md(x):  # mean over dormant filters
        return x[:, L].mean(1)

    def ml(x):  # mean over large filters
        return x[:, H].mean(1)

    sig.update({
        # ---- G. Spatial / frequency family (NEW, from the rich features) ----
        "G_hf_mean":        hf.mean(1),
        "G_hf_dorm":        md(hf),
        "G_hf_low_large":   md(hf) / (ml(hf) + EPS),
        "G_tv_mean":        tv.mean(1),
        "G_tv_dorm":        md(tv),
        "G_std_dorm":       md(std),
        "G_gram_offdiag":   gram,
        # ---- H/J. strongest manifold metrics from the exploration ----
        "H_dorm_entropy":   _entropy(m[:, L]),
        "J_pca_tail_resid": _tail_resid(m, mu, m_std(m, mu), groups, full=True),
    })
    for k in sig:
        sig[k] = np.nan_to_num(np.asarray(sig[k], dtype=np.float64),
                               nan=0.0, posinf=0.0, neginf=0.0)
    return sig


# --- helpers for the manifold metrics (ID stats are captured via closures) ---
_ID_CACHE: dict = {}


def m_std(m, mu):
    return _ID_CACHE["sd"]


def _entropy(sub):
    p = sub / (sub.sum(1, keepdims=True) + EPS)
    return -(p * np.log(p + EPS)).sum(1)


def _tail_resid(m, mu, sd, groups, full=True):
    Vk = _ID_CACHE["Vk"]
    xs = (m - mu) / sd
    proj = xs @ Vk.T @ Vk
    return np.sqrt(((xs - proj) ** 2).sum(1))


def analyse_model(model: str) -> pd.DataFrame | None:
    fdir = config.FEATURES_DIR
    idd = _load(fdir / f"featfull_{model}_id.h5")
    if idd is None:
        print(f"  [warn] no rich ID features for {model}")
        return None
    m = idd["filter_means"]
    mu, prec = sig08.id_reference(m)
    groups = sig08.neuron_groups(m)
    # cache ID stats for the manifold metrics
    sd = m.std(0) + EPS
    Xs = (m - mu) / sd
    _, S, Vt = svd(Xs, full_matrices=False)
    kpca = int(np.searchsorted(np.cumsum(S ** 2) / (S ** 2).sum(), 0.90) + 1)
    _ID_CACHE.update({"sd": sd, "Vk": Vt[:kpca]})

    print(f"\n  === {model} ===  filters C={m.shape[1]}  "
          f"dormant={len(groups['low'])}  PCA k={kpca}")

    id_sig = base_and_spatial(idd, groups, mu, prec)
    names = list(id_sig.keys())

    ood, adv = {}, {}
    for n in config.OOD_DATASETS:
        sp = _load(fdir / f"featfull_{model}_ood_{n}.h5")
        if sp is not None:
            ood[n] = base_and_spatial(sp, groups, mu, prec)
    for n in config.ATTACKS:
        sp = _load(fdir / f"featfull_{model}_adv_{n}.h5")
        if sp is not None:
            adv[n] = base_and_spatial(sp, groups, mu, prec)

    def pool(d, s):
        return np.concatenate([v[s] for v in d.values()]) if d else np.array([])

    rows = []
    for s in names:
        i, o, a = id_sig[s], pool(ood, s), pool(adv, s)
        a1, _ = sig08.auroc_directionless(i, o) if len(o) else (np.nan, np.nan)
        a2, _ = sig08.auroc_directionless(i, a) if len(a) else (np.nan, np.nan)
        a3, _ = sig08.auroc_directionless(o, a) if len(o) and len(a) else (np.nan, np.nan)
        rows.append({"signature": s, "T1_ID_vs_OOD": a1,
                     "T2_ID_vs_ADV": a2, "T3_OOD_vs_ADV": a3})
    df = pd.DataFrame(rows).set_index("signature")
    out = config.ANALYSIS_DIR / f"signature_auroc_full_{model}.csv"
    df.to_csv(out)
    for task in ["T1_ID_vs_OOD", "T2_ID_vs_ADV", "T3_OOD_vs_ADV"]:
        col = df[task]
        if col.notna().any():
            best = col.idxmax()
            print(f"    best {task:14}: {best:20} {col[best]:.3f}")
        else:
            print(f"    best {task:14}: (all NaN — no valid splits)")
    print(f"    → {out}")
    return df


def main() -> None:
    args = _parse_args()
    config.set_dataset(args.dataset)
    accepted, dropped = config.accepted_models(args.dataset, args.models)
    if dropped:
        print(f"[gate] dropped (clean acc < {config.ACC_FLOOR.get(args.dataset)}%): "
              + ", ".join(f"{m}={a:.1f}" for m, a in dropped))
    print(f"=== Step 9: full signature battery [{args.dataset}] — {len(accepted)} near-SOTA models ===")
    print(f"  OOD ({len(config.OOD_DATASETS)}): {list(config.OOD_DATASETS)}")
    dfs = {m: analyse_model(m) for m in accepted}
    dfs = {m: d for m, d in dfs.items() if d is not None}
    if dfs:
        mean_df = sum(d for d in dfs.values()) / len(dfs)
        print("\n  === mean over models — top 5 per task ===")
        for task in ["T1_ID_vs_OOD", "T2_ID_vs_ADV", "T3_OOD_vs_ADV"]:
            top = mean_df[task].sort_values(ascending=False).head(5)
            print(f"\n  {task}:")
            for name, v in top.items():
                print(f"    {name:22} {v:.3f}")


if __name__ == "__main__":
    main()
