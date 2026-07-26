"""Step 8 – Statistical signature battery for ID / OOD / ADV separation.

Consumes the per-sample first-layer statistics + logits written by step 06
(filter_means m, filter_maxs M, filter_l2, inf_norms, logits) and computes a
battery of ~28 scalar signatures per sample, grouped into families:

  A. Norm family            – L1/L2/L∞/energy of the per-filter profile.
  B. Group-ratio family     – low/large, mid/large, energy fractions
                              (operationalises "ADV wakes up silent filters").
  C. Sparsity family        – participation ratio, Gini, Hoyer, spectral entropy.
  D. Crest family           – peak-to-mean / spikiness (high-freq ADV outliers).
  E. Profile-deviation      – first-layer Mahalanobis / cosine / L1 drift vs ID.
  F. Logit/energy family    – energy score (Liu 2020), MSP, max-logit, margin,
                              softmax entropy, KL-to-uniform.

For every signature it reports separability (AUROC, FPR@95TPR) for three tasks:
  T1  ID  vs OOD     T2  ID vs ADV     T3  OOD vs ADV   (the headline)

then fits LDA + multinomial logistic regression as a 3-way ID/OOD/ADV detector
and reports the confusion matrix and the most discriminative signatures.

Outputs:
  results/analysis/signatures.json          – all AUROC tables + classifier metrics
  results/analysis/signature_auroc_<model>.csv
  results/plots/<model>_roc_panel.png
  results/plots/<model>_signature_heatmap.png
  results/plots/<model>_separability_bar.png
  results/plots/<model>_ecdf_best.png
  results/plots/<model>_confusion.png
  results/plots/_summary_t3_auroc.png        – cross-model headline heatmap

Pure CPU; runs in seconds. No GPU required.

Run:
    uv run python experiments/08_signatures.py
"""

from __future__ import annotations

import argparse
import json
import warnings

import config
import matplotlib
import numpy as np
import pandas as pd
from config import (
    ANALYSIS_DIR,
    ATTACKS,
    FEATURES_DIR,
    MODELS,
    NEURON_LARGE_PCT,
    NEURON_LOW_PCT,
    OOD_DATASETS,
    PLOTS_DIR,
)
from data_utils import load_feature_h5
from scipy.special import logsumexp, softmax
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

warnings.filterwarnings("ignore")
_EPS = 1e-8

# Near/far/texture grouping for per-kind breakdown.
OOD_KIND = {name: meta["kind"] for name, meta in OOD_DATASETS.items()}


# ---------------------------------------------------------------------------
# Neuron groups + ID reference statistics
# ---------------------------------------------------------------------------

def neuron_groups(id_m: np.ndarray) -> dict[str, np.ndarray]:
    """Rank filters by mean ID activation → large / low / mid index sets.

    The dormant ("low") band is taken among ALIVE channels only. Some first-conv
    channels are permanently dead (mean==0, e.g. 25/64 in densenet121); selecting
    the global lowest-k would pick those dead channels, making every dormant-band
    signature all-zero and the AUROC a spurious 0.5. Restricting to active channels
    keeps the dormant statistic meaningful for every architecture.
    """
    C = id_m.shape[1]
    per_filter = id_m.mean(axis=0)
    order = np.argsort(per_filter)[::-1]                       # high → low (all)
    alive = np.where(per_filter > 1e-4)[0]
    if len(alive) == 0:
        alive = np.arange(C)
    aorder = alive[np.argsort(per_filter[alive])[::-1]]        # high → low (alive)
    n_large = max(1, int(C * NEURON_LARGE_PCT))
    n_low = max(1, int(len(alive) * NEURON_LOW_PCT))
    large = order[:n_large]
    low = aorder[-n_low:]
    excl = set(large.tolist()) | set(low.tolist())
    mid = np.array([c for c in order if c not in excl], dtype=order.dtype)
    return {"large": large, "low": low, "mid": mid}


def id_reference(id_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean profile and (shrinkage-regularised) precision matrix for Mahalanobis."""
    mu = id_m.mean(axis=0)
    cov = np.cov(id_m, rowvar=False)
    cov = np.atleast_2d(cov)
    C = cov.shape[0]
    ridge = 1e-3 * (np.trace(cov) / max(C, 1)) + 1e-6
    prec = np.linalg.pinv(cov + ridge * np.eye(C))
    return mu, prec


# ---------------------------------------------------------------------------
# Signature computation (all vectorised over N samples)
# ---------------------------------------------------------------------------

def _gini(x: np.ndarray) -> np.ndarray:
    """Row-wise Gini coefficient of a non-negative matrix (N, C)."""
    x = np.sort(np.clip(x, 0, None), axis=1)
    n = x.shape[1]
    idx = np.arange(1, n + 1)
    denom = n * x.sum(axis=1) + _EPS
    return (2.0 * (idx * x).sum(axis=1)) / denom - (n + 1.0) / n


def compute_signatures(
    m: np.ndarray,        # (N, C) per-filter mean |act|
    Mx: np.ndarray,       # (N, C) per-filter max |act|
    l2: np.ndarray,       # (N, C) per-filter RMS act
    infn: np.ndarray,     # (N,)   sample inf-norm
    logits: np.ndarray | None,  # (N, K) or None
    groups: dict[str, np.ndarray],
    mu: np.ndarray,       # (C,) ID mean profile
    prec: np.ndarray,     # (C, C) ID precision
) -> dict[str, np.ndarray]:
    """Return {signature_name: (N,) float array} for one split."""
    N, C = m.shape
    L, H, D = groups["low"], groups["large"], groups["mid"]
    sig: dict[str, np.ndarray] = {}

    # ---- A. Norm family ----
    l1 = m.sum(axis=1)
    l2_prof = np.sqrt((m ** 2).sum(axis=1))
    sig["A_l1"] = l1
    sig["A_l2"] = l2_prof
    sig["A_linf_mean"] = m.max(axis=1)
    sig["A_mean"] = m.mean(axis=1)
    sig["A_energy"] = (l2 ** 2).sum(axis=1)          # total activation power
    sig["A_inf_norm"] = infn                          # Viyog statistic
    sig["A_peak_l2"] = np.sqrt((Mx ** 2).sum(axis=1))

    # ---- B. Group-ratio family ----
    mean_large = m[:, H].mean(axis=1)
    mean_low = m[:, L].mean(axis=1)
    mean_mid = m[:, D].mean(axis=1)
    sig["B_ratio_low_large"] = mean_low / (mean_large + _EPS)
    sig["B_ratio_mid_large"] = mean_mid / (mean_large + _EPS)
    sig["B_large_frac"] = m[:, H].sum(axis=1) / (l1 + _EPS)
    sig["B_low_frac"] = m[:, L].sum(axis=1) / (l1 + _EPS)

    # ---- C. Sparsity family ----
    sig["C_participation"] = l1 ** 2 / ((m ** 2).sum(axis=1) + _EPS)
    sig["C_gini"] = _gini(m)
    sig["C_hoyer"] = (np.sqrt(C) - l1 / (l2_prof + _EPS)) / (np.sqrt(C) - 1 + _EPS)
    p = m / (l1[:, None] + _EPS)
    sig["C_spectral_entropy"] = -(p * np.log(p + _EPS)).sum(axis=1)
    k = max(1, int(0.1 * C))
    topk = np.sort(m, axis=1)[:, -k:].sum(axis=1)
    sig["C_topk_frac"] = topk / (l1 + _EPS)

    # ---- D. Crest family ----
    sig["D_crest_mean"] = (Mx / (m + _EPS)).mean(axis=1)
    sig["D_spikiness"] = infn / (m.mean(axis=1) + _EPS)
    sig["D_max_to_l2"] = Mx.max(axis=1) / (l2_prof + _EPS)

    # ---- E. Profile-deviation family ----
    d = m - mu[None, :]
    sig["E_mahalanobis"] = np.einsum("ni,ij,nj->n", d, prec, d)
    sig["E_cos_id"] = (m @ mu) / (np.linalg.norm(m, axis=1) * np.linalg.norm(mu) + _EPS)
    sig["E_l1_drift"] = np.abs(d).sum(axis=1)

    # ---- F. Logit / energy family ----
    if logits is not None:
        z = logits.astype(np.float32)
        K = z.shape[1]
        sig["F_energy"] = -logsumexp(z, axis=1)           # ID low, OOD high (neg here)
        sm = softmax(z, axis=1)
        sig["F_msp"] = sm.max(axis=1)
        sig["F_max_logit"] = z.max(axis=1)
        zs = np.sort(z, axis=1)
        sig["F_margin"] = zs[:, -1] - zs[:, -2]
        ent = -(sm * np.log(sm + _EPS)).sum(axis=1)
        sig["F_softmax_entropy"] = ent
        sig["F_kl_uniform"] = np.log(K) - ent

    # sanitise
    for kk in sig:
        sig[kk] = np.nan_to_num(sig[kk].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    return sig


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def auroc_directionless(scores0: np.ndarray, scores1: np.ndarray) -> tuple[float, float]:
    """AUROC for class1-vs-class0; returns (auroc, fpr@95tpr).

    AUROC is reported in [0,1]; separability = 2*|auroc-0.5|. Direction is
    folded so the returned AUROC is always >= 0.5 (best orientation).
    """
    y = np.r_[np.zeros(len(scores0)), np.ones(len(scores1))]
    s = np.r_[scores0, scores1]
    if len(np.unique(s)) < 2:
        return 0.5, 1.0
    a = roc_auc_score(y, s)
    flip = a < 0.5
    if flip:
        a = 1.0 - a
        s = -s
    fpr, tpr, _ = roc_curve(y, s)
    idx = np.searchsorted(tpr, 0.95)
    fpr95 = float(fpr[min(idx, len(fpr) - 1)])
    return float(a), fpr95


# ---------------------------------------------------------------------------
# Per-model analysis
# ---------------------------------------------------------------------------

def _load_split(model: str, split_file: str) -> dict | None:
    p = FEATURES_DIR / split_file
    if not p.exists():
        return None
    d = load_feature_h5(p)
    return {
        "m": d["filter_means"].astype(np.float32),
        "Mx": d["filter_maxs"].astype(np.float32),
        "l2": d.get("filter_l2", d["filter_means"]).astype(np.float32),
        "infn": d["inf_norms"].astype(np.float32),
        "logits": d["logits"].astype(np.float32) if "logits" in d else None,
    }


def analyse_model(model: str) -> dict:
    print(f"\n  === {model} ===")
    idd = _load_split(model, f"feat_{model}_id.h5")
    if idd is None:
        print(f"  [warn] no ID features for {model}; skipping")
        return {"model": model, "skipped": True}

    groups = neuron_groups(idd["m"])
    mu, prec = id_reference(idd["m"])
    has_logits = idd["logits"] is not None
    C = idd["m"].shape[1]
    print(f"  filters C={C}  logits={'yes' if has_logits else 'NO'}  "
          f"large={len(groups['large'])} low={len(groups['low'])}")

    def sigs(split: dict) -> dict[str, np.ndarray]:
        return compute_signatures(
            split["m"], split["Mx"], split["l2"], split["infn"],
            split["logits"] if has_logits else None, groups, mu, prec,
        )

    id_sig = sigs(idd)
    sig_names = list(id_sig.keys())

    # ---- collect OOD / ADV per-split signatures ----
    ood_sig: dict[str, dict[str, np.ndarray]] = {}
    for name in OOD_DATASETS:
        sp = _load_split(model, f"feat_{model}_ood_{name}.h5")
        if sp is not None:
            ood_sig[name] = sigs(sp)
    adv_sig: dict[str, dict[str, np.ndarray]] = {}
    for name in ATTACKS:
        sp = _load_split(model, f"feat_{model}_adv_{name}.h5")
        if sp is not None:
            adv_sig[name] = sigs(sp)

    if not ood_sig or not adv_sig:
        print(f"  [warn] missing OOD ({len(ood_sig)}) or ADV ({len(adv_sig)}) splits")

    # pooled
    def pool(d: dict[str, dict[str, np.ndarray]], s: str) -> np.ndarray:
        return np.concatenate([v[s] for v in d.values()]) if d else np.array([])

    # ---- AUROC table: signature × {T1,T2,T3} ----
    rows = []
    for s in sig_names:
        id_s = id_sig[s]
        ood_s = pool(ood_sig, s)
        adv_s = pool(adv_sig, s)
        a1, f1 = auroc_directionless(id_s, ood_s) if len(ood_s) else (np.nan, np.nan)
        a2, f2 = auroc_directionless(id_s, adv_s) if len(adv_s) else (np.nan, np.nan)
        a3, f3 = (auroc_directionless(ood_s, adv_s)
                  if len(ood_s) and len(adv_s) else (np.nan, np.nan))
        rows.append({
            "signature": s,
            "T1_ID_vs_OOD": a1, "T1_fpr95": f1,
            "T2_ID_vs_ADV": a2, "T2_fpr95": f2,
            "T3_OOD_vs_ADV": a3, "T3_fpr95": f3,
        })
    table = pd.DataFrame(rows).set_index("signature")
    table.to_csv(ANALYSIS_DIR / f"signature_auroc_{model}.csv")

    # ---- per-attack / per-OOD AUROC for the headline task, best signature ----
    best_t3 = table["T3_OOD_vs_ADV"].idxmax()
    per_attack_t3 = {}
    ood_pool_best = pool(ood_sig, best_t3)
    for atk, v in adv_sig.items():
        per_attack_t3[atk], _ = auroc_directionless(ood_pool_best, v[best_t3])
    per_ood_kind = {}
    for kind in {"near_ood", "far_ood", "texture_ood"}:
        members = [n for n in ood_sig if OOD_KIND.get(n) == kind]
        if members:
            ks = np.concatenate([ood_sig[n][best_t3] for n in members])
            adv_pool_best = pool(adv_sig, best_t3)
            per_ood_kind[kind], _ = auroc_directionless(ks, adv_pool_best)

    # ---- 3-way classifier (LDA + LR) ----
    classifier = train_3way(id_sig, ood_sig, adv_sig, sig_names, model)

    # ---- plots ----
    try:
        plot_roc_panel(model, id_sig, ood_sig, adv_sig, table)
        plot_signature_heatmap(model, table)
        plot_separability_bar(model, table)
        plot_ecdf_best(model, id_sig, ood_sig, adv_sig, best_t3)
        if classifier.get("confusion") is not None:
            plot_confusion(model, classifier["confusion"])
    except Exception as e:
        print(f"  [warn] plotting failed: {e}")

    print(f"  best OOD↔ADV signature: {best_t3}  AUROC={table.loc[best_t3, 'T3_OOD_vs_ADV']:.3f}")
    print(f"  3-way detector acc: LDA={classifier['lda_acc']:.3f} LR={classifier['lr_acc']:.3f}")

    return {
        "model": model,
        "n_filters": int(C),
        "has_logits": bool(has_logits),
        "auroc_table": table.round(4).reset_index().to_dict(orient="records"),
        "best_T3_signature": best_t3,
        "best_T3_auroc": float(table.loc[best_t3, "T3_OOD_vs_ADV"]),
        "per_attack_T3_auroc": {k: float(v) for k, v in per_attack_t3.items()},
        "per_ood_kind_T3_auroc": {k: float(v) for k, v in per_ood_kind.items()},
        "classifier": {k: v for k, v in classifier.items() if k != "confusion"},
        "confusion_matrix": (classifier["confusion"].tolist()
                             if classifier.get("confusion") is not None else None),
        "top5_T1": table["T1_ID_vs_OOD"].nlargest(5).round(4).to_dict(),
        "top5_T2": table["T2_ID_vs_ADV"].nlargest(5).round(4).to_dict(),
        "top5_T3": table["T3_OOD_vs_ADV"].nlargest(5).round(4).to_dict(),
    }


def train_3way(
    id_sig: dict[str, np.ndarray],
    ood_sig: dict[str, dict[str, np.ndarray]],
    adv_sig: dict[str, dict[str, np.ndarray]],
    sig_names: list[str],
    model: str,
) -> dict:
    """Fit LDA + multinomial LR as a 3-way ID/OOD/ADV detector."""
    def stack(sig_dict: dict[str, np.ndarray]) -> np.ndarray:
        return np.column_stack([sig_dict[s] for s in sig_names])

    X_id = stack(id_sig)
    X_ood = np.vstack([stack(v) for v in ood_sig.values()]) if ood_sig else np.empty((0, len(sig_names)))
    X_adv = np.vstack([stack(v) for v in adv_sig.values()]) if adv_sig else np.empty((0, len(sig_names)))
    if len(X_ood) == 0 or len(X_adv) == 0:
        return {"lda_acc": float("nan"), "lr_acc": float("nan"), "confusion": None}

    X = np.vstack([X_id, X_ood, X_adv])
    y = np.r_[np.zeros(len(X_id)), np.ones(len(X_ood)), np.full(len(X_adv), 2)]
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)

    lda = LinearDiscriminantAnalysis().fit(Xtr, ytr)
    lr = LogisticRegression(max_iter=2000, C=1.0, multi_class="multinomial").fit(Xtr, ytr)
    lda_acc = float(lda.score(Xte, yte))
    lr_acc = float(lr.score(Xte, yte))
    cm = confusion_matrix(yte, lr.predict(Xte), labels=[0, 1, 2])

    # feature importance = mean |LR coef| across the 3 one-vs-rest rows
    importance = np.abs(lr.coef_).mean(axis=0)
    top_feats = sorted(zip(sig_names, importance), key=lambda t: -t[1])[:8]

    return {
        "lda_acc": lda_acc,
        "lr_acc": lr_acc,
        "confusion": cm,
        "top_features": [(n, float(w)) for n, w in top_feats],
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

_C_ID, _C_OOD, _C_ADV = "#27ae60", "#2980b9", "#c0392b"


def _roc_xy(neg: np.ndarray, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    y = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
    s = np.r_[neg, pos]
    a = roc_auc_score(y, s)
    if a < 0.5:
        a, s = 1 - a, -s
    fpr, tpr, _ = roc_curve(y, s)
    return fpr, tpr, a


def plot_roc_panel(model, id_sig, ood_sig, adv_sig, table) -> None:
    def pool(d, s):
        return np.concatenate([v[s] for v in d.values()])

    tasks = [
        ("T1_ID_vs_OOD", "ID vs OOD", id_sig, lambda s: pool(ood_sig, s)),
        ("T2_ID_vs_ADV", "ID vs ADV", id_sig, lambda s: pool(adv_sig, s)),
        ("T3_OOD_vs_ADV", "OOD vs ADV", None, None),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (col, title, negd, posf) in zip(axes, tasks):
        top3 = table[col].nlargest(3).index.tolist()
        for s in top3:
            if col == "T3_OOD_vs_ADV":
                neg, pos = pool(ood_sig, s), pool(adv_sig, s)
            else:
                neg, pos = negd[s], posf(s)
            fpr, tpr, a = _roc_xy(neg, pos)
            ax.plot(fpr, tpr, lw=1.6, label=f"{s} ({a:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=0.7, alpha=0.5)
        ax.set_title(f"{title}", fontsize=10)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.legend(fontsize=6, loc="lower right")
    fig.suptitle(f"{model}: ROC — top-3 signatures per task", fontsize=11)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"{model}_roc_panel.png", dpi=150)
    plt.close(fig)


def plot_signature_heatmap(model, table) -> None:
    cols = ["T1_ID_vs_OOD", "T2_ID_vs_ADV", "T3_OOD_vs_ADV"]
    data = table[cols].sort_values("T3_OOD_vs_ADV", ascending=False)
    fig, ax = plt.subplots(figsize=(6, max(6, 0.32 * len(data))))
    sns.heatmap(data, annot=True, fmt=".2f", cmap="viridis", vmin=0.5, vmax=1.0,
                cbar_kws={"label": "AUROC"}, ax=ax)
    ax.set_title(f"{model}: signature separability (AUROC)", fontsize=10)
    ax.set_xticklabels(["ID/OOD", "ID/ADV", "OOD/ADV"], rotation=0)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"{model}_signature_heatmap.png", dpi=150)
    plt.close(fig)


def plot_separability_bar(model, table) -> None:
    top = table["T3_OOD_vs_ADV"].nlargest(12)[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(range(len(top)), top.values, color="#8e44ad")
    ax.set_yticks(range(len(top))); ax.set_yticklabels(top.index, fontsize=8)
    ax.axvline(0.5, color="k", ls="--", lw=0.8)
    ax.set_xlim(0.5, 1.0); ax.set_xlabel("AUROC (OOD vs ADV)")
    ax.set_title(f"{model}: top signatures for OOD↔ADV separation", fontsize=10)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"{model}_separability_bar.png", dpi=150)
    plt.close(fig)


def plot_ecdf_best(model, id_sig, ood_sig, adv_sig, best) -> None:
    def ecdf(x):
        xs = np.sort(x)
        return xs, np.arange(1, len(xs) + 1) / len(xs)

    ood_all = np.concatenate([v[best] for v in ood_sig.values()])
    adv_all = np.concatenate([v[best] for v in adv_sig.values()])
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for x, c, lab in [(id_sig[best], _C_ID, "ID"), (ood_all, _C_OOD, "OOD"),
                      (adv_all, _C_ADV, "ADV")]:
        xs, ys = ecdf(x)
        ax.plot(xs, ys, color=c, lw=2, label=lab)
    ax.set_xlabel(best); ax.set_ylabel("ECDF"); ax.legend()
    ax.set_title(f"{model}: ECDF of best OOD↔ADV signature ({best})", fontsize=10)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"{model}_ecdf_best.png", dpi=150)
    plt.close(fig)


def plot_confusion(model, cm) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    sns.heatmap(cmn, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
                xticklabels=["ID", "OOD", "ADV"], yticklabels=["ID", "OOD", "ADV"], ax=ax)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"{model}: 3-way detector (row-normalised)", fontsize=10)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"{model}_confusion.png", dpi=150)
    plt.close(fig)


def plot_summary(all_results: list[dict]) -> None:
    rows = [r for r in all_results if not r.get("skipped")]
    if not rows:
        return
    models = [r["model"] for r in rows]
    data = pd.DataFrame({
        "best OOD↔ADV": [r["best_T3_auroc"] for r in rows],
        "3way LR acc": [r["classifier"].get("lr_acc", np.nan) for r in rows],
        "3way LDA acc": [r["classifier"].get("lda_acc", np.nan) for r in rows],
    }, index=models)
    fig, ax = plt.subplots(figsize=(6, max(3, 0.6 * len(models))))
    sns.heatmap(data, annot=True, fmt=".3f", cmap="magma", vmin=0.5, vmax=1.0, ax=ax)
    ax.set_title("Cross-model: headline separability + 3-way detector", fontsize=10)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "_summary_t3_auroc.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Signature battery")
    p.add_argument("--dataset", default="cifar100", choices=list(config.DATASET_SPECS),
                   help="ID dataset to analyse (default: cifar100)")
    p.add_argument("--models", nargs="+", default=list(config.MODEL_ARCHS),
                   choices=list(config.MODEL_ARCHS), metavar="MODEL")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    config.set_dataset(args.dataset)
    global ANALYSIS_DIR, FEATURES_DIR, PLOTS_DIR, MODELS, OOD_DATASETS, OOD_KIND
    ANALYSIS_DIR, FEATURES_DIR, PLOTS_DIR = config.ANALYSIS_DIR, config.FEATURES_DIR, config.PLOTS_DIR
    MODELS, OOD_DATASETS = config.MODELS, config.OOD_DATASETS
    OOD_KIND = {name: meta["kind"] for name, meta in OOD_DATASETS.items()}
    print(f"=== Step 8: Statistical signature battery (ID/OOD/ADV) [{args.dataset}] ===")
    all_results = []
    for model in args.models:
        try:
            all_results.append(analyse_model(model))
        except Exception as e:
            print(f"  [error] {model}: {e}")
            all_results.append({"model": model, "error": str(e)})

    plot_summary(all_results)
    out = ANALYSIS_DIR / "signatures.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Signature analysis saved → {out}")
    print(f"  {len(list(PLOTS_DIR.glob('*.png')))} total plots in {PLOTS_DIR}/")
    print("=== Step 8 complete ===")


if __name__ == "__main__":
    main()
