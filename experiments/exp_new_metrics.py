"""Test new first-layer signatures computable from the EXISTING per-filter
features (profile = per-filter mean |activation|, plus ID mean/std/cov).

Compares against the current best: B_low_frac (T3=0.856) and E_mahalanobis
(T2=0.970, T3=0.684). Directionless AUROC, OOD pooled / ADV pooled, per 08.
"""
from __future__ import annotations
import sys; sys.path.insert(0, "experiments")
import glob
import h5py
import numpy as np
from numpy.linalg import pinv, svd
from sklearn.metrics import roc_auc_score
import config

config.set_dataset("cifar100")
FEAT = config.FEATURES_DIR
MODELS = list(config.MODEL_ARCHS)
EPS = 1e-8


def means(path):
    with h5py.File(path, "r") as f:
        return f["filter_means"][:].astype(np.float64)


def auroc(s0, s1):
    y = np.r_[np.zeros(len(s0)), np.ones(len(s1))]
    s = np.r_[s0, s1]
    if len(np.unique(s)) < 2:
        return 0.5
    a = roc_auc_score(y, s)
    return max(a, 1 - a)


def collect(model):
    idp = FEAT / f"feat_{model}_id.h5"
    if not idp.exists():
        return None
    ood = [means(p) for p in sorted(glob.glob(str(FEAT / f"feat_{model}_ood_*.h5")))]
    adv = [means(p) for p in sorted(glob.glob(str(FEAT / f"feat_{model}_adv_*.h5")))]
    if not ood or not adv:
        return None
    return means(idp), np.concatenate(ood), np.concatenate(adv)


def build_metrics(id_m):
    """Return a dict name -> fn(profile_matrix)->scalar-per-row, using ID refs."""
    C = id_m.shape[1]
    mu = id_m.mean(0)
    sd = id_m.std(0) + EPS
    order = np.argsort(mu)[::-1]
    k = max(1, int(0.10 * C))
    dorm = order[-k:]                       # bottom 10% by ID mean
    # variance-aware dormant: bottom 10% by ID mean AMONG low-variance filters
    lowvar = np.argsort(sd)[:max(1, C // 2)]          # quietest-varying half
    dorm_lv = np.intersect1d(dorm, lowvar)
    if dorm_lv.size == 0:
        dorm_lv = dorm
    # ID PCA on standardized profiles
    Xs = (id_m - mu) / sd
    U, S, Vt = svd(Xs, full_matrices=False)
    var = S ** 2
    cum = np.cumsum(var) / var.sum()
    kpca = int(np.searchsorted(cum, 0.90) + 1)
    Vk = Vt[:kpca]                          # top-k content subspace
    # full-cov precision for reference Mahalanobis
    cov = np.cov(id_m, rowvar=False)
    prec = pinv(cov + 1e-3 * np.eye(C))

    def z(m):
        return (m - mu) / sd

    return {
        # baseline
        "B_low_frac@10":      lambda m: m[:, dorm].sum(1) / (m.sum(1) + EPS),
        "E_mahalanobis":      lambda m: np.einsum("ni,ij,nj->n", z(m) * sd, prec, z(m) * sd),
        # NEW — variance-normalised dormant band (SNR-weighted B_low_frac)
        "Zdorm_mean":         lambda m: z(m)[:, dorm].mean(1),
        "Zdorm_lowvar_mean":  lambda m: z(m)[:, dorm_lv].mean(1),
        # NEW — out-of-ID-range recruitment count (broadband vs localized)
        "OOR_count_all(>3sd)":   lambda m: (z(m) > 3).mean(1),
        "OOR_count_dorm(>3sd)":  lambda m: (z(m)[:, dorm] > 3).mean(1),
        # NEW — diagonal Mahalanobis (variance-normalised, all filters)
        "Diag_maha_all":      lambda m: (z(m) ** 2).mean(1),
        # NEW — dormant-band spread (broadband ADV → flat → high entropy)
        "Dorm_entropy":       lambda m: _entropy(m[:, dorm]),
        # NEW — PCA residual: energy OUTSIDE the top-k ID content subspace
        "PCA_tail_resid":     lambda m: _tail_resid(z(m), Vk),
        "PCA_tail_resid_dorm":lambda m: _tail_resid(z(m)[:, dorm], svd((id_m[:, dorm]-mu[dorm])/sd[dorm], full_matrices=False)[2][:max(1, len(dorm)//2)]),
    }


def _entropy(sub):
    p = sub / (sub.sum(1, keepdims=True) + EPS)
    return -(p * np.log(p + EPS)).sum(1)


def _tail_resid(xs, Vk):
    proj = xs @ Vk.T @ Vk
    return np.sqrt(((xs - proj) ** 2).sum(1))


# run
agg = {}
for model in MODELS:
    got = collect(model)
    if got is None:
        print(f"[skip] {model}"); continue
    id_m, ood_m, adv_m = got
    fns = build_metrics(id_m)
    for name, fn in fns.items():
        i, o, a = fn(id_m), fn(ood_m), fn(adv_m)
        t1, t2, t3 = auroc(i, o), auroc(i, a), auroc(o, a)
        agg.setdefault(name, []).append((t1, t2, t3))

print(f"\n{'signature':24} | {'T1 ID-OOD':>10} {'T2 ID-ADV':>10} {'T3 OOD-ADV':>11}   (mean/{len(MODELS)} models)")
print("-" * 64)
for name, rows in agg.items():
    arr = np.array(rows)
    t1, t2, t3 = arr.mean(0)
    print(f"{name:24} | {t1:>10.3f} {t2:>10.3f} {t3:>11.3f}")
