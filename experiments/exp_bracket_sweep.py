"""Sweep the low/large neuron-group bracket % and measure B_low_frac /
B_large_frac / B_ratio_low_large separability (AUROC) on T1/T2/T3.

Replicates 08_signatures' recipe exactly: groups from ID-mean ranking, OOD
pooled over all OOD datasets, ADV pooled over all attacks, directionless AUROC.
Reads the already-extracted CIFAR-100 features in results/features/.
"""
from __future__ import annotations
import sys; sys.path.insert(0, "experiments")
import glob
import numpy as np
from sklearn.metrics import roc_auc_score
import config

config.set_dataset("cifar100")
FEAT = config.FEATURES_DIR
MODELS = list(config.MODEL_ARCHS)
PCTS = [1, 2, 5, 10, 15, 20, 25]


def load_means(path):
    import h5py
    with h5py.File(path, "r") as f:
        return f["filter_means"][:].astype(np.float64)


def auroc_dl(s0, s1):
    y = np.r_[np.zeros(len(s0)), np.ones(len(s1))]
    s = np.r_[s0, s1]
    if len(np.unique(s)) < 2:
        return 0.5
    a = roc_auc_score(y, s)
    return max(a, 1 - a)


def fracs(m, idx):
    return m[:, idx].sum(1) / (m.sum(1) + 1e-12)


def collect(model):
    idp = FEAT / f"feat_{model}_id.h5"
    if not idp.exists():
        return None
    id_m = load_means(idp)
    ood = [load_means(p) for p in sorted(glob.glob(str(FEAT / f"feat_{model}_ood_*.h5")))]
    adv = [load_means(p) for p in sorted(glob.glob(str(FEAT / f"feat_{model}_adv_*.h5")))]
    if not ood or not adv:
        return None
    return id_m, np.concatenate(ood), np.concatenate(adv)


# results[metric][pct] -> list of (model, T1, T2, T3)
results = {"B_low_frac": {}, "B_large_frac": {}, "B_ratio_low_large": {}}

for model in MODELS:
    got = collect(model)
    if got is None:
        print(f"[skip] {model}")
        continue
    id_m, ood_m, adv_m = got
    C = id_m.shape[1]
    order = np.argsort(id_m.mean(0))[::-1]   # high → low, same as 08
    for pct in PCTS:
        n = max(1, int(C * pct / 100))
        large, low = order[:n], order[-n:]

        def metrics(fn):
            i, o, a = fn(id_m), fn(ood_m), fn(adv_m)
            return (auroc_dl(i, o), auroc_dl(i, a), auroc_dl(o, a))

        low_f = metrics(lambda m: fracs(m, low))
        lrg_f = metrics(lambda m: fracs(m, large))
        mean_low = m_idx = None
        # ratio low/large mean
        def ratio(m):
            return m[:, low].mean(1) / (m[:, large].mean(1) + 1e-12)
        rat = metrics(ratio)

        results["B_low_frac"].setdefault(pct, []).append((model, *low_f))
        results["B_large_frac"].setdefault(pct, []).append((model, *lrg_f))
        results["B_ratio_low_large"].setdefault(pct, []).append((model, *rat))


def show(metric):
    print(f"\n===== {metric}  (AUROC, mean over {len(MODELS)} models) =====")
    print(f"{'bracket%':>9} | {'T1 ID-OOD':>10} {'T2 ID-ADV':>10} {'T3 OOD-ADV':>11}")
    print("-" * 48)
    for pct in PCTS:
        rows = results[metric].get(pct, [])
        if not rows:
            continue
        arr = np.array([[r[1], r[2], r[3]] for r in rows])
        t1, t2, t3 = arr.mean(0)
        print(f"{pct:>8}% | {t1:>10.3f} {t2:>10.3f} {t3:>11.3f}")


for metric in results:
    show(metric)

# Per-model T3 for the headline metric, to expose model spread.
print("\n===== B_low_frac  T3 (OOD-vs-ADV) per model =====")
print(f"{'bracket%':>9} | " + " ".join(f"{m[:10]:>11}" for m in MODELS))
print("-" * (11 + 12 * len(MODELS)))
for pct in PCTS:
    rows = {r[0]: r[3] for r in results["B_low_frac"].get(pct, [])}
    print(f"{pct:>8}% | " + " ".join(f"{rows.get(m, float('nan')):>11.3f}" for m in MODELS))
