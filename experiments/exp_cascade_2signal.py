"""2-signal stage-1 gate (Energy ⊕ first-conv deviation) to lift weak e2e ADV recall.

The single-signal cascade (cascade_eval.py) gates non-ID with Energy alone. Energy is
blind to adversarials (they are confidently *wrong* → low energy → not flagged), so few
ADV reach stage-2 and end-to-end ADV recall is capped regardless of the stage-2 router
(the honest A-w1 / D-w1 gap). This tests a fix that costs nothing extra at deployment:
gate on max(z_energy, |z_dorm|) — i.e. flag non-ID if EITHER the logit energy is high OR
the first-conv dorm-TV deviates from its ID mean (in either direction). ADV suppresses the
dorm band (negative deviation) so the |z_dorm| arm catches exactly what energy misses.

Crucially the 2-signal gate is calibrated to the SAME 5% ID false-positive rate as the
energy-only gate (threshold at the 95th percentile of the ID combined score), so any gain
in ADV recall is NOT bought with extra ID false positives. The first-conv statistic does
double duty: |z| (2-sided) flags at stage-1, signed value routes OOD-vs-ADV at stage-2.

    python experiments/exp_cascade_2signal.py --dataset cifar100
"""
from __future__ import annotations

import argparse
import glob
import os

import config
import h5py
import numpy as np

EPS = 1e-8
CORRUPT = {"mobileone_s1"}


def energy(logits):
    z = logits.astype(np.float64)
    m = z.max(1)
    return -(m + np.log(np.exp(z - m[:, None]).sum(1) + EPS))


def adaptive_band(prof, p=10.0):
    live = np.where(prof > 1e-4 * prof.max())[0]
    if len(live) < 4:
        live = np.arange(len(prof))
    order = live[np.argsort(prof[live])]
    return order[: max(1, int(round(p / 100.0 * len(order))))]


def best_balanced_threshold(neg, pos):
    best, bt, bs = -1.0, 0.0, 1
    for s in (1, -1):
        n, p = s * neg, s * pos
        for t in np.quantile(np.r_[n, p], np.linspace(0.02, 0.98, 60)):
            ba = 0.5 * ((p > t).mean() + (n <= t).mean())
            if ba > best:
                best, bt, bs = ba, float(t), s
    return bt, bs


def load(p):
    with h5py.File(p, "r") as f:
        return {k: f[k][:].astype(np.float64) for k in ("filter_means", "filter_tv", "logits")}


def pool(FD, model, kind):
    ds = [load(p) for p in sorted(glob.glob(str(FD / f"featfull_{model}_{kind}_*.h5")))
          if load(p)["filter_means"].max() > 0]
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar100", choices=list(config.DATASET_SPECS))
    ap.add_argument("--id-tpr", type=float, default=0.95, help="1 - target ID FPR")
    ap.add_argument("--low-pct", type=float, default=10.0)
    args = ap.parse_args()
    config.set_dataset(args.dataset)
    FD = config.FEATURES_DIR
    import pandas as pd

    models = sorted({os.path.basename(p).split("featfull_")[1].split("_id.h5")[0]
                     for p in glob.glob(str(FD / "featfull_*_id.h5"))})
    models = [m for m in models if m not in CORRUPT]
    rows = []
    for m in models:
        idp = FD / f"featfull_{m}_id.h5"
        if not idp.exists():
            continue
        idd = load(str(idp))
        if idd["filter_means"].max() == 0:
            continue
        dorm = adaptive_band(idd["filter_means"].mean(0), args.low_pct)
        oodd, advd = pool(FD, m, "ood"), pool(FD, m, "adv")
        if not oodd or not advd:
            continue

        def dorm_tv(d):
            return d["filter_tv"][:, dorm].mean(1)

        # signals: energy (high=OOD-like), dorm-TV value
        e_id = energy(idd["logits"]); v_id = dorm_tv(idd)
        e_ood = np.concatenate([energy(d["logits"]) for d in oodd]); v_ood = np.concatenate([dorm_tv(d) for d in oodd])
        e_adv = np.concatenate([energy(d["logits"]) for d in advd]); v_adv = np.concatenate([dorm_tv(d) for d in advd])
        # z-scores vs ID
        ze = lambda e: (e - e_id.mean()) / (e_id.std() + EPS)         # high = OOD-like
        zv = lambda v: (v - v_id.mean()) / (v_id.std() + EPS)         # signed; |.| = deviation
        # combined gate scores
        def comb(e, v):
            return np.maximum(ze(e), np.abs(zv(v)))                   # 2-signal: energy OR dorm-deviation
        c_id, c_ood, c_adv = comb(e_id, v_id), comb(e_ood, v_ood), comb(e_adv, v_adv)

        def run_gate(s_id, s_ood, s_adv, name):
            tau = np.quantile(s_id, args.id_tpr)                     # → ID FPR = 1-id_tpr (5%)
            f_id, f_ood, f_adv = s_id > tau, s_ood > tau, s_adv > tau
            # stage-2: route flagged OOD vs ADV with signed dorm value
            fo, fa = v_ood[f_ood], v_adv[f_adv]
            if len(fo) < 5 or len(fa) < 5:
                e2e_adv = e2e_ood = idfp = np.nan
            else:
                thr, sgn = best_balanced_threshold(fo, fa)
                route = lambda v: (sgn * v > thr).astype(int)         # 1 ⇒ ADV side
                # ABSOLUTE end-to-end recall over the FULL class (must be flagged AND routed right)
                e2e_adv = float((f_adv & (route(v_adv) == 1)).mean())   # P(pred=ADV | true ADV)
                e2e_ood = float((f_ood & (route(v_ood) == 0)).mean())   # P(pred=OOD | true OOD)
                idfp = float((f_id & (route(v_id) == 1)).mean())        # P(pred=ADV | true ID): the dangerous FP
            return {f"{name}_idFPR": round(float(f_id.mean()), 3),
                    f"{name}_s1_adv": round(float(f_adv.mean()), 3),
                    f"{name}_s1_ood": round(float(f_ood.mean()), 3),
                    f"{name}_e2e_adv": round(e2e_adv, 3) if e2e_adv == e2e_adv else np.nan,
                    f"{name}_e2e_ood": round(e2e_ood, 3) if e2e_ood == e2e_ood else np.nan,
                    f"{name}_id2adv": round(idfp, 3) if idfp == idfp else np.nan}

        row = {"model": m}
        row.update(run_gate(ze(e_id), ze(e_ood), ze(e_adv), "energy"))   # baseline: energy only
        row.update(run_gate(c_id, c_ood, c_adv, "twosig"))               # 2-signal gate
        rows.append(row)
        print(f"  {m} done", flush=True)

    df = pd.DataFrame(rows)
    out = str(config.ANALYSIS_DIR / f"cascade_2signal_{args.dataset}.csv")
    df.to_csv(out, index=False)
    mean = df.select_dtypes("number").mean().round(3)
    print(f"\n=== 2-signal stage-1 gate vs energy-only [{args.dataset}], {len(df)} models, ID FPR≈{1-args.id_tpr:.0%} ===")
    print(f"  stage-1 ADV recall : energy {mean['energy_s1_adv']:.3f} → 2-signal {mean['twosig_s1_adv']:.3f}  "
          f"({mean['twosig_s1_adv']-mean['energy_s1_adv']:+.3f})")
    print(f"  stage-1 OOD recall : energy {mean['energy_s1_ood']:.3f} → 2-signal {mean['twosig_s1_ood']:.3f}")
    print(f"  e2e ADV recall     : energy {mean['energy_e2e_adv']:.3f} → 2-signal {mean['twosig_e2e_adv']:.3f}  "
          f"({mean['twosig_e2e_adv']-mean['energy_e2e_adv']:+.3f})  ← the A-w1/D-w1 gap")
    print(f"  e2e OOD recall     : energy {mean['energy_e2e_ood']:.3f} → 2-signal {mean['twosig_e2e_ood']:.3f}")
    print(f"  ID→ADV mis-escalate: energy {mean['energy_id2adv']:.3f} → 2-signal {mean['twosig_id2adv']:.3f}  (lower safer)")
    print(f"  realized ID FPR    : energy {mean['energy_idFPR']:.3f} / 2-signal {mean['twosig_idFPR']:.3f}  (held equal)")
    print(f"\n  saved → {out}")


if __name__ == "__main__":
    main()
