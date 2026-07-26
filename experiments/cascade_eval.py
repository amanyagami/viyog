"""End-to-end stage-1 + stage-2 cascade evaluation (CPU-only, on featfull_*.h5).

Closes reviewer concerns A-1, A-2 (which first-stage detector? end-to-end metrics?
how are stage-1 FP/FN handled?) and D-1 (does Viyog mitigate or amplify upstream
errors?). The paper evaluates Viyog *in isolation* on already-non-ID samples; this
script wires a real first-stage OOD detector in front of it and reports the full
three-way {ID, OOD, ADV} system behaviour, including error propagation.

Pipeline:
  * Stage-1 (non-ID gate): a standard logit-based OOD detector — Energy
    (E = -logsumexp(logits)) or MSP (max softmax) — computed from the logits
    stored in featfull. Threshold calibrated to keep `--id-tpr` (default 95%)
    of ID inputs (so stage-1 FPR on ID = 5%). Inputs above threshold are
    "flagged non-ID" and forwarded to stage-2.
  * Stage-2 (Viyog router): on flagged inputs, route ADV-vs-OOD with the Viyog
    statistic (L∞ first-layer norm, ORIGINAL; or dormant-fraction, NEW),
    centered by ID mean. Threshold = balanced-accuracy-optimal on the flagged pool.

Reports, per model × (stage-1 detector × stage-2 score):
  - stage-1 recall on OOD / ADV (what fraction even reaches stage-2),
  - end-to-end 3-way confusion matrix + per-class recall + overall accuracy,
  - AMPLIFICATION: of the stage-1 errors (ID flagged as non-ID; OOD/ADV missed),
    how many does the cascade still get wrong — i.e. does Viyog clean up or
    compound upstream mistakes.

    python experiments/cascade_eval.py --dataset cifar100 [--prefix featfull] \
        [--id-tpr 0.95] [--csv out.csv]
"""
from __future__ import annotations

import argparse
import glob

import config
import h5py
import numpy as np

EPS = 1e-8


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / (e.sum(1, keepdims=True) + EPS)


def energy(logits: np.ndarray) -> np.ndarray:
    """Energy OOD score: E = -logsumexp(logits). Higher ⇒ more OOD-like."""
    z = logits.astype(np.float64)
    m = z.max(1)
    return -(m + np.log(np.exp(z - m[:, None]).sum(1) + EPS))


def msp(logits: np.ndarray) -> np.ndarray:
    """Max-softmax-probability. Lower ⇒ more OOD-like; negate to align high=OOD."""
    return -softmax(logits.astype(np.float64)).max(1)


STAGE1 = {"energy": energy, "msp": msp}


def linf_stat(d: dict) -> np.ndarray:
    return d["inf_norms"].astype(np.float64)


def dorm_stat(d: dict, dorm_idx: np.ndarray) -> np.ndarray:
    m = d["filter_means"].astype(np.float64)
    return m[:, dorm_idx].sum(1) / (m.sum(1) + EPS)


def best_balanced_threshold(neg: np.ndarray, pos: np.ndarray) -> tuple[float, int]:
    """Threshold + sign maximizing balanced accuracy for pos-vs-neg (orientation-free)."""
    best_acc, best_thr, best_sign = -1.0, 0.0, 1
    for sign in (+1, -1):
        n, p = sign * neg, sign * pos
        cuts = np.quantile(np.r_[n, p], np.linspace(0.01, 0.99, 99))
        for t in cuts:
            tpr = (p > t).mean()
            tnr = (n <= t).mean()
            ba = 0.5 * (tpr + tnr)
            if ba > best_acc:
                best_acc, best_thr, best_sign = ba, float(t), sign
    return best_thr, best_sign


def load(prefix: str, model: str, split: str):
    p = config.FEATURES_DIR / f"{prefix}_{model}_{split}.h5"
    if not p.exists():
        return None
    with h5py.File(p, "r") as f:
        keys = ("filter_means", "inf_norms", "logits")
        return {k: f[k][:] for k in keys if k in f}


def pool(prefix: str, model: str, kind: str):
    """Concatenate all OOD or all ADV splits for a model."""
    paths = sorted(glob.glob(str(config.FEATURES_DIR / f"{prefix}_{model}_{kind}_*.h5")))
    outs = []
    for p in paths:
        with h5py.File(p, "r") as f:
            outs.append({k: f[k][:] for k in ("filter_means", "inf_norms", "logits") if k in f})
    return outs


def stack(dicts: list[dict], key: str) -> np.ndarray:
    return np.concatenate([d[key] for d in dicts]) if dicts else np.zeros((0,))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar100", choices=list(config.DATASET_SPECS))
    ap.add_argument("--prefix", default="featfull")
    ap.add_argument("--id-tpr", type=float, default=0.95,
                    help="fraction of ID kept by stage-1 (so stage-1 FPR = 1-id_tpr)")
    ap.add_argument("--low-pct", type=float, default=0.10)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    config.set_dataset(args.dataset)

    import pandas as pd
    rows = []
    models = [m for m in config.MODEL_ARCHS
              if (config.FEATURES_DIR / f"{args.prefix}_{m}_id.h5").exists()]
    print(f"=== cascade eval [{args.dataset}, prefix={args.prefix}] models={models} ===")

    for model in models:
        idd = load(args.prefix, model, "id")
        ood = pool(args.prefix, model, "ood")
        adv = pool(args.prefix, model, "adv")
        if idd is None or "logits" not in idd or not ood or not adv:
            print(f"[skip] {model}: missing logits/ood/adv")
            continue

        # dormant filter ranking from clean ID
        m_id = idd["filter_means"].astype(np.float64)
        order = np.argsort(m_id.mean(0))
        k = max(1, int(args.low_pct * m_id.shape[1]))
        dorm = order[:k]

        # assemble per-split logits + viyog statistics
        id_log, ood_log, adv_log = idd["logits"], stack(ood, "logits"), stack(adv, "logits")
        viyog = {
            "viyog_linf": (linf_stat(idd),
                           np.concatenate([linf_stat(d) for d in ood]),
                           np.concatenate([linf_stat(d) for d in adv])),
            "viyog_dorm": (dorm_stat(idd, dorm),
                           np.concatenate([dorm_stat(d, dorm) for d in ood]),
                           np.concatenate([dorm_stat(d, dorm) for d in adv])),
        }

        for s1name, s1fn in STAGE1.items():
            s_id, s_ood, s_adv = s1fn(id_log), s1fn(ood_log), s1fn(adv_log)
            tau = np.quantile(s_id, args.id_tpr)          # keep id_tpr of ID below tau
            flag_id, flag_ood, flag_adv = s_id > tau, s_ood > tau, s_adv > tau
            s1_rec_ood, s1_rec_adv = flag_ood.mean(), flag_adv.mean()

            for s2name, (v_id, v_ood, v_adv) in viyog.items():
                # stage-2 boundary fit on flagged OOD vs flagged ADV
                fo, fa = v_ood[flag_ood], v_adv[flag_adv]
                if len(fo) < 5 or len(fa) < 5:
                    continue
                thr, sign = best_balanced_threshold(fo, fa)

                def route(v):  # 1 ⇒ ADV side, 0 ⇒ OOD side
                    return (sign * v > thr).astype(int)

                # 3-way predictions: 0=ID, 1=OOD, 2=ADV
                def predict(flag, v):
                    pred = np.zeros(len(v), dtype=int)          # default ID (not flagged)
                    pred[flag] = np.where(route(v[flag]) == 1, 2, 1)
                    return pred

                p_id = predict(flag_id, v_id)
                p_ood = predict(flag_ood, v_ood)
                p_adv = predict(flag_adv, v_adv)

                # confusion: rows = true {ID,OOD,ADV}, cols = pred {ID,OOD,ADV}
                conf = np.zeros((3, 3), dtype=int)
                for t, pr in [(0, p_id), (1, p_ood), (2, p_adv)]:
                    for c in (0, 1, 2):
                        conf[t, c] = (pr == c).sum()
                per_class_recall = conf.diagonal() / conf.sum(1).clip(min=1)
                overall_acc = conf.diagonal().sum() / conf.sum()

                # amplification: ADV missed by stage-1 (FN) are routed to ID = dangerous.
                # of stage-1 ID false-positives, fraction the cascade mislabels as ADV
                # (escalates a benign ID input to a security response = worst FP).
                id_fp = flag_id
                id_fp_to_adv = (p_id[id_fp] == 2).mean() if id_fp.any() else 0.0
                adv_missed_to_id = (p_adv == 0).mean()      # ADV predicted ID (security miss)

                rows.append({
                    "model": model, "stage1": s1name, "stage2": s2name,
                    "s1_recall_ood": round(s1_rec_ood, 3),
                    "s1_recall_adv": round(s1_rec_adv, 3),
                    "e2e_recall_ID": round(per_class_recall[0], 3),
                    "e2e_recall_OOD": round(per_class_recall[1], 3),
                    "e2e_recall_ADV": round(per_class_recall[2], 3),
                    "e2e_acc": round(overall_acc, 3),
                    "ADV_missed_to_ID": round(adv_missed_to_id, 3),
                    "ID_FP_escalated_to_ADV": round(id_fp_to_adv, 3),
                })
                print(f"  {model:16} {s1name:6} | {s2name:11} "
                      f"s1_rec(ood/adv)={s1_rec_ood:.2f}/{s1_rec_adv:.2f} "
                      f"e2e_acc={overall_acc:.3f} "
                      f"rec[ID/OOD/ADV]={per_class_recall[0]:.2f}/{per_class_recall[1]:.2f}/{per_class_recall[2]:.2f} "
                      f"ADV→ID(miss)={adv_missed_to_id:.3f}")

    df = pd.DataFrame(rows)
    if len(df):
        print("\n=== mean over models (stage1 × stage2) ===")
        g = df.groupby(["stage1", "stage2"])[
            ["e2e_acc", "e2e_recall_OOD", "e2e_recall_ADV", "ADV_missed_to_ID",
             "ID_FP_escalated_to_ADV"]].mean().round(3)
        print(g.to_string())
    out = args.csv or str(config.ANALYSIS_DIR / f"cascade_{args.dataset}.csv")
    df.to_csv(out, index=False)
    print(f"\n  saved → {out}")


if __name__ == "__main__":
    main()
