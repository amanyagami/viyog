"""Comprehensive head-to-head: best Viyog first-layer signatures vs all logit
OOD baselines, across every valid metric and every task the reviewers ask for.

Tasks            T1 = ID-vs-OOD, T2 = ID-vs-ADV, T3 = OOD-vs-ADV (deployment).
Metrics          Held-out AUROC with calibration-fixed direction + raw AUROC +
                 cross-model direction-consistency; calibrated FPR/TPR; AUPR.
Protocol         Disjoint deterministic calibration/test partitions per source.
Cuts             pooled / far-OOD / near-OOD ; per-attack (T2,T3).
Methods          Viyog first-layer signatures (L-inf, dorm-TV, dorm-HF) computed
                 from the stored first-conv stats, PLUS logit baselines (MSP,
                 MaxLogit, Energy, Entropy, GEN, KL-Matching) from stored logits.
                 Feature baselines (Mahalanobis/KNN/ViM) come from
                 baselines_feature.py (needs a model pass) and are merged in
                 downstream — this script is pure post-hoc / CPU.

    python experiments/full_eval.py --dataset cifar100
Outputs results/analysis/full_eval_<ds>_{permodel,summary,perattack}.csv
"""

from __future__ import annotations

import argparse
import glob
import os

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
import config
import h5py
import numpy as np
from eval_protocol import (
    PROTOCOL_NAME,
    evaluate_calibrated_binary,
    evaluate_fixed_orientation,
    split_rows,
)

CORE6 = ["resnet50", "densenet121", "convnextv2_base", "vit_base", "swin_tiny", "mobilenetv3_l"]
NEAR_OOD = {"cifar10", "stl10"}
DEGEN_TOL = 0.02
EPS = 1e-8


def eval_pair(cal_pos, cal_neg, test_pos, test_neg):
    """Fit direction and operating points on calibration; evaluate on test."""
    result = evaluate_calibrated_binary(cal_pos, cal_neg, test_pos, test_neg)
    result["auroc_dl"] = result["auroc_oriented"]  # legacy output column name
    result["fpr95"] = result["test_fpr_at_calibrated_tpr"]
    return result


# ---------- feature / score builders ----------
def load(p, keys):
    with h5py.File(p, "r") as f:
        return {k: f[k][:].astype(np.float64) for k in keys if k in f}


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def logit_scores(logits, klm_ref=None):
    """OOD-direction scores from logits (higher => more OOD/ADV)."""
    p = softmax(logits)
    out = {
        "MSP": -p.max(1),  # high => OOD
        "MaxLogit": -logits.max(1),
        "Energy": -(
            np.log(np.exp(logits - logits.max(1, keepdims=True)).sum(1)) + logits.max(1)
        ),  # -logsumexp
        "Entropy": -(p * np.log(p + EPS)).sum(1),
        "GEN": np.sum(p**0.1 * (1 - p) ** 0.1, axis=1),  # generalized entropy gamma=0.1
    }
    if klm_ref is not None:  # KL-Matching to ID class means
        # KL(p || m) = sum(p log p) - p @ log(m). The rowwise matrix
        # formulation avoids a sample-by-reference Python loop while preserving
        # float64 computation and the existing epsilon regularization.
        entropy_term = np.sum(p * np.log(p + EPS), axis=1, keepdims=True)
        kl = entropy_term - p @ np.log(klm_ref + EPS).T
        out["KLMatching"] = kl.min(1)
    return out


SIG_KEYS = ("inf_norms", "filter_tv", "filter_hf", "filter_means", "logits", "labels")

REQUIRED_SCORE_KEYS = {"inf_norms", "filter_tv", "filter_hf", "filter_means", "logits"}


def feature_problem(data, *, require_labels=False):
    """Return a provenance reason when a stored feature mapping is unusable."""
    required = REQUIRED_SCORE_KEYS | ({"labels"} if require_labels else set())
    missing = sorted(required.difference(data))
    if missing:
        return f"missing_keys:{','.join(missing)}"
    lengths = {key: len(data[key]) for key in required}
    if len(set(lengths.values())) != 1:
        return f"row_count_mismatch:{lengths}"
    if next(iter(lengths.values())) < 2:
        return "fewer_than_two_rows"
    for key in REQUIRED_SCORE_KEYS:
        if not np.isfinite(data[key]).all():
            return f"nonfinite:{key}"
    # An all-zero first-layer hook is an extraction failure, not a detector score.
    if float(np.max(data["filter_means"])) <= 0.0:
        return "all_zero_first_layer_features"
    return None


def sig_scores(d, dorm):
    """First-layer Viyog signatures (higher => more OOD/ADV for the natural sign)."""
    return {
        "Viyog_Linf": d["inf_norms"],
        "ViyogD_tv_dorm": d["filter_tv"][:, dorm].mean(1),
        "ViyogD_hf_dorm": d["filter_hf"][:, dorm].mean(1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar100")
    ap.add_argument("--models", nargs="+", default=CORE6)
    ap.add_argument("--low-pct", type=float, default=10.0)
    ap.add_argument("--calibration-fraction", type=float, default=0.20)
    ap.add_argument(
        "--split-seed", type=int, default=0, help="deterministic calibration/test split seed"
    )
    args = ap.parse_args()
    config.set_dataset(args.dataset)
    FD = config.FEATURES_DIR
    models = [
        m
        for m in args.models
        if (FD / f"featfull_{m}_id.h5").exists()
        and glob.glob(str(FD / f"featfull_{m}_adv_*.h5"))
        and glob.glob(str(FD / f"featfull_{m}_ood_*.h5"))
    ]
    if not models:
        print(f"[{args.dataset}] no complete Core-6 features — skip")
        return
    import pandas as pd

    rows, parows, exclusions = [], [], []

    def excluded(model, source, path, reason):
        exclusions.append(dict(dataset=args.dataset, model=model, source=source, file=path, reason=reason))
        print(f"[{args.dataset}] excluding {model} ({source}): {reason} ({path})")

    for m in models:
        id_path = str(FD / f"featfull_{m}_id.h5")
        id_all = load(id_path, SIG_KEYS)
        problem = feature_problem(id_all, require_labels=True)
        if problem:
            excluded(m, "id", id_path, problem)
            continue
        id_cal, id_test = split_rows(id_all, args.calibration_fraction, args.split_seed)
        ch = id_cal["filter_means"].mean(0)
        C = len(ch)
        alive = np.where(ch > 1e-4)[0]
        alive = alive if len(alive) else np.arange(C)
        k = max(1, int(args.low_pct / 100.0 * len(alive)))
        dorm = alive[np.argsort(ch[alive])[:k]]

        id_prob = softmax(id_cal["logits"])
        valid_labels = id_cal["labels"] >= 0
        classes = np.unique(id_cal["labels"][valid_labels])
        if len(classes):
            klm_ref = np.stack([id_prob[id_cal["labels"] == label].mean(0) for label in classes])
        else:
            klm_ref = id_prob.mean(0, keepdims=True)

        ood_files = sorted(glob.glob(str(FD / f"featfull_{m}_ood_*.h5")))
        adv_files = sorted(glob.glob(str(FD / f"featfull_{m}_adv_*.h5")))
        ood_all, adv_all = {}, {}
        for p in ood_files:
            name = os.path.basename(p).split("_ood_")[1][:-3]
            data = load(p, SIG_KEYS)
            problem = feature_problem(data)
            if problem:
                excluded(m, f"ood:{name}", p, problem)
                continue
            ood_all[name] = data
        for p in adv_files:
            name = os.path.basename(p).split("_adv_")[1][:-3]
            data = load(p, SIG_KEYS)
            problem = feature_problem(data)
            if problem:
                excluded(m, f"adv:{name}", p, problem)
                continue
            adv_all[name] = data
        if not ood_all or not adv_all:
            excluded(m, "model", str(FD), "no_usable_ood_or_adv_sources")
            continue
        ood_parts = {
            name: split_rows(data, args.calibration_fraction, args.split_seed)
            for name, data in ood_all.items()
        }
        adv_parts = {
            name: split_rows(data, args.calibration_fraction, args.split_seed)
            for name, data in adv_all.items()
        }

        def all_scores(d):
            s = sig_scores(d, dorm)
            s.update(logit_scores(d["logits"], klm_ref))
            return s

        id_cal_s, id_test_s = all_scores(id_cal), all_scores(id_test)
        ood_cal_s = {name: all_scores(parts[0]) for name, parts in ood_parts.items()}
        ood_test_s = {name: all_scores(parts[1]) for name, parts in ood_parts.items()}
        adv_cal_s = {name: all_scores(parts[0]) for name, parts in adv_parts.items()}
        adv_test_s = {name: all_scores(parts[1]) for name, parts in adv_parts.items()}
        methods = list(id_cal_s.keys())

        def pool(dct, sel=None):
            return {
                mth: np.concatenate([v[mth] for s, v in dct.items() if sel is None or sel(s)])
                for mth in methods
            }

        OOD_CAL, OOD_TEST = pool(ood_cal_s), pool(ood_test_s)
        ADV_CAL, ADV_TEST = pool(adv_cal_s), pool(adv_test_s)
        has_far = any(name not in NEAR_OOD for name in ood_all)
        has_near = any(name in NEAR_OOD for name in ood_all)
        FAR_TEST = pool(ood_test_s, lambda name: name not in NEAR_OOD) if has_far else None
        NEAR_TEST = pool(ood_test_s, lambda name: name in NEAR_OOD) if has_near else None

        for mth in methods:
            t1 = eval_pair(OOD_CAL[mth], id_cal_s[mth], OOD_TEST[mth], id_test_s[mth])
            t2 = eval_pair(ADV_CAL[mth], id_cal_s[mth], ADV_TEST[mth], id_test_s[mth])
            t3 = eval_pair(ADV_CAL[mth], OOD_CAL[mth], ADV_TEST[mth], OOD_TEST[mth])
            t1f = (
                evaluate_fixed_orientation(FAR_TEST[mth], id_test_s[mth], sign=t1["sign"])
                if FAR_TEST
                else {}
            )
            t1n = (
                evaluate_fixed_orientation(NEAR_TEST[mth], id_test_s[mth], sign=t1["sign"])
                if NEAR_TEST
                else {}
            )
            rows.append(
                dict(
                    dataset=args.dataset,
                    model=m,
                    method=mth,
                    protocol=PROTOCOL_NAME,
                    calibration_fraction=args.calibration_fraction,
                    split_seed=args.split_seed,
                    klm_reference_classes=len(classes),
                    n_cal_id=len(id_cal_s[mth]),
                    n_test_id=len(id_test_s[mth]),
                    n_cal_ood=len(OOD_CAL[mth]),
                    n_test_ood=len(OOD_TEST[mth]),
                    n_cal_adv=len(ADV_CAL[mth]),
                    n_test_adv=len(ADV_TEST[mth]),
                    calibrated_orientation=True,
                    T1_dl=round(t1["auroc_dl"], 4),
                    T1_raw=round(t1["auroc_raw"], 4),
                    T1_sign=t1["sign"],
                    T1_cal_auc=round(t1["calibration_auroc_raw"], 4),
                    T1_fpr95=round(t1["fpr95"], 4),
                    T1_aupr=round(t1["aupr"], 4),
                    T1_test_tpr_at_cal95tpr=round(t1["test_tpr_at_calibrated_tpr"], 4),
                    T1_recall_at_cal5fpr=round(t1["test_tpr_at_calibrated_fpr"], 4),
                    T1_test_fpr_at_cal5fpr=round(t1["test_fpr_at_calibrated_fpr"], 4),
                    T1far_dl=round(t1f.get("auroc_oriented", np.nan), 4),
                    T1near_dl=round(t1n.get("auroc_oriented", np.nan), 4),
                    T2_dl=round(t2["auroc_dl"], 4),
                    T2_raw=round(t2["auroc_raw"], 4),
                    T2_sign=t2["sign"],
                    T2_cal_auc=round(t2["calibration_auroc_raw"], 4),
                    T2_fpr95=round(t2["fpr95"], 4),
                    T2_aupr=round(t2["aupr"], 4),
                    T2_test_tpr_at_cal95tpr=round(t2["test_tpr_at_calibrated_tpr"], 4),
                    T2_recall_at_cal5fpr=round(t2["test_tpr_at_calibrated_fpr"], 4),
                    T2_test_fpr_at_cal5fpr=round(t2["test_fpr_at_calibrated_fpr"], 4),
                    T3_dl=round(t3["auroc_dl"], 4),
                    T3_raw=round(t3["auroc_raw"], 4),
                    T3_sign=t3["sign"],
                    T3_cal_auc=round(t3["calibration_auroc_raw"], 4),
                    T3_fpr95=round(t3["fpr95"], 4),
                    T3_aupr=round(t3["aupr"], 4),
                    T3_test_tpr_at_cal95tpr=round(t3["test_tpr_at_calibrated_tpr"], 4),
                    T3_recall_at_cal5fpr=round(t3["test_tpr_at_calibrated_fpr"], 4),
                    T3_test_fpr_at_cal5fpr=round(t3["test_fpr_at_calibrated_fpr"], 4),
                    T3_degen=int(abs(t3["calibration_auroc_raw"] - 0.5) < DEGEN_TOL),
                )
            )
            # per-attack
            for atk, av in adv_test_s.items():
                t2a = evaluate_fixed_orientation(av[mth], id_test_s[mth], sign=t2["sign"])
                t3a = evaluate_fixed_orientation(av[mth], OOD_TEST[mth], sign=t3["sign"])
                parows.append(
                    dict(
                        dataset=args.dataset,
                        model=m,
                        method=mth,
                        attack=atk,
                        protocol=PROTOCOL_NAME,
                        calibration_fraction=args.calibration_fraction,
                        split_seed=args.split_seed,
                        n_cal_attack=len(adv_cal_s[atk][mth]),
                        n_test_attack=len(av[mth]),
                        T2_dl=round(t2a["auroc_oriented"], 4),
                        T2_sign=t2["sign"],
                        T3_dl=round(t3a["auroc_oriented"], 4),
                        T3_sign=t3["sign"],
                    )
                )

    df = pd.DataFrame(rows)
    pa = pd.DataFrame(parows)
    ad = config.ANALYSIS_DIR
    ad.mkdir(parents=True, exist_ok=True)
    df.to_csv(ad / f"full_eval_{args.dataset}_permodel.csv", index=False)
    pa.to_csv(ad / f"full_eval_{args.dataset}_perattack.csv", index=False)
    pd.DataFrame(exclusions).to_csv(ad / f"full_eval_{args.dataset}_exclusions.csv", index=False)
    if not rows:
        print(f"[{args.dataset}] every requested model was excluded — no results written")
        return

    # ---- summary; direction consistency is diagnostic, never a test-time gate ----
    def summarize(task):
        out = []
        for mth, sub in df.groupby("method"):
            nd = sub[sub["T3_degen"] == 0] if task == "T3" else sub
            sg = nd[f"{task}_sign"]
            pos = int((sg > 0).sum())
            neg = int((sg < 0).sum())
            consistent = pos == 0 or neg == 0
            out.append(
                dict(
                    method=mth,
                    task=task,
                    protocol=PROTOCOL_NAME,
                    calibration_fraction=args.calibration_fraction,
                    split_seed=args.split_seed,
                    heldout_protocol=True,
                    calibrated_orientation=True,
                    mean_dl=round(sub[f"{task}_dl"].mean(), 4),
                    mean_oriented=round(sub[f"{task}_dl"].mean(), 4),
                    mean_raw=round(sub[f"{task}_raw"].mean(), 4),
                    deployable=round(sub[f"{task}_dl"].mean(), 4),
                    dir_consistent=consistent,
                    n_pos=pos,
                    n_neg=neg,
                    mean_fpr95=round(sub[f"{task}_fpr95"].mean(), 4),
                    mean_test_tpr_at_cal95tpr=round(sub[f"{task}_test_tpr_at_cal95tpr"].mean(), 4),
                    mean_tpr_at_cal_fpr=round(sub[f"{task}_recall_at_cal5fpr"].mean(), 4),
                    mean_test_fpr_at_cal_fpr=round(sub[f"{task}_test_fpr_at_cal5fpr"].mean(), 4),
                    mean_aupr=round(sub[f"{task}_aupr"].mean(), 4),
                    T1far_dl=round(sub["T1far_dl"].mean(), 4) if task == "T1" else np.nan,
                    T1near_dl=round(sub["T1near_dl"].mean(), 4) if task == "T1" else np.nan,
                )
            )
        return out

    summ = []
    for t in ("T1", "T2", "T3"):
        summ += summarize(t)
    sdf = pd.DataFrame(summ)
    sdf.to_csv(ad / f"full_eval_{args.dataset}_summary.csv", index=False)

    import pandas as pd

    pd.set_option("display.width", 230)
    pd.set_option("display.max_columns", 30)
    print(f"\n############  FULL EVAL — {args.dataset}  ({len(models)} models)  ############")
    for t, name in [("T3", "OOD-vs-ADV (DEPLOYMENT)"), ("T2", "ID-vs-ADV"), ("T1", "ID-vs-OOD")]:
        s = sdf[sdf.task == t].sort_values("deployable", ascending=False)
        cols = [
            "method",
            "mean_oriented",
            "dir_consistent",
            "mean_fpr95",
            "mean_tpr_at_cal_fpr",
            "mean_aupr",
        ]
        if t == "T1":
            cols += ["T1far_dl", "T1near_dl"]
        print(f"\n--- {t}: {name}  (sorted by deployable AUROC) ---")
        print(s[cols].to_string(index=False))
    print(f"\n  saved -> {ad}/full_eval_{args.dataset}_*.csv")


if __name__ == "__main__":
    main()
