"""Leakage-free utilities for post-hoc detector evaluation.

The feature files used by the experiment scripts contain one finite pool per
semantic source (ID, an OOD data set, or an adversarial attack).  This module
provides a small, deterministic protocol for separating calibration from test
data and for fixing score direction and operating thresholds on calibration
data only.

The helpers are deliberately NumPy-only so they can be tested without loading
models or feature files.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

PROTOCOL_NAME = "heldout_calibration_v1"


def _scores(values: np.ndarray | Sequence[float], name: str) -> np.ndarray:
    scores = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(scores).all():
        raise ValueError(f"{name} contains non-finite scores")
    return scores


def deterministic_split_indices(
    n_samples: int,
    calibration_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return disjoint, exhaustive calibration/test row indices.

    Applying the same ``seed`` to equally ordered attack files preserves source
    grouping across attacks: row ``i`` is always assigned to the same side.
    """
    if n_samples < 2:
        raise ValueError("at least two samples are required for a held-out split")
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be strictly between 0 and 1")
    n_calibration = round(n_samples * calibration_fraction)
    n_calibration = min(max(n_calibration, 1), n_samples - 1)
    permutation = np.random.default_rng(seed).permutation(n_samples)
    calibration = np.sort(permutation[:n_calibration])
    test = np.sort(permutation[n_calibration:])
    return calibration, test


def split_rows(
    arrays: Mapping[str, np.ndarray],
    calibration_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Split every row-aligned array in a mapping with identical indices."""
    if not arrays:
        raise ValueError("cannot split an empty array mapping")
    lengths = {key: len(value) for key, value in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"row-aligned arrays have different lengths: {lengths}")
    calibration_idx, test_idx = deterministic_split_indices(
        next(iter(lengths.values())), calibration_fraction, seed
    )
    calibration = {key: np.asarray(value)[calibration_idx] for key, value in arrays.items()}
    test = {key: np.asarray(value)[test_idx] for key, value in arrays.items()}
    return calibration, test


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    cumulative = np.cumsum(counts)
    average = (cumulative - counts + cumulative + 1) / 2.0
    return average[inverse]


def auroc(
    positive: np.ndarray | Sequence[float],
    negative: np.ndarray | Sequence[float],
) -> float:
    """AUROC with ``positive`` as class 1 and higher scores predicting class 1."""
    pos = _scores(positive, "positive")
    neg = _scores(negative, "negative")
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = _ranks(np.concatenate([pos, neg]))
    rank_sum = ranks[: len(pos)].sum()
    return float((rank_sum - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def average_precision(
    positive: np.ndarray | Sequence[float],
    negative: np.ndarray | Sequence[float],
) -> float:
    """Non-interpolated average precision for scores oriented high=positive."""
    pos = _scores(positive, "positive")
    neg = _scores(negative, "negative")
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    labels = np.concatenate([np.ones(len(pos), dtype=np.int8), np.zeros(len(neg), dtype=np.int8)])
    scores = np.concatenate([pos, neg])
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    true_positives = np.cumsum(sorted_labels)
    group_ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(sorted_scores) - 1]
    precision = true_positives[group_ends] / (group_ends + 1)
    positive_counts = np.diff(np.r_[0, true_positives[group_ends]])
    return float(np.sum(precision * positive_counts) / len(pos))


def fit_orientation(
    calibration_positive: np.ndarray | Sequence[float],
    calibration_negative: np.ndarray | Sequence[float],
    *,
    tie_sign: int = 1,
) -> int:
    """Fit a high-positive score sign using calibration data only."""
    if tie_sign not in (-1, 1):
        raise ValueError("tie_sign must be -1 or +1")
    value = auroc(calibration_positive, calibration_negative)
    if np.isnan(value):
        raise ValueError("both calibration classes must be non-empty")
    if value > 0.5:
        return 1
    if value < 0.5:
        return -1
    return tie_sign


def fit_balanced_threshold(
    calibration_negative: np.ndarray | Sequence[float],
    calibration_positive: np.ndarray | Sequence[float],
    *,
    sign: int,
) -> tuple[float, float]:
    """Fit an oriented threshold maximizing calibration balanced accuracy.

    Predictions use ``sign * score > threshold``.  The exact empirical optimum
    is found in O(n log n), with a deterministic lowest-threshold tie break.
    """
    if sign not in (-1, 1):
        raise ValueError("sign must be -1 or +1")
    neg = sign * _scores(calibration_negative, "calibration_negative")
    pos = sign * _scores(calibration_positive, "calibration_positive")
    if len(neg) == 0 or len(pos) == 0:
        raise ValueError("both calibration classes must be non-empty")

    values = np.concatenate([neg, pos])
    labels = np.concatenate([np.zeros(len(neg), dtype=np.int8), np.ones(len(pos), dtype=np.int8)])
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_labels = labels[order]
    cumulative_pos = np.cumsum(sorted_labels)
    cumulative_neg = np.cumsum(1 - sorted_labels)
    group_ends = np.r_[np.flatnonzero(np.diff(sorted_values)), len(sorted_values) - 1]
    thresholds = sorted_values[group_ends]
    true_positive_rate = (len(pos) - cumulative_pos[group_ends]) / len(pos)
    true_negative_rate = cumulative_neg[group_ends] / len(neg)
    balanced_accuracy = 0.5 * (true_positive_rate + true_negative_rate)
    best = int(np.argmax(balanced_accuracy))
    return float(thresholds[best]), float(balanced_accuracy[best])


def threshold_for_tpr(
    calibration_positive: np.ndarray | Sequence[float],
    target_tpr: float,
    *,
    sign: int,
) -> float:
    """Fit a threshold with at least the requested empirical calibration TPR.

    Predictions at this operating point use ``sign * score >= threshold``.
    """
    if sign not in (-1, 1):
        raise ValueError("sign must be -1 or +1")
    if not 0.0 < target_tpr <= 1.0:
        raise ValueError("target_tpr must be in (0, 1]")
    pos = np.sort(sign * _scores(calibration_positive, "calibration_positive"))
    if len(pos) == 0:
        raise ValueError("calibration_positive must be non-empty")
    allowed_misses = int(np.floor((1.0 - target_tpr) * len(pos) + 1e-12))
    return float(pos[min(allowed_misses, len(pos) - 1)])


def threshold_for_fpr(
    calibration_negative: np.ndarray | Sequence[float],
    target_fpr: float,
    *,
    sign: int,
) -> float:
    """Fit a threshold with at most the requested empirical calibration FPR.

    Predictions at this operating point use ``sign * score > threshold``.
    """
    if sign not in (-1, 1):
        raise ValueError("sign must be -1 or +1")
    if not 0.0 <= target_fpr < 1.0:
        raise ValueError("target_fpr must be in [0, 1)")
    neg = np.sort(sign * _scores(calibration_negative, "calibration_negative"))
    if len(neg) == 0:
        raise ValueError("calibration_negative must be non-empty")
    allowed_false_positives = int(np.floor(target_fpr * len(neg) + 1e-12))
    index = max(0, len(neg) - allowed_false_positives - 1)
    return float(neg[index])


def evaluate_fixed_orientation(
    test_positive: np.ndarray | Sequence[float],
    test_negative: np.ndarray | Sequence[float],
    *,
    sign: int,
) -> dict[str, float | int]:
    """Evaluate test AUROC/AP with a sign fixed before seeing test labels."""
    if sign not in (-1, 1):
        raise ValueError("sign must be -1 or +1")
    pos = _scores(test_positive, "test_positive")
    neg = _scores(test_negative, "test_negative")
    raw_auc = auroc(pos, neg)
    return {
        "auroc_raw": raw_auc,
        "auroc_oriented": auroc(sign * pos, sign * neg),
        "sign": sign,
        "aupr": average_precision(sign * pos, sign * neg),
    }


def evaluate_calibrated_binary(
    calibration_positive: np.ndarray | Sequence[float],
    calibration_negative: np.ndarray | Sequence[float],
    test_positive: np.ndarray | Sequence[float],
    test_negative: np.ndarray | Sequence[float],
    *,
    target_tpr: float = 0.95,
    target_fpr: float = 0.05,
    tie_sign: int = 1,
) -> dict[str, float | int]:
    """Fit sign/operating points on calibration and evaluate once on test."""
    cal_pos = _scores(calibration_positive, "calibration_positive")
    cal_neg = _scores(calibration_negative, "calibration_negative")
    test_pos = _scores(test_positive, "test_positive")
    test_neg = _scores(test_negative, "test_negative")
    sign = fit_orientation(cal_pos, cal_neg, tie_sign=tie_sign)
    tpr_threshold = threshold_for_tpr(cal_pos, target_tpr, sign=sign)
    fpr_threshold = threshold_for_fpr(cal_neg, target_fpr, sign=sign)
    result = evaluate_fixed_orientation(test_pos, test_neg, sign=sign)
    result.update(
        {
            "calibration_auroc_raw": auroc(cal_pos, cal_neg),
            "threshold_at_target_tpr": tpr_threshold,
            "test_tpr_at_calibrated_tpr": float(np.mean(sign * test_pos >= tpr_threshold)),
            "test_fpr_at_calibrated_tpr": float(np.mean(sign * test_neg >= tpr_threshold)),
            "threshold_at_target_fpr": fpr_threshold,
            "test_tpr_at_calibrated_fpr": float(np.mean(sign * test_pos > fpr_threshold)),
            "test_fpr_at_calibrated_fpr": float(np.mean(sign * test_neg > fpr_threshold)),
        }
    )
    return result


def bootstrap_indices(
    n_samples: int,
    seed: int | Sequence[int],
) -> np.ndarray:
    """Return one deterministic, evaluation-only bootstrap resample."""
    if n_samples < 1:
        raise ValueError("cannot bootstrap an empty test set")
    return np.random.default_rng(seed).integers(0, n_samples, n_samples)
