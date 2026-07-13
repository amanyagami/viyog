import numpy as np

from viyog import viyog_metrics


def test_viyog_metrics_basic_separation() -> None:
    # generate separable scores: ID centered near 0, OOD higher values
    id_scores = np.random.normal(loc=0.0, scale=0.5, size=500)
    ood_scores = np.random.normal(loc=3.0, scale=0.5, size=200)

    metrics = viyog_metrics(id_scores, ood_scores, recall_level=0.95)
    # basic expected keys exist
    for k in ("AUROC", "AUPR_IN", "AUPR_OUT", "FPR95", "DetectionError", "AUTC", "AUTC_components"):
        assert k in metrics

    # AUROC should be well above chance for clearly separable scores
    assert 0.0 <= metrics["AUROC"] <= 1.0
    assert metrics["AUROC"] > 0.9

    # FPR95 is a rate between 0 and 1
    assert 0.0 <= metrics["FPR95"] <= 1.0


def test_autc_is_finite() -> None:
    # Regression: sklearn's roc_curve prepends an infinite threshold, which used
    # to make the AUTC trapezoidal integral NaN. It must be finite.
    rng = np.random.default_rng(0)
    m = viyog_metrics(rng.normal(0, 1, 300), rng.normal(2, 1, 300))
    assert np.isfinite(m["AUTC"])
    assert np.isfinite(m["AUTC_components"]["AUFPR"])
    assert np.isfinite(m["AUTC_components"]["AUFNR"])
