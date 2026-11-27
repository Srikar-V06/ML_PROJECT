"""Block 4 (P4) tests. CI-safe: synthetic data plus tracked artifacts only (no data/ directory needed)."""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.special import expit

from src import block4
from src.evaluation import calibration as cal
from src.evaluation.metrics import evaluate
from src.utils.io import digest


def synthetic(n=20_000, shift=1.5, seed=0):
    """True probabilities q; scores p are q pushed up in log-odds (like class-weighted training)."""
    rng = np.random.default_rng(seed)
    z = rng.normal(-1.4, 1.0, n)
    y = (rng.random(n) < expit(z)).astype(int)
    return y, expit(z + shift)


def test_platt_removes_prior_odds_shift():
    y, p = synthetic()
    fitted = cal.Platt().fit(p, y)
    assert fitted.slope_ == pytest.approx(1.0, abs=0.05)
    assert fitted.intercept_ == pytest.approx(-1.5, abs=0.1)
    assert cal.ece(y, fitted.transform(p)) < 0.01 < cal.ece(y, p)


def test_isotonic_is_monotone_and_bounded():
    y, p = synthetic()
    q = cal.Isotonic().fit(p, y).transform(np.sort(p))
    assert np.all(np.diff(q) >= 0) and q.min() > 0 and q.max() < 1


def test_ece_matches_evaluate_definition():
    y, p = synthetic(shift=0.4)
    result = evaluate(y, p, "val", "synthetic", None, n_bootstrap=1)
    assert cal.ece(y, p) == pytest.approx(result["ece"])


def test_cross_fit_is_out_of_fold_and_complete():
    y, p = synthetic(n=5_000)
    oof = cal.cross_fit(y, p, "isotonic", folds=5, seed=1)
    in_sample = cal.Isotonic().fit(p, y).transform(p)
    assert oof.shape == p.shape and np.isfinite(oof).all()
    assert not np.allclose(oof, in_sample)


def test_bayes_threshold_rule():
    assert cal.bayes_thresholds([5, 10, 15]) == {"5.0": 1 / 6, "10.0": 1 / 11, "15.0": 1 / 16}


def test_summary_reports_bayes_and_tuned_costs():
    y, p = synthetic(shift=0.0)
    row = cal.summary(y, p, [10.0], tuned_thresholds={"10.0": 0.2})
    assert {"bayes_cost_10", "tuned_cost_10", "ece", "cal_slope"} <= row.keys()
    assert row["cal_slope"] == pytest.approx(1.0, abs=0.08)


@pytest.mark.skipif(block4.LOCK.exists(), reason="Test read-out already locked")
def test_test_predictions_blocked_before_lock():
    with pytest.raises(RuntimeError):
        block4.predictions("catboost", "test")


def test_p3_freeze_untouched():
    lock = json.loads(Path("artifacts/p3/test_lock.json").read_text())
    assert lock["model_count"] == 26
    assert digest("artifacts/p3/model_manifest.json") == lock["manifest_sha256"]


@pytest.mark.skipif(not block4.LOCK.exists(), reason="Test read-out not yet performed")
def test_p4_lock_matches_test_predictions():
    lock = json.loads(block4.LOCK.read_text())
    for name, sha in lock["p3_test_predictions_sha256"].items():
        assert digest(f"artifacts/p3/predictions/{name}_test.parquet") == sha
    assert digest("artifacts/p3/test_lock.json") == lock["p3_test_lock_sha256"]
