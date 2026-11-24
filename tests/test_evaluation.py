import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from src.evaluation.cost_sensitive import curve, select
from src.evaluation.metrics import evaluate
from src.evaluation.stats_tests import compare_models


def test_ties_metrics_thresholds_and_null(fixture_run):
    _, frames = fixture_run
    y = frames["val"].target.to_numpy()
    p = np.round(frames["val"].loan_amnt.to_numpy()/1000, 1)
    r = evaluate(y, p, "val", "tied", None, n_bootstrap=20)
    assert r["roc_auc"] == pytest.approx(roc_auc_score(y, p))
    assert r["pr_auc"] == pytest.approx(average_precision_score(y, p))
    fpr, tpr, _ = roc_curve(y, p)
    assert r["ks"] == pytest.approx(max(abs(tpr-fpr)))
    assert r["brier"] == pytest.approx(np.mean((p-y)**2))
    for ratio in (5., 10., 15.):
        t, costs = curve(y, p, ratio)
        brute = [1000*np.mean(ratio*((y == 1) & (p < v)) + ((y == 0) & (p >= v))) for v in t]
        assert np.allclose(costs, brute)
        assert select(y, p, [ratio])[str(ratio)] == t[np.argmin(costs)]
    with pytest.raises(ValueError, match="validation thresholds"):
        evaluate(y, p, "test", "bad", None, n_bootstrap=2)
    same = compare_models(y, p, p, n_bootstrap=20)
    assert same["delong"]["p_value"] == 1
    assert same["paired_bootstrap"]["brier"]["ci95"] == [0, 0]


def test_delong_against_pairwise_influence(fixture_run):
    _, frames = fixture_run
    f = frames["val"]
    y = f.target.to_numpy()
    a, b = f.loan_amnt.to_numpy()/1000, f.int_rate.to_numpy(dtype=float)/100
    def placements(p):
        diff = p[y == 1, None]-p[None, y == 0]
        return (diff > 0) + .5*(diff == 0)
    pair = placements(a)-placements(b)
    se = np.sqrt(pair.mean(1).var(ddof=1)/pair.shape[0]+pair.mean(0).var(ddof=1)/pair.shape[1])
    observed = compare_models(y, a, b, 10)["delong"]
    assert observed["difference"] == pytest.approx(pair.mean())
    assert observed["z"] == pytest.approx(pair.mean()/se)
