"""Post-hoc probability calibration (Block 4, Step 10).

Calibrators are fitted on Validation predictions only. Both maps are monotone, so they leave
ranking (ROC-AUC) unchanged (isotonic can only merge ties). Their value shows up when decisions use the
Bayes cost rule t* = 1 / (1 + R), which is only optimal for calibrated probabilities.
"""
import numpy as np
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

from src.evaluation.cost_sensitive import policies
from src.evaluation.stats_tests import NAMES, Ranked, inputs

EPS = 1e-6
METHODS = ("raw", "platt", "isotonic")


def _logit(p):
    return logit(np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)).reshape(-1, 1)


class Identity:
    method = "raw"

    def fit(self, p, y):
        return self

    def transform(self, p):
        return np.asarray(p, dtype=float)


class Platt:
    """Logistic regression on the logit of the score: p' = sigmoid(a * logit(p) + b)."""
    method = "platt"

    def fit(self, p, y):
        lr = LogisticRegression(penalty=None, tol=1e-9, max_iter=1000).fit(_logit(p), np.asarray(y))
        self.slope_, self.intercept_ = float(lr.coef_[0, 0]), float(lr.intercept_[0])
        return self

    def transform(self, p):
        return np.clip(expit(self.slope_ * _logit(p)[:, 0] + self.intercept_), EPS, 1 - EPS)


class Isotonic:
    """Monotone step function; outputs are clipped away from 0/1 so log-loss stays finite."""
    method = "isotonic"

    def fit(self, p, y):
        self.iso_ = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(
            np.asarray(p, dtype=float), np.asarray(y, dtype=float))
        return self

    def transform(self, p):
        return np.clip(self.iso_.predict(np.asarray(p, dtype=float)), EPS, 1 - EPS)


def make(method):
    return {"raw": Identity, "platt": Platt, "isotonic": Isotonic}[method]()


def cross_fit(y, p, method, folds=5, seed=42):
    """Out-of-fold calibrated scores, so Validation metrics are not judged on the rows that fitted the map."""
    y, p = inputs(y, p)
    out = np.empty(len(y))
    for fit_idx, oof_idx in StratifiedKFold(folds, shuffle=True, random_state=seed).split(p, y):
        out[oof_idx] = make(method).fit(p[fit_idx], y[fit_idx]).transform(p[oof_idx])
    return out


def ece(y, p, bins=10):
    """Same definition as src.evaluation.metrics.evaluate: 10 equal-width bins, |sum(p - y)| / n."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    idx = np.minimum((bins * p).astype(int), bins - 1)
    return float(sum(abs((p[idx == i] - y[idx == i]).sum()) for i in range(bins)) / len(y))


def slope_intercept(y, p):
    lr = LogisticRegression(penalty=None, tol=1e-9, max_iter=1000).fit(_logit(p), np.asarray(y))
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def bayes_thresholds(ratios):
    """Reject when expected loss of approving exceeds that of rejecting: R * p > (1 - p)  =>  p > 1 / (1 + R)."""
    return {str(float(r)): 1.0 / (1.0 + float(r)) for r in ratios}


def summary(y, p, ratios, tuned_thresholds=None):
    """Discrimination, probability-quality and cost metrics for one score vector."""
    y, p = inputs(y, p)
    row = dict(zip(NAMES, map(float, Ranked(y, p).metrics(np.ones(len(y), dtype=int)))))
    slope, intercept = slope_intercept(y, p)
    row.update(log_loss=float(log_loss(y, p)), ece=ece(y, p), cal_slope=slope, cal_intercept=intercept,
               mean_p=float(p.mean()), base_rate=float(y.mean()))
    for policy, thresholds in (("bayes", bayes_thresholds(ratios)), ("tuned", tuned_thresholds)):
        if thresholds is None:
            continue
        for key, res in policies(y, p, thresholds).items():
            r = int(float(key))
            row[f"{policy}_thr_{r}"] = res["threshold"]
            row[f"{policy}_cost_{r}"] = res["optimized"]
            row[f"{policy}_approval_{r}"] = res["approval_rate"]
    return row
