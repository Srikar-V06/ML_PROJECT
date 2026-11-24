from pathlib import Path

import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from src.evaluation.cost_sensitive import policies, select
from src.evaluation.stats_tests import NAMES, Ranked, bootstrap, inputs
from src.utils.io import write_json


def evaluate(y_true, y_prob, split_name, model_name, out_dir, cost_ratios=(5.0, 10.0, 15.0),
             val_thresholds=None, n_bootstrap=1000, seed=42):
    y, p = inputs(y_true, y_prob)
    rank = Ranked(y, p)
    measures = dict(zip(NAMES, map(float, rank.metrics(np.ones(len(y), dtype=int)))))
    bins = np.minimum((10*p).astype(int), 9)
    ece = sum(abs((p[bins == i] - y[bins == i]).sum()) for i in range(10))/len(y)
    slope = intercept = None
    if np.ptp(p) > 0:
        lr = LogisticRegression(penalty=None, tol=1e-9)
        lr.fit(logit(p.clip(1e-12, 1-1e-12)).reshape(-1, 1), y)
        slope, intercept = float(lr.coef_[0, 0]), float(lr.intercept_[0])
    draws = bootstrap(y, [rank], n_bootstrap, seed)[:, 0]
    if val_thresholds is None:
        if split_name.lower() not in ("val", "validation"):
            raise ValueError("Frozen validation thresholds are required for other splits")
        val_thresholds = select(y, p, cost_ratios)
    result = dict(model=model_name, split=split_name, rows=len(y), **measures, log_loss=float(log_loss(y, p)),
                  ece=float(ece), calibration_slope=slope, calibration_intercept=intercept,
                  ci95={name: np.quantile(draws[:, i], [.025, .975]).tolist() for i, name in enumerate(NAMES)},
                  thresholds=val_thresholds, cost_per_1000=policies(y, p, val_thresholds))
    if out_dir is not None:
        write_json(Path(out_dir) / f"{model_name}_{split_name}.json", result)
    return result
