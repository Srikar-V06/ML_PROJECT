"""Picklable ensemble wrappers for Block 3 (simple average and forward-chained stacker).

Members are complete fitted pipelines that all consume the same raw 50-field model frame.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

EPS = 1e-6


def to_logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


class AveragingEnsemble(ClassifierMixin, BaseEstimator):
    """Unweighted mean of member default probabilities (no fitting)."""

    def __init__(self, members=None):
        self.members = members  # list of (name, fitted_pipeline)
        self.classes_ = np.array([0, 1])

    def fit(self, X=None, y=None):
        return self

    def member_probabilities(self, X):
        return np.column_stack([m.predict_proba(X)[:, 1] for _, m in self.members])

    def predict_proba(self, X):
        p = self.member_probabilities(X).mean(axis=1)
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class StackedEnsemble(ClassifierMixin, BaseEstimator):
    """Logistic meta-learner over member logits.

    The meta-learner is fitted externally on strictly out-of-fold, out-of-time member
    scores; members are the full-Train refits.
    """

    def __init__(self, members=None, meta=None):
        self.members = members
        self.meta = meta
        self.classes_ = np.array([0, 1])

    def fit(self, X=None, y=None):
        return self

    def member_logits(self, X):
        return np.column_stack([to_logit(m.predict_proba(X)[:, 1]) for _, m in self.members])

    def predict_proba(self, X):
        return self.meta.predict_proba(self.member_logits(X))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
