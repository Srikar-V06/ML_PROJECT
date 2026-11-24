"""Train-only WoE binning with additive log-odds points."""
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.models.registry import register_model
from src.utils.io import artifact

BASE_SCORE, BASE_GOOD_BAD_ODDS, PDO = 600, 10, 20
FACTOR = PDO / np.log(2)
OFFSET = BASE_SCORE - FACTOR * np.log(BASE_GOOD_BAD_ODDS)


class Binner:
    def fit(self, x, y):
        self.numeric = pd.api.types.is_numeric_dtype(x)
        self.optimal = None
        observed = x.notna()
        self.backend = "monotonic_quantile" if self.numeric else "pooled_categories"
        if observed.any() and x[observed].nunique() > 1:
            try:
                from optbinning import OptimalBinning
                self.optimal = OptimalBinning(dtype="numerical" if self.numeric else "categorical",
                                              monotonic_trend="auto", time_limit=10)
                self.optimal.fit(x.to_numpy(), np.asarray(y))
                if self.optimal.status not in ("OPTIMAL", "FEASIBLE"):
                    raise ValueError(self.optimal.status)
                self.bins(x)
                self.backend = "optbinning"
            except (ImportError, ValueError, RuntimeError, TypeError, ArithmeticError):
                self.optimal = None
        if self.optimal is None:
            if self.numeric:
                values = x[observed]
                candidates = np.unique(values.quantile(np.linspace(0, 1, 11)).dropna())[1:-1]
                ix = np.searchsorted(candidates, values, side="right")
                rates = pd.DataFrame({"ix": ix, "y": np.asarray(y)[observed]}).groupby("ix").y.agg(["mean", "size"])
                self.edges = np.array([])
                if len(rates) > 1:
                    direction = np.cov(rates.index, rates["mean"], aweights=rates["size"])[0, 1] >= 0
                    levels = IsotonicRegression(increasing=bool(direction)).fit_transform(
                        rates.index, rates["mean"], sample_weight=rates["size"])
                    changes = rates.index.to_numpy()[1:][np.diff(levels) != 0]
                    self.edges = candidates[changes-1]
            else:
                counts = x.value_counts()
                self.frequent = set(counts[counts >= max(20, .005*len(x))].index)
        labels = self.bins(x)
        counts = pd.DataFrame({"bin": labels.to_numpy(), "bad": np.asarray(y)}).groupby("bin").bad.agg(["sum", "size"])
        good, bad = counts["size"]-counts["sum"]+.5, counts["sum"]+.5
        good, bad = good/good.sum(), bad/bad.sum()
        self.woe = np.log(good/bad).to_dict()
        self.iv = float(((good-bad)*np.log(good/bad)).sum())
        return self

    def bins(self, x):
        if self.optimal is not None:
            return pd.Series(self.optimal.transform(x.to_numpy(), metric="bins", show_digits=8),
                             index=x.index).astype(str)
        if self.numeric:
            groups = pd.cut(x, np.r_[-np.inf, self.edges, np.inf], right=False, precision=12)
            return groups.astype("string").fillna("<missing>")
        return x.where(x.isin(self.frequent), "<other>").where(x.notna(), "<missing>").astype(str)

    def transform(self, x):
        return self.bins(x).map(self.woe).fillna(0.).astype(float)


@register_model("woe_scorecard")
class Scorecard:
    def __init__(self, X, flags, cfg):
        self.cfg = cfg

    def fit(self, X, y):
        self.binners = {col: Binner().fit(X[col], y) for col in X}
        summary = pd.DataFrame([dict(feature=c, iv=b.iv, backend=b.backend, selected=b.iv >= .02)
                                for c, b in self.binners.items()]).sort_values("iv", ascending=False)
        summary.to_csv(artifact(self.cfg, "tables/woe_iv_summary.csv"), index=False)
        self.features = summary.loc[summary.selected, "feature"].tolist()
        if not self.features:
            raise ValueError("No features meet IV floor")
        self.model = LogisticRegression(max_iter=800, tol=1e-6)
        self.model.fit(self.transform(X), y)
        rows = [dict(feature="(base)", bin="all", woe=0., coefficient=self.model.intercept_[0],
                     points=OFFSET - FACTOR*self.model.intercept_[0])]
        for col, coeff in zip(self.features, self.model.coef_[0]):
            for label, woe in self.binners[col].woe.items():
                rows.append(dict(feature=col, bin=label, woe=woe, coefficient=coeff, points=-FACTOR*coeff*woe))
            rows.append(dict(feature=col, bin="<unseen fallback>", woe=0., coefficient=coeff, points=0.))
        pd.DataFrame(rows).to_csv(artifact(self.cfg, "tables/credit_scorecard_points.csv"), index=False)
        return self

    def transform(self, X):
        return pd.DataFrame({c: self.binners[c].transform(X[c]) for c in self.features})

    def predict_proba(self, X):
        return self.model.predict_proba(self.transform(X))

    def points(self, X):
        return OFFSET - FACTOR*self.model.decision_function(self.transform(X))
