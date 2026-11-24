import numpy as np
from scipy.stats import norm, rankdata

NAMES = ("roc_auc", "pr_auc", "brier", "ks")


def inputs(y, p):
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or set(np.unique(y)) != {0, 1}:
        raise ValueError("Expected aligned one-dimensional scores and both binary classes")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Expected finite probabilities in [0, 1]")
    return y.astype(int), p


class Ranked:
    """Cache sorting and tie blocks, then apply paired bootstrap multiplicities."""
    def __init__(self, y, p):
        self.y, self.p = inputs(y, p)
        self.order = np.argsort(-self.p, kind="stable")
        score = self.p[self.order]
        self.end = np.flatnonzero(np.r_[score[:-1] != score[1:], True])
        self.labels = self.y[self.order]
        self.squared = (self.p-self.y)**2

    def metrics(self, weights):
        w = weights[self.order]
        tp = np.cumsum(w*self.labels)[self.end]
        fp = np.cumsum(w*(1-self.labels))[self.end]
        tpr, fpr = tp/tp[-1], fp/fp[-1]
        auc = np.trapezoid(np.r_[0, tpr], np.r_[0, fpr])
        precision = np.divide(tp, tp+fp, out=np.zeros(len(tp)), where=tp+fp > 0)
        ap = np.sum(np.diff(tpr, prepend=0)*precision)
        return np.array([auc, ap, np.sum(weights*self.squared)/weights.sum(), np.max(np.abs(tpr-fpr))])


def bootstrap(y, ranked, iterations, seed):
    if iterations < 1:
        raise ValueError("At least one bootstrap iteration is required")
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < iterations:
        weights = np.bincount(rng.integers(len(y), size=len(y)), minlength=len(y))
        if weights[y == 1].sum() and weights[y == 0].sum():
            out.append([r.metrics(weights) for r in ranked])
    return np.array(out)


def delong(y, a, b):
    pos = y == 1
    m, n = int(pos.sum()), int((~pos).sum())
    if min(m, n) < 2:
        raise ValueError("Paired DeLong needs at least two observations per class")
    effects = []
    for scores in (a, b):
        x, z = scores[pos], scores[~pos]
        ranks = rankdata(np.r_[x, z])
        effects.append(((ranks[:m]-rankdata(x))/n, 1-(ranks[m:]-rankdata(z))/m))
    dx, dz = (effects[0][i]-effects[1][i] for i in (0, 1))
    diff = float(dx.mean())
    se = np.sqrt(dx.var(ddof=1)/m + dz.var(ddof=1)/n)
    z = float(diff/se) if se else (None if diff else 0.0)
    return dict(difference=diff, z=z, p_value=float(2*norm.sf(abs(z))) if z is not None else 0.,
                ci95=[float(diff-1.959964*se), float(diff+1.959964*se)])


def compare_models(y_true, y_prob_a, y_prob_b, n_bootstrap=1000, seed=42):
    y, a = inputs(y_true, y_prob_a)
    _, b = inputs(y, y_prob_b)
    models = [Ranked(y, a), Ranked(y, b)]
    unit = np.ones(len(y), dtype=int)
    observed = models[0].metrics(unit) - models[1].metrics(unit)
    samples = bootstrap(y, models, n_bootstrap, seed)
    draws = samples[:, 0] - samples[:, 1]
    paired = {}
    for i, name in enumerate(NAMES[:3]):
        delta = draws[:, i]
        # Two-sided test from the null-centered paired bootstrap distribution.
        p = (1+np.count_nonzero(abs(delta-observed[i]) >= abs(observed[i])))/(len(delta)+1)
        paired[name] = dict(difference=float(observed[i]), ci95=np.quantile(delta, [.025, .975]).tolist(),
                            p_value=float(p))
    return {"delong": delong(y, a, b), "paired_bootstrap": paired}
