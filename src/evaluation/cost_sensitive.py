import numpy as np


def curve(y, p, ratio):
    """Evaluate every distinct decision rule, including reject-none, in descending threshold order."""
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    order = np.argsort(-p, kind="stable")
    scores, labels = p[order], y[order]
    end = np.r_[scores[:-1] != scores[1:], True]
    tp = np.r_[0, labels.cumsum()[end]]
    fp = np.r_[0, (1-labels).cumsum()[end]]
    thresholds = np.r_[np.nextafter(scores[0], np.inf), scores[end]]
    return thresholds, (ratio*(y.sum()-tp) + fp)*1000/len(y)


def select(y, p, ratios):
    answer = {}
    for ratio in ratios:
        thresholds, costs = curve(y, p, ratio)
        answer[str(float(ratio))] = float(thresholds[np.argmin(costs)])
    return answer


def policies(y, p, thresholds):
    y, p = np.asarray(y), np.asarray(p)
    result = {}
    for key, t in thresholds.items():
        ratio = float(key)
        row = {"threshold": t}
        for name, rejected in [("optimized", p >= t), ("naive_050", p >= .5),
                               ("approve_all", np.zeros(len(p), dtype=bool))]:
            cost = np.sum(rejected & (y == 0)) + ratio*np.sum(~rejected & (y == 1))
            row[name] = float(cost * 1000 / len(y))
        row["approval_rate"] = float((p < t).mean())
        result[key] = row
    return result
