"""Analysis D - temporal stability and sensitivity (Block 4).

Temporal: monthly (issue_d vintage) discrimination and calibration from the frozen prediction files, plus
feature and score PSI from Train. Validation is used during development; the Test months are produced by
`--final` from P3's existing Test predictions (nothing is re-scored).

Sensitivity: frozen pipelines re-score perturbed copies of the Validation features (no retraining):
missingness injection, Gaussian noise, and an interest-rate shock that re-prices `installment` with the
amortization formula so the shocked borrower stays internally consistent.
"""
import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.data.splits import load  # noqa: E402
from src.evaluation.calibration import ece  # noqa: E402
from src.evaluation.stats_tests import Ranked  # noqa: E402
from src.models.baselines import model_frame  # noqa: E402

P3, P4 = Path("artifacts/p3"), Path("artifacts/p4")
MODELS = ["logreg_baseline", "xgb_hist", "catboost", "lgbm_native_flags", "lgbm_monotone", "stack_lr"]
MISSING_FRACS = (0.05, 0.10, 0.20)
NOISE_SIGMAS = (0.05, 0.10, 0.25)
RATE_SHOCKS = (1.0, 2.0, 3.0)  # percentage points


def manifest():
    return json.loads((P3 / "model_manifest.json").read_text())


def basic_metrics(y, p):
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    roc, pr, brier, _ = Ranked(y, p).metrics(np.ones(len(y), dtype=int))
    return dict(rows=len(y), base_rate=float(y.mean()), mean_p=float(p.mean()), roc_auc=float(roc),
                pr_auc=float(pr), brier=float(brier), ece=ece(y, p))


# ----------------------------------------------------------------------------- perturbations (pure functions)

def numeric_columns(X):
    return [c for c in X.columns if X[c].dtype.kind in "fi"]


def inject_missing(X, frac, rng, columns=None):
    """Set a random `frac` of observed values to NaN in each numeric column."""
    out = X.copy()
    for c in columns or numeric_columns(X):
        mask = rng.random(len(out)) < frac
        out.loc[mask, c] = np.nan
    return out


def continuous_columns(X, min_levels=11):
    """Numeric fields with more than 10 distinct values. Two-level fields such as `term` (36/60) are
    excluded: Gaussian noise on them is not a meaningful perturbation (xgb_hist splits term at exactly
    60, so tiny noise flips 60-month loans into the 36-month branch; LightGBM splits at 48)."""
    return [c for c in numeric_columns(X) if X[c].nunique(dropna=True) >= min_levels]


def gaussian_noise(X, sigma, rng, columns=None):
    """Add N(0, (sigma * column std)^2) noise; clip to the column's observed range to stay in support."""
    out = X.copy()
    for c in columns or continuous_columns(X):
        x = out[c].to_numpy(dtype=float)
        sd = np.nanstd(x)
        if not np.isfinite(sd) or sd == 0:
            continue
        noisy = x + rng.normal(0.0, sigma * sd, len(x))
        out[c] = np.clip(noisy, np.nanmin(x), np.nanmax(x))
    return out


def amortized_installment(loan_amnt, term_months, annual_rate_pct):
    r = np.asarray(annual_rate_pct, dtype=float) / 1200.0
    n = np.asarray(term_months, dtype=float)
    return np.asarray(loan_amnt, dtype=float) * r / (1 - (1 + r) ** -n)


def rate_shock(X, points, reprice=True):
    """Raise int_rate by `points` percentage points; optionally recompute installment consistently."""
    out = X.copy()
    out["int_rate"] = out["int_rate"] + points
    if reprice:
        out["installment"] = amortized_installment(out["loan_amnt"], out["term"], out["int_rate"])
    return out


def scenarios(X, seed):
    yield "baseline", 0.0, X
    for f in MISSING_FRACS:
        yield "missing", f, inject_missing(X, f, np.random.default_rng(seed))
    for s in NOISE_SIGMAS:
        yield "noise", s, gaussian_noise(X, s, np.random.default_rng(seed))
    yield "term_noise_only", 0.05, gaussian_noise(X, 0.05, np.random.default_rng(seed), columns=["term"])
    for d in RATE_SHOCKS:
        yield "rate_shock", d, rate_shock(X, d, reprice=True)
        yield "rate_shock_no_reprice", d, rate_shock(X, d, reprice=False)


# ----------------------------------------------------------------------------- PSI

def psi(expected, actual, bins=10):
    """Population stability index with Train-decile bins (numeric) or levels (categorical); NaN is its own bin."""
    e, a = pd.Series(expected), pd.Series(actual)
    if e.dtype.kind in "fi":
        edges = np.unique(np.nanquantile(e.to_numpy(dtype=float), np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        e = pd.cut(e, edges, include_lowest=True).astype(str)
        a = pd.cut(a, edges, include_lowest=True).astype(str)
    pe = e.astype(str).value_counts(normalize=True)
    pa = a.astype(str).value_counts(normalize=True)
    levels = pe.index.union(pa.index)
    pe, pa = pe.reindex(levels, fill_value=0) + 1e-6, pa.reindex(levels, fill_value=0) + 1e-6
    return float(((pa - pe) * np.log(pa / pe)).sum())


# ----------------------------------------------------------------------------- runners

def load_predictions(name, split):
    m = manifest()["models"][name]
    path = m["val_pred_path"] if split == "val" else P3 / "predictions" / f"{name}_test.parquet"
    return pd.read_parquet(path).set_index("loan_id")


def temporal(cfg, split="val"):
    frames = load(cfg, names=(split,))
    month = frames[split]["issue_d"].dt.to_period("M").astype(str)
    rows = []
    for name in MODELS:
        pred = load_predictions(name, split)
        pred = pred.join(month.rename("month"), how="inner")
        assert len(pred) == len(frames[split]), f"{name}: prediction/frame id mismatch"
        for mth, g in pred.groupby("month"):
            rows.append(dict(model=name, split=split, month=mth, **basic_metrics(g.y_true, g.y_prob)))
        rows.append(dict(model=name, split=split, month="all", **basic_metrics(pred.y_true, pred.y_prob)))
    table = pd.DataFrame(rows)
    table.to_csv(P4 / "tables" / f"temporal_stability_{split}.csv", index=False)
    temporal_plot(table[table.month != "all"], split)
    return table


def temporal_plot(t, split):
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for name, g in t.groupby("model", sort=False):
        axes[0].plot(g.month, g.roc_auc, marker="o", label=name)
        axes[1].plot(g.month, g.ece, marker="o", label=name)
        axes[2].plot(g.month, g.mean_p - g.base_rate, marker="o", label=name)
    base = t.drop_duplicates("month")
    ax2 = axes[0].twinx()
    ax2.bar(base.month, base.rows, alpha=0.12, color="grey")
    ax2.set_ylabel("loans per month", color="grey")
    axes[0].set(title="ROC-AUC by issue month", ylabel="ROC-AUC")
    axes[1].set(title="ECE by issue month", ylabel="ECE")
    axes[2].axhline(0, c="grey", lw=0.8)
    axes[2].set(title="Mean predicted - observed default rate", ylabel="calibration-in-the-large")
    handles, labels = axes[2].get_legend_handles_labels()
    for ax in axes:
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle(f"Temporal stability - {split}", fontsize=10)
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(P4 / "plots" / f"temporal_stability_{split}.png", dpi=140)
    plt.close(fig)


def drift(cfg, split="val"):
    """Feature PSI Train -> split for all 50 fields, plus score PSI for each model."""
    m = manifest()
    cols = m["feature_order"]
    frames = load(cfg, names=("train", split))
    train, other = model_frame(frames["train"][cols]), model_frame(frames[split][cols])
    # Bureau fields were not reported before 2012 (100% NaN 2007-2011), so total PSI mixes a reporting-regime
    # change with genuine drift; psi_observed compares only non-missing values.
    feat = pd.DataFrame([dict(field=c, split=split, psi=psi(train[c], other[c]),
                              psi_observed=psi(train[c].dropna(), other[c].dropna()),
                              missing_train=float(train[c].isna().mean()), missing_split=float(other[c].isna().mean()))
                         for c in cols])
    feat.sort_values("psi", ascending=False).to_csv(P4 / "tables" / f"feature_psi_{split}.csv", index=False)
    scores = []
    for name in MODELS:
        pipe = joblib.load(m["models"][name]["path"])
        p_train = pipe.predict_proba(train)[:, 1]
        p_other = load_predictions(name, split).loc[other.index, "y_prob"].to_numpy()
        scores.append(dict(model=name, split=split, score_psi=psi(p_train, p_other),
                           train_mean_p=float(p_train.mean()), split_mean_p=float(p_other.mean())))
    pd.DataFrame(scores).to_csv(P4 / "tables" / f"score_psi_{split}.csv", index=False)
    return feat, pd.DataFrame(scores)


def sensitivity(cfg):
    m = manifest()
    cols = m["feature_order"]
    val = load(cfg, names=("val",))["val"]
    X, y = model_frame(val[cols]), val["target"].to_numpy(dtype=int)
    pipes = {n: joblib.load(m["models"][n]["path"]) for n in MODELS}
    rows, base_p = [], {}
    for kind, level, Xs in scenarios(X, cfg["seed"]):
        for name, pipe in pipes.items():
            p = pipe.predict_proba(Xs)[:, 1]
            if kind == "baseline":
                base_p[name] = p
                saved = load_predictions(name, "val").loc[X.index]
                assert np.array_equal(saved.y_true.to_numpy(), y), f"{name}: label mismatch"
                gap = float(np.abs(saved.y_prob.to_numpy() - p).max())
                assert gap < 1e-6, f"{name}: re-scored Validation differs from frozen predictions by {gap:.2e}"
            cutoff = m["val_thresholds"][name]["10.0"]
            rows.append(dict(model=name, scenario=kind, level=level, **basic_metrics(y, p),
                             approval_rate_10=float((p < cutoff).mean()),
                             mean_abs_dp=float(np.abs(p - base_p[name]).mean())))
    table = pd.DataFrame(rows)
    base = table[table.scenario == "baseline"].set_index("model")
    for col in ("roc_auc", "pr_auc", "mean_p", "approval_rate_10", "ece"):
        table[f"d_{col}"] = table[col] - table.model.map(base[col])
    table.to_csv(P4 / "tables" / "sensitivity.csv", index=False)
    sensitivity_plot(table)
    return table


def sensitivity_plot(t):
    kinds = ["missing", "noise", "rate_shock", "rate_shock_no_reprice"]
    fig, axes = plt.subplots(2, len(kinds), figsize=(4 * len(kinds), 6.5), sharey="row")
    for j, kind in enumerate(kinds):
        sub = pd.concat([t[t.scenario == "baseline"], t[t.scenario == kind]])
        for name, g in sub.groupby("model", sort=False):
            g = g.sort_values("level")
            axes[0, j].plot(g.level, g.roc_auc, marker="o", label=name)
            axes[1, j].plot(g.level, g.approval_rate_10, marker="o", label=name)
        axes[0, j].set_title(kind.replace("_", " "), fontsize=10)
        axes[1, j].set_xlabel({"missing": "fraction set to NaN", "noise": "noise sd (x column sd)"}
                              .get(kind, "int_rate shock (pp)"))
    axes[0, 0].set_ylabel("ROC-AUC (Validation)")
    axes[1, 0].set_ylabel("approval rate @ frozen 10:1 cutoff")
    axes[0, -1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(P4 / "plots" / "sensitivity.png", dpi=140)
    plt.close(fig)
