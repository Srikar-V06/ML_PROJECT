"""Matched PCA experiments with a common training representation and validation cohort."""
from copy import copy

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

from src.analysis.block2_trees import save_plot
from src.features.block2 import FEATURE_GROUPS, build_preprocessor
from src.models.block2 import estimator
from src.utils.io import artifact


def save_validation(name, pipeline, y, prob, cfg):
    joblib.dump(pipeline, artifact(cfg, f"block2/models/{name}.joblib"), compress=3)
    pred = pd.DataFrame(dict(loan_id=y.index, y_true=y.to_numpy(dtype=np.int8),
                             y_prob=np.asarray(prob, dtype=float), split="val"))
    pred.to_parquet(artifact(cfg, f"predictions/{name}_val.parquet"), index=False)
    return dict(model=name, roc_auc=roc_auc_score(y, prob), pr_auc=average_precision_score(y, prob))


def pca_study(train, val, flags, forest_params, cfg):
    X, y = train
    V, vy = val
    prep = build_preprocessor("linear", flags)
    x, v = prep.fit_transform(X), prep.transform(V)
    full = PCA(svd_solver="covariance_eigh").fit(x)
    variance = full.explained_variance_ratio_.cumsum()
    fig, ax = plt.subplots()
    components = np.arange(1, len(variance) + 1)
    ax.bar(components, full.explained_variance_ratio_, label="Individual")
    ax.plot(components, variance, label="Cumulative")
    ax.set(xlabel="Principal component", ylabel="Fraction of Train variance")
    ax.legend()
    save_plot(fig, cfg, "scree")
    weights = pd.DataFrame(full.components_[:3].T, index=prep.get_feature_names_out(),
                           columns=["PC1", "PC2", "PC3"])
    weights.to_csv(artifact(cfg, "tables/block2_pc_weights.csv"), index_label="feature")
    # Match encoded levels to the longest raw name (e.g. annual_inc_joint before annual_inc).
    sources = sorted(((field, group) for group, fields in FEATURE_GROUPS.items() for field in fields),
                     key=lambda item: -len(item[0]))
    groups = []
    for encoded in weights.index:
        prefix, field = encoded.split("__", 1)
        if prefix == "missing":
            groups.append("missing indicators")
        else:
            groups.append(next(group for raw, group in sources if field == raw or field.startswith(raw + "_")))
    notes = []
    for pc, w in weights.items():
        leading = w.loc[w.abs().nlargest(4).index]
        terms = ", ".join(f"{col} ({weight:+.2f})" for col, weight in leading.items())
        shares = w.pow(2).groupby(groups).sum()
        notes.append(f"{pc} explains {full.explained_variance_ratio_[int(pc[-1])-1]:.1%} "
                     f"of training variance. Its largest signed axis weights are {terms}. "
                     f"The {shares.idxmax()} group contributes the largest share "
                     f"({shares.max():.1%} of squared axis weights). "
                     "Positive weights move the score up as that encoded input increases, "
                     "and negative weights move it down. The overall axis sign is arbitrary.")

    rows = []
    for percent in (None, 80, 90, 95, 99):
        steps = [("preprocess", prep)]
        a, b = x, v
        if percent is not None:
            k = min(len(variance), int(np.searchsorted(variance, percent / 100)) + 1)
            # Reuse the same fitted axes. Each persisted transformer retains just its prefix.
            reduced = copy(full)
            reduced.n_components = reduced.n_components_ = k
            for attr in ("components_", "explained_variance_", "explained_variance_ratio_", "singular_values_"):
                setattr(reduced, attr, getattr(full, attr)[:k])
            reduced.noise_variance_ = full.explained_variance_[k:].mean() if k < len(variance) else 0.
            a, b = reduced.transform(x), reduced.transform(v)
            steps.append(("pca", reduced))
        for family in ("lr", "random_forest", "lightgbm"):
            if family == "lr":
                model = LogisticRegression(max_iter=1000, random_state=cfg["seed"])
            else:
                model = estimator(family, cfg["seed"])
                if family == "random_forest":
                    model.set_params(**forest_params)
            model.fit(a, y)
            name = f"pca_{family}_{percent or 'control'}"
            pipeline = Pipeline([*steps, ("model", model)])
            row = save_validation(name, pipeline, vy, model.predict_proba(b)[:, 1], cfg)
            rows.append(dict(**row, family=family, representation=str(percent or "control"),
                             input_columns=a.shape[1]))
    results = pd.DataFrame(rows)
    controls = results.query("representation == 'control'").set_index("family").roc_auc
    results["auc_change"] = results.roc_auc - results.family.map(controls)
    results.to_csv(artifact(cfg, "tables/block2_pca.csv"), index=False)
    for family, group in results.query("representation != 'control'").groupby("family"):
        deltas = ", ".join(f"{row.representation}% {row.auc_change:+.4f}" for row in group.itertuples())
        notes.append(f"{family} validation AUC changes relative to its matched control were {deltas}.")
    notes.append("PCA retains high-variance directions without using the default label. A useful income "
                 "or delinquency cutoff can span several rotated axes, requiring more tree splits. "
                 "The matched validation changes show whether this affected this cohort.")
    artifact(cfg, "block2/pca_findings.md").write_text("\n\n".join(notes) + "\n", encoding="utf-8")
