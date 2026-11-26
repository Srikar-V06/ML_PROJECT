import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.audit.missingness import study
from src.audit.schema import classify, select_inputs
from src.data.splits import load
from src.models.baselines import model_frame, sanity
from src.utils.io import artifact, cli_config, write_json
from src.utils.logging import log


def screen(s, y, seed):
    if s.nunique(dropna=False) < 2:
        return .5
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
        values = model_frame(s.to_frame()).iloc[:, 0]
        auc = roc_auc_score(y, values.fillna(values.median()))
        missing_auc = roc_auc_score(y, values.isna())
        return max(auc, 1-auc, missing_auc, 1-missing_auc)
    # Out-of-fold category means avoid label memorization for unique or rare categories.
    values = s.astype("string").fillna("__missing__").reset_index(drop=True)
    labels = np.asarray(y)
    pred = np.full(len(s), labels.mean())
    for fit, valid in StratifiedKFold(4, shuffle=True, random_state=seed).split(values, labels):
        group = pd.DataFrame({"v": values.iloc[fit], "y": labels[fit]}).groupby("v").y.agg(["sum", "size"])
        prior = labels[fit].mean()
        means = (group["sum"] + 20*prior)/(group["size"]+20)
        pred[valid] = values.iloc[valid].map(means).fillna(prior)
    auc = roc_auc_score(y, pred)
    return max(auc, 1-auc)


def compare(cfg, train, val, clean, post):
    rows, imports = [], None
    for name, cols in [("origination", clean), ("with_servicing", clean + post)]:
        x, v = (model_frame(df[cols]) for df in (train, val))
        for col in x.select_dtypes(exclude="number"):
            dtype = pd.CategoricalDtype(x[col].value_counts().loc[lambda counts: counts >= 25].index)
            x[col], v[col] = x[col].astype(dtype), v[col].astype(dtype)
        model = LGBMClassifier(n_estimators=120, num_leaves=31, importance_type="gain", verbosity=-1,
                               deterministic=True, force_col_wise=True, random_state=cfg["seed"], n_jobs=4)
        model.fit(x, train.target)
        p = model.predict_proba(v)[:, 1]
        rows.append(dict(features=name, roc_auc=roc_auc_score(val.target, p), pr_auc=average_precision_score(val.target, p)))
        if name == "with_servicing":
            imports = pd.Series(model.feature_importances_, index=cols).loc[post].sort_values(ascending=False)
    result = pd.DataFrame(rows).set_index("features")
    result.to_csv(artifact(cfg, "tables/leakage_lgbm_comparison.csv"))
    imports.rename("gain").to_csv(artifact(cfg, "tables/leaked_feature_importance.csv"), index_label="feature")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    result.plot.bar(ax=axes[0], rot=0, ylim=(0, 1.05), title="Validation discrimination")
    imports.head(6).sort_values().plot.barh(ax=axes[1], title="Servicing features by gain")
    fig.tight_layout()
    fig.savefig(artifact(cfg, "plots/leakage_comparison.png"))
    plt.close(fig)


def run(cfg):
    splits = load(cfg, names=["train", "val"])
    clean = [c for c in splits["train"] if c not in ("issue_d", "target")]
    extra = pd.read_parquet(cfg["paths"]["diagnostics"]).set_index("id", verify_integrity=True)
    train, val = (splits[key].join(extra, validate="one_to_one").reset_index() for key in ("train", "val"))
    rows = []
    for col in train.columns.drop("target"):
        stage, reason = classify(col)
        auc = screen(train[col], train.target, cfg["seed"])
        if col in clean and auc > .75:
            reason += " Flag for underwriting review: strong Train univariate discrimination."
        rows.append(dict(feature_name=col, dtype=str(train[col].dtype), missing_rate=train[col].isna().mean(),
                         stage=stage, decision="keep" if col in clean else "drop", single_feature_auc=auc, reason=reason))
    table = pd.DataFrame(rows)
    if cfg.get("model_features") == 50:
        selected = select_inputs(train)
        reserve = table.decision.eq("keep") & ~table.feature_name.isin(selected)
        table.loc[reserve, "decision"] = "reserve"
        table.loc[reserve, "reason"] += " Reserved outside the 50-field training-coverage selection."
    table.to_csv(artifact(cfg, "feature_audit.csv"), index=False)
    keep = table.query("decision == 'keep'")
    assert len(keep) >= 50 and keep.stage.eq("origination_time").all()
    assert keep.loc[~keep.feature_name.isin(cfg["auc_allowlist"]), "single_feature_auc"].max() <= .85
    study(train, cfg)
    write_json(artifact(cfg, "metrics/overfit_sanity.json"), sanity(model_frame(train[clean].iloc[:1000]), train.target, cfg["seed"]))
    compare(cfg, train, val, keep.feature_name.tolist(), table.loc[table.stage.eq("post_origination_leakage"), "feature_name"].tolist())
    log(cfg, "audit", len(train), len(clean))


if __name__ == "__main__":
    run(cli_config())
