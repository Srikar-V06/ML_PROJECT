"""Analysis F - interpretability (Block 4).

SHAP values are computed on a seeded Validation sample, in log-odds, and then aggregated from the
transformed model inputs (one-hot levels, missing-value flags) back to the 50 raw fields plus the 4
engineered ratios, so every model is explained in the same business vocabulary.

stack_lr combines member logits linearly:  logit(stack) = b + sum_k w_k * logit(member_k).
Each member logit is exactly its SHAP sum (TreeSHAP for the boosters, coef * (z - mean z) for the
logistic member), so sum_k w_k * phi_k(field) is an exact additive attribution of the Champion.
"""
import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import shap
from catboost import Pool
from scipy.special import logit
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.data.splits import load  # noqa: E402
from src.features.block2 import FEATURE_GROUPS  # noqa: E402
from src.models.baselines import model_frame  # noqa: E402

P3, P4 = Path("artifacts/p3"), Path("artifacts/p4")
TREE_MODELS = ["xgb_hist", "catboost", "lgbm_native_flags", "lgbm_monotone"]
MONOTONE = {"dti": 1, "int_rate": 1, "revol_util": 1, "loan_to_income": 1, "payment_to_income": 1,
            "fico_range_low": -1, "annual_inc": -1}
DEPENDENCE = ["int_rate", "fico_range_low", "dti", "annual_inc"]


# ----------------------------------------------------------------------------- attribution core

def field_of(name, fields):
    """Map a transformed column name to its raw field (or engineered ratio)."""
    base = name.split("__", 1)[1] if "__" in name else name
    base = base.removeprefix("missingindicator_")  # sklearn MissingIndicator naming
    if base in fields:
        return base
    matches = [f for f in fields if base.startswith(f + "_")]
    if not matches:
        raise KeyError(f"Cannot map transformed feature {name!r} to a field")
    return max(matches, key=len)


def aggregate(values, names, fields):
    """Sum SHAP columns per raw field -> DataFrame (rows x fields), preserving additivity."""
    frame = pd.DataFrame(values, columns=names)
    return frame.T.groupby([field_of(n, fields) for n in names]).sum().T


def unwrap(pipe):
    steps = pipe.named_steps
    if "model" in steps:
        return steps["preprocess"], steps["model"]
    return pipe[:-1], pipe[-1]  # logreg_baseline: (..., "logistic")


def explain(pipe, X, fields):
    """Return (field-level SHAP in log-odds, base value) for one fitted pipeline."""
    pre, model = unwrap(pipe)
    Z = pre.transform(X)
    names = list(pre.get_feature_names_out())
    kind = type(model).__name__
    if kind == "CatBoostClassifier":
        raw = model.get_feature_importance(Pool(Z, cat_features=model.get_cat_feature_indices()), type="ShapValues")
        values, base = raw[:, :-1], float(raw[0, -1])
    elif kind == "LogisticRegression":
        Z = np.asarray(Z.toarray() if hasattr(Z, "toarray") else Z, dtype=float)
        coef = model.coef_[0]
        values = coef * (Z - Z.mean(axis=0))
        base = float(model.intercept_[0] + coef @ Z.mean(axis=0))
    else:
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(Z)
        if isinstance(values, list):
            values = values[1]
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[:, :, 1]
        ev = np.atleast_1d(explainer.expected_value)
        base = float(ev[-1])
    return aggregate(values, names, fields), base


def check_additivity(pipe, X, phi, base, tol=1e-3):
    p = np.clip(pipe.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
    err = float(np.abs(phi.sum(axis=1).to_numpy() + base - logit(p)).max())
    if err > tol:
        raise AssertionError(f"SHAP additivity error {err:.2e} exceeds {tol}")
    return err


# ----------------------------------------------------------------------------- outputs

def group_of(field):
    for group, members in FEATURE_GROUPS.items():
        if field in members:
            return group
    return "other"


def bar_plot(imp, title, path, top=20):
    s = imp.sort_values(ascending=True).tail(top)
    fig, ax = plt.subplots(figsize=(6.5, 0.28 * len(s) + 1.2))
    ax.barh(s.index, s.values, color="#3b6ea8")
    ax.set_xlabel("Mean |SHAP| (log-odds)")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def beeswarm(phi, X, title, path, top=15):
    numeric = [c for c in phi.columns if c in X.columns and pd.api.types.is_numeric_dtype(X[c])]
    order = phi[numeric].abs().mean().sort_values(ascending=False).index[:top]
    exp = shap.Explanation(values=phi[order].to_numpy(), data=X[order].to_numpy(dtype=float),
                           feature_names=list(order))
    plt.figure()
    shap.plots.beeswarm(exp, max_display=top, show=False)
    plt.title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close("all")


def dependence_plot(phis, X, path):
    fig, axes = plt.subplots(1, len(DEPENDENCE), figsize=(4 * len(DEPENDENCE), 3.6))
    for ax, field in zip(axes, DEPENDENCE):
        x = X[field].to_numpy(dtype=float)
        hi = np.nanquantile(x, 0.99)
        keep = np.isfinite(x) & (x <= hi)
        for label, phi in phis.items():
            ax.scatter(x[keep], phi[field].to_numpy()[keep], s=3, alpha=0.35, label=label)
        ax.axhline(0, c="grey", lw=0.7)
        ax.set(xlabel=field, ylabel="SHAP (log-odds)" if field == DEPENDENCE[0] else None)
    axes[0].legend(markerscale=4, fontsize=8)
    fig.suptitle("Field-level SHAP dependence: unconstrained vs monotone LightGBM (Validation sample)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def waterfall(contrib, base, prob, title, path, top=10):
    s = contrib.reindex(contrib.abs().sort_values(ascending=False).index)
    shown = s.iloc[:top][::-1]
    rest = s.iloc[top:].sum()
    labels = [f"other {len(s) - top} fields"] + list(shown.index)
    vals = np.r_[rest, shown.to_numpy()]
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    ax.barh(labels, vals, color=np.where(vals > 0, "#c0392b", "#2e86c1"))
    ax.axvline(0, c="grey", lw=0.7)
    ax.set_xlabel("Contribution to stack log-odds (red = more risk)")
    ax.set_title(f"{title}\nbase {base:+.3f} -> log-odds {base + s.sum():+.3f}  (p = {prob:.3f})", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ----------------------------------------------------------------------------- runner

def run(cfg, sample=5000):
    manifest = json.loads((P3 / "model_manifest.json").read_text())
    cols = manifest["feature_order"]
    fields = sorted(set(cols) | set(FEATURE_GROUPS["engineered"]))
    val = load(cfg, names=("val",))["val"]
    X = model_frame(val[cols]).sample(sample, random_state=cfg["seed"])
    feats = X.copy()
    pre = joblib.load(manifest["models"]["xgb_hist"]["path"]).named_steps["preprocess"]
    engineered = pre.named_steps["features"].transform(X)  # raw values for engineered-ratio dependence plots
    for f in FEATURE_GROUPS["engineered"]:
        feats[f] = engineered[f].to_numpy()
    (P4 / "plots").mkdir(parents=True, exist_ok=True)
    (P4 / "tables").mkdir(parents=True, exist_ok=True)

    phis, bases, checks = {}, {}, []
    for name in TREE_MODELS:
        pipe = joblib.load(manifest["models"][name]["path"])
        phis[name], bases[name] = explain(pipe, X, fields)
        checks.append(dict(model=name, max_additivity_error=check_additivity(pipe, X, phis[name], bases[name])))

    # stack_lr: exact attribution through its own members and meta weights.
    stack = joblib.load(manifest["models"]["stack_lr"]["path"])
    w = dict(zip([n for n, _ in stack.members], stack.meta.coef_[0]))
    member_phi, member_base = {}, {}
    for mname, mpipe in stack.members:
        member_phi[mname], member_base[mname] = explain(mpipe, X, fields)
        checks.append(dict(model=f"stack_lr/{mname}",
                           max_additivity_error=check_additivity(mpipe, X, member_phi[mname], member_base[mname])))
    phis["stack_lr"] = sum(w[k] * member_phi[k].reindex(columns=fields, fill_value=0.0) for k in w)
    bases["stack_lr"] = float(stack.meta.intercept_[0] + sum(w[k] * member_base[k] for k in w))
    checks.append(dict(model="stack_lr", max_additivity_error=check_additivity(stack, X, phis["stack_lr"],
                                                                               bases["stack_lr"])))
    pd.DataFrame(checks).to_csv(P4 / "tables" / "shap_additivity.csv", index=False)
    pd.DataFrame([dict(member=k, meta_weight=float(v)) for k, v in w.items()]
                 + [dict(member="intercept", meta_weight=float(stack.meta.intercept_[0]))]
                 ).to_csv(P4 / "tables" / "stack_meta_weights.csv", index=False)

    # Global importance (field level) + group level + cross-model rank agreement.
    imp = pd.DataFrame({m: phi.reindex(columns=fields, fill_value=0.0).abs().mean() for m, phi in phis.items()})
    long = imp.rename_axis("field").reset_index().melt("field", var_name="model", value_name="mean_abs_shap")
    long["group"] = long.field.map(group_of)
    long["rank"] = long.groupby("model").mean_abs_shap.rank(ascending=False, method="min").astype(int)
    long.sort_values(["model", "rank"]).to_csv(P4 / "tables" / "shap_importance.csv", index=False)
    long.groupby(["model", "group"]).mean_abs_shap.sum().unstack().round(5).to_csv(
        P4 / "tables" / "shap_group_importance.csv")
    rho = imp.corr(method="spearman")
    rho.round(4).to_csv(P4 / "tables" / "shap_rank_agreement.csv")
    for m in imp.columns:
        bar_plot(imp[m], f"{m}: top-20 fields by mean |SHAP| (Validation n={sample:,})",
                 P4 / "plots" / f"shap_bar_{m}.png")
    beeswarm(phis["stack_lr"], feats, f"stack_lr (exact, via members) - numeric fields, n={sample:,}",
             P4 / "plots" / "shap_beeswarm_stack_lr.png")
    beeswarm(phis["xgb_hist"], feats, f"xgb_hist - numeric fields, n={sample:,}",
             P4 / "plots" / "shap_beeswarm_xgb_hist.png")

    # Monotone audit: direction of SHAP vs feature value, constrained vs unconstrained LightGBM.
    mono = []
    for field, sign in MONOTONE.items():
        x = feats[field].to_numpy(dtype=float)
        ok = np.isfinite(x)
        for m in ("lgbm_native_flags", "lgbm_monotone"):
            r = spearmanr(x[ok], phis[m][field].to_numpy()[ok]).statistic
            mono.append(dict(field=field, expected_sign=sign, model=m, spearman_value_vs_shap=float(r),
                             sign_ok=bool(np.sign(r) == sign)))
    pd.DataFrame(mono).to_csv(P4 / "tables" / "shap_monotone_check.csv", index=False)
    dependence_plot({m: phis[m] for m in ("lgbm_native_flags", "lgbm_monotone")}, feats,
                    P4 / "plots" / "shap_dependence_monotone.png")

    # Local explanations: borderline borrowers around the Champion's frozen 10:1 cutoff.
    cutoff = manifest["val_thresholds"]["stack_lr"]["10.0"]
    p_stack = stack.predict_proba(X)[:, 1]
    above = np.where(p_stack >= cutoff)[0]
    below = np.where(p_stack < cutoff)[0]
    picks = {"rejected": above[np.argmin(p_stack[above] - cutoff)],
             "approved": below[np.argmin(cutoff - p_stack[below])]}
    local = []
    for decision, i in picks.items():
        contrib = phis["stack_lr"].iloc[i]
        waterfall(contrib, bases["stack_lr"], p_stack[i],
                  f"stack_lr borderline {decision} borrower (loan_id {X.index[i]}, cutoff {cutoff:.4f})",
                  P4 / "plots" / f"shap_local_{decision}.png")
        top = contrib.reindex(contrib.abs().sort_values(ascending=False).index[:10])
        for field, value in top.items():
            local.append(dict(decision=decision, loan_id=int(X.index[i]), p_stack=float(p_stack[i]), cutoff=cutoff,
                              field=field, value=str(feats[field].iloc[i]), shap_log_odds=float(value)))
    pd.DataFrame(local).to_csv(P4 / "tables" / "shap_local.csv", index=False)
    return imp, rho, pd.DataFrame(checks), pd.DataFrame(mono)
