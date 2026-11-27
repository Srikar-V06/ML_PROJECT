"""Block 5 — Fairness audit across proxy/demographic-adjacent groups & Fairlearn mitigation."""
from pathlib import Path
import matplotlib
import numpy as np
import pandas as pd
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.base import BaseEstimator, ClassifierMixin

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.analysis.block5_cost import load_calibrated_scores  # noqa: E402
from src.analysis.block5_errors import THR_10  # noqa: E402
from src.data.splits import load  # noqa: E402
from src.evaluation.calibration import ece  # noqa: E402
from src.utils.io import config  # noqa: E402

P5 = Path("artifacts/p5")

# US Census Regions for addr_state
STATE_TO_REGION = {
    "CT": "Northeast", "ME": "Northeast", "MA": "Northeast", "NH": "Northeast", "RI": "Northeast", "VT": "Northeast",
    "NJ": "Northeast", "NY": "Northeast", "PA": "Northeast",
    "IL": "Midwest", "IN": "Midwest", "MI": "Midwest", "OH": "Midwest", "WI": "Midwest",
    "IA": "Midwest", "KS": "Midwest", "MN": "Midwest", "MO": "Midwest", "NE": "Midwest", "ND": "Midwest", "SD": "Midwest",
    "DE": "South", "FL": "South", "GA": "South", "MD": "South", "NC": "South", "SC": "South", "VA": "South",
    "DC": "South", "WV": "South", "AL": "South", "KY": "South", "MS": "South", "TN": "South", "AR": "South",
    "LA": "South", "OK": "South", "TX": "South",
    "AZ": "West", "CO": "West", "ID": "West", "MT": "West", "NV": "West", "NM": "West", "UT": "West", "WY": "West",
    "AK": "West", "CA": "West", "HI": "West", "OR": "West", "WA": "West",
}


class PrecomputedScoreEstimator(BaseEstimator, ClassifierMixin):
    """Adapter for fairlearn.ThresholdOptimizer using precomputed probabilities passed as 1-col X."""
    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = np.asarray(X, dtype=float).ravel()
        return np.column_stack([1.0 - p, p])


def build_group_frame(df: pd.DataFrame, inc_bins=None) -> tuple[pd.DataFrame, np.ndarray]:
    if inc_bins is None:
        _, inc_bins = pd.qcut(df["annual_inc"], 4, retbins=True)
        inc_bins[0], inc_bins[-1] = -np.inf, np.inf
    g = pd.DataFrame(index=df.index)
    g["region"] = df["addr_state"].astype(str).map(STATE_TO_REGION).fillna("Other")
    ho = df["home_ownership"].astype(str)
    g["home_ownership"] = np.where(ho.isin(["MORTGAGE", "RENT", "OWN"]), ho, "OTHER")
    g["income_band"] = pd.cut(df["annual_inc"], bins=inc_bins, labels=["Q1_low", "Q2", "Q3", "Q4_high"]).astype(str)
    g["emp_length_missing"] = np.where(df["emp_length"].isna(), "missing", "present")
    return g, inc_bins


def group_metrics(y: np.ndarray, p: np.ndarray, approved: np.ndarray, groups: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rej = ~approved
    for dim in groups.columns:
        for grp, idx in groups.groupby(dim, observed=False).groups.items():
            pos = groups.index.get_indexer(idx)
            yi, pi, ri = y[pos], p[pos], rej[pos]
            tp = int(np.sum(ri & (yi == 1)))
            fp = int(np.sum(ri & (yi == 0)))
            fn = int(np.sum(~ri & (yi == 1)))
            tn = int(np.sum(~ri & (yi == 0)))
            tpr = tp / max(tp + fn, 1)
            fpr = fp / max(fp + tn, 1)
            rows.append({
                "dimension": dim,
                "group": str(grp),
                "n": int(len(yi)),
                "actual_default_rate": float(yi.mean()),
                "mean_pred_p": float(pi.mean()),
                "ece": float(ece(yi, pi)) if len(yi) >= 20 else np.nan,
                "approval_rate": float((~ri).mean()),
                "tpr_reject_default": float(tpr),
                "fpr_reject_good": float(fpr),
                "fnr_approve_default": float(fn / max(tp + fn, 1)),
            })
    return pd.DataFrame(rows)


def audit_all(val_df: pd.DataFrame, test_df: pd.DataFrame, test_scores: dict) -> pd.DataFrame:
    _, inc_bins = build_group_frame(val_df)
    g_test, _ = build_group_frame(test_df, inc_bins=inc_bins)
    frames = []
    for m in ["stack_lr", "catboost_platt"]:
        y, p, _ = test_scores[m]
        app = p < THR_10
        df_m = group_metrics(y, p, app, g_test)
        df_m.insert(0, "model", m)
        frames.append(df_m)
    return pd.concat(frames, ignore_index=True)


def mitigate_fairlearn(val_df: pd.DataFrame, test_df: pd.DataFrame, val_scores: dict, test_scores: dict) -> pd.DataFrame:
    """Fit Fairlearn ThresholdOptimizer on Validation for stack_lr and evaluate on Test."""
    g_val, inc_bins = build_group_frame(val_df)
    g_test, _ = build_group_frame(test_df, inc_bins=inc_bins)

    yv, pv, _ = val_scores["stack_lr"]
    yt, pt, _ = test_scores["stack_lr"]

    # Binary label: 1 = reject/default risk, 0 = approve. We evaluate across sensitive dimensions.
    rows = []
    for dim in ["region", "home_ownership", "income_band"]:
        sv = g_val[dim].to_numpy()
        st = g_test[dim].to_numpy()

        # Baseline (unmitigated 10:1 Bayes rule)
        rej_base = pt >= THR_10
        rows.append(_mitigation_summary("unmitigated_10_to_1", dim, yt, rej_base, st))

        for constraint in ["equalized_odds", "demographic_parity"]:
            est = PrecomputedScoreEstimator().fit(pv.reshape(-1, 1), yv)
            opt = ThresholdOptimizer(
                estimator=est,
                constraints=constraint,
                objective="balanced_accuracy_score",
                prefit=True,
                predict_method="predict_proba",
            )
            opt.fit(pv.reshape(-1, 1), yv, sensitive_features=sv)
            rej_mit = opt.predict(pt.reshape(-1, 1), sensitive_features=st, random_state=42).astype(bool)
            rows.append(_mitigation_summary(f"fairlearn_{constraint}", dim, yt, rej_mit, st))
    return pd.DataFrame(rows)


def _mitigation_summary(arm: str, dim: str, y: np.ndarray, rej: np.ndarray, sens: np.ndarray) -> dict:
    app = ~rej
    fp = int(np.sum(rej & (y == 0)))
    fn = int(np.sum(app & (y == 1)))
    tp = int(np.sum(rej & (y == 1)))
    tn = int(np.sum(app & (y == 0)))
    cost_1k = float((fp + 10.0 * fn) * 1000.0 / len(y))
    bal_acc = 0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1))

    app_rates, tprs, fprs = [], [], []
    for grp in np.unique(sens):
        m = sens == grp
        yi, ri = y[m], rej[m]
        app_rates.append(float((~ri).mean()))
        tprs.append(float(np.sum(ri & (yi == 1)) / max(np.sum(yi == 1), 1)))
        fprs.append(float(np.sum(ri & (yi == 0)) / max(np.sum(yi == 0), 1)))

    return {
        "dimension": dim,
        "arm": arm,
        "approval_rate": float(app.mean()),
        "default_rate_approved": float(y[app].mean()) if app.any() else 0.0,
        "cost_per_1k_R10": cost_1k,
        "balanced_accuracy": float(bal_acc),
        "dp_gap_approval": float(max(app_rates) - min(app_rates)),
        "tpr_gap": float(max(tprs) - min(tprs)),
        "fpr_gap": float(max(fprs) - min(fprs)),
        "eo_gap_max": float(max(max(tprs) - min(tprs), max(fprs) - min(fprs))),
    }


def plot_fairness(audit_df: pd.DataFrame, mit_df: pd.DataFrame):
    sub = audit_df[audit_df["model"] == "stack_lr"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, dim in zip(axes.flat, ["region", "home_ownership", "income_band", "emp_length_missing"]):
        d = sub[sub["dimension"] == dim]
        x = np.arange(len(d))
        ax.bar(x - 0.2, 100 * d["approval_rate"], width=0.25, label="Approval %", color="#1f77b4")
        ax.bar(x + 0.05, 100 * d["tpr_reject_default"], width=0.25, label="TPR (Decline Def) %", color="#2ca02c")
        ax.bar(x + 0.3, 100 * d["fpr_reject_good"], width=0.25, label="FPR (Decline Good) %", color="#d62728")
        ax.set_xticks(x, d["group"], fontsize=8.5)
        ax.set(ylabel="Rate (%)", title=f"stack_lr Fairness Metrics by {dim}")
        ax.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Fairness Audit Across Available Demographic-Adjacent Groups (Test, 10:1 Cutoff)", fontsize=11)
    fig.tight_layout()
    fig.savefig(P5 / "plots" / "fairness_groups.png", dpi=150)
    plt.close(fig)

    # Mitigation plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    dims = ["region", "home_ownership", "income_band"]
    x = np.arange(len(dims))
    w = 0.25
    for idx, (arm, lbl, col) in enumerate([
        ("unmitigated_10_to_1", "Unmitigated (10:1)", "#7f7f7f"),
        ("fairlearn_equalized_odds", "Fairlearn Equalized Odds", "#1f77b4"),
        ("fairlearn_demographic_parity", "Fairlearn Demographic Parity", "#ff7f0e"),
    ]):
        d = mit_df[mit_df["arm"] == arm].set_index("dimension").loc[dims]
        axes[0].bar(x + (idx - 1) * w, 100 * d["eo_gap_max"], width=w, label=lbl, color=col)
        axes[1].bar(x + (idx - 1) * w, d["cost_per_1k_R10"], width=w, label=lbl, color=col)
    axes[0].set_xticks(x, dims)
    axes[0].set(ylabel="Max Equalized Odds Gap (%)", title="Equalized Odds Gap: Before vs After Mitigation")
    axes[1].set_xticks(x, dims)
    axes[1].set(ylabel="10:1 Cost per 1,000 Applicants", title="Cost Trade-Off of Fairness Mitigation")
    axes[0].legend(fontsize=8)
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(P5 / "plots" / "fairness_mitigation.png", dpi=150)
    plt.close(fig)


def run(cfg=None):
    cfg = cfg or config()
    (P5 / "tables").mkdir(parents=True, exist_ok=True)
    (P5 / "plots").mkdir(parents=True, exist_ok=True)

    frames = load(cfg, names=("val", "test"))
    val_df = frames["val"].sort_index()
    test_df = frames["test"].sort_index()

    val_scores = load_calibrated_scores("val")
    test_scores = load_calibrated_scores("test")

    audit_df = audit_all(val_df, test_df, test_scores)
    audit_df.to_csv(P5 / "tables" / "fairness_audit.csv", index=False)

    mit_df = mitigate_fairlearn(val_df, test_df, val_scores, test_scores)
    mit_df.to_csv(P5 / "tables" / "fairness_mitigation.csv", index=False)

    plot_fairness(audit_df, mit_df)
    return audit_df, mit_df


if __name__ == "__main__":
    aud, mit = run()
    print("Fairness Mitigation:\n", mit.to_string(index=False))
