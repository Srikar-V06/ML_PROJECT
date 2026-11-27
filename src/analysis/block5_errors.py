"""Block 5 — Analysis E (Error Analysis) and Analysis B (Prediction Visualization)."""
from pathlib import Path
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.analysis.block5_cost import load_calibrated_scores  # noqa: E402
from src.data.splits import load  # noqa: E402
from src.utils.io import config  # noqa: E402

P5 = Path("artifacts/p5")
KEY_MODELS = ["stack_lr", "catboost_platt", "xgb_hist_platt", "lgbm_native_flags_platt", "logreg_baseline"]
ALL_VIS_MODELS = KEY_MODELS + ["grade_only"]
THR_10 = 1.0 / 11.0  # Bayes optimal threshold for 10:1 cost ratio


def _confusion_outcomes(y: np.ndarray, p: np.ndarray, thr: float = THR_10) -> np.ndarray:
    """Return array of 'TP', 'FP', 'TN', 'FN' where positive = reject (p >= thr)."""
    rej = p >= thr
    out = np.empty(len(y), dtype=object)
    out[rej & (y == 1)] = "TP"
    out[rej & (y == 0)] = "FP"
    out[~rej & (y == 0)] = "TN"
    out[~rej & (y == 1)] = "FN"
    return out


def compute_error_profiles(test_df: pd.DataFrame, test_scores: dict) -> pd.DataFrame:
    profile_cols = ["fico_range_low", "dti", "int_rate", "annual_inc", "loan_amnt", "revol_util", "term"]
    rows = []
    for m in KEY_MODELS:
        y, p, _ = test_scores[m]
        outcomes = _confusion_outcomes(y, p, THR_10)
        df = test_df[profile_cols].copy()
        df["outcome"] = outcomes
        df["y_prob"] = p
        for oc in ["TP", "FP", "TN", "FN"]:
            sub = df[df["outcome"] == oc]
            row = {"model": m, "outcome": oc, "count": int(len(sub)), "share_pct": float(100.0 * len(sub) / len(df)),
                   "mean_p": float(sub["y_prob"].mean()) if len(sub) else np.nan}
            for c in profile_cols:
                row[f"{c}_mean"] = float(sub[c].mean()) if len(sub) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def compute_segment_errors(test_df: pd.DataFrame, test_scores: dict) -> pd.DataFrame:
    seg_df = pd.DataFrame(index=test_df.index)
    seg_df["grade"] = test_df["grade"].astype(str)
    seg_df["term"] = test_df["term"].astype(int).astype(str) + "m"
    top_purp = test_df["purpose"].value_counts().head(6).index
    seg_df["purpose"] = np.where(test_df["purpose"].isin(top_purp), test_df["purpose"].astype(str), "other")
    seg_df["income_band"] = pd.qcut(test_df["annual_inc"], 4, labels=["Q1_low", "Q2", "Q3", "Q4_high"]).astype(str)
    seg_df["issue_month"] = pd.to_datetime(test_df["issue_d"]).dt.strftime("%Y-%m")
    seg_df["emp_length_missing"] = np.where(test_df["emp_length"].isna(), "missing", "present")

    rows = []
    for m in ["stack_lr", "catboost_platt"]:
        y, p, _ = test_scores[m]
        rej = p >= THR_10
        for dim in ["grade", "term", "purpose", "income_band", "issue_month", "emp_length_missing"]:
            for val, idx in seg_df.groupby(dim, observed=False).groups.items():
                pos = seg_df.index.get_indexer(idx)
                yi, ri = y[pos], rej[pos]
                tp = int(np.sum(ri & (yi == 1)))
                fp = int(np.sum(ri & (yi == 0)))
                fn = int(np.sum(~ri & (yi == 1)))
                tn = int(np.sum(~ri & (yi == 0)))
                n = len(yi)
                rows.append({
                    "model": m,
                    "dimension": dim,
                    "segment": str(val),
                    "n": n,
                    "default_rate": float(yi.mean()),
                    "approval_rate": float((~ri).mean()),
                    "fnr": float(fn / max(fn + tp, 1)),
                    "fpr": float(fp / max(fp + tn, 1)),
                    "error_rate": float((fp + fn) / n),
                    "fn_count": fn,
                    "fp_count": fp,
                })
    return pd.DataFrame(rows)


def compute_fn_overlap(test_df: pd.DataFrame, test_scores: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    fn_masks = {}
    probs = {}
    for m in KEY_MODELS:
        y, p, _ = test_scores[m]
        fn_masks[m] = (p < THR_10) & (y == 1)
        probs[m] = p

    # Pairwise Jaccard
    rows = []
    for m1 in KEY_MODELS:
        for m2 in KEY_MODELS:
            a, b = fn_masks[m1], fn_masks[m2]
            inter = int(np.sum(a & b))
            union = int(np.sum(a | b))
            rows.append({
                "model_a": m1,
                "model_b": m2,
                "fn_a": int(a.sum()),
                "fn_b": int(b.sum()),
                "intersection": inter,
                "jaccard": float(inter / union) if union > 0 else 0.0,
            })
    overlap_df = pd.DataFrame(rows)

    # Multi-model count
    fn_count = sum(fn_masks[m].astype(int) for m in KEY_MODELS)
    y_true = test_scores["stack_lr"][0]
    counts_summary = {f"missed_by_{k}_models": int(np.sum((fn_count == k) & (y_true == 1))) for k in range(1, len(KEY_MODELS) + 1)}
    counts_summary["total_defaults"] = int(y_true.sum())

    # Hardest examples: missed by all 5 models, sorted by lowest stack_lr prob
    hard_mask = fn_count == len(KEY_MODELS)
    hard = test_df.loc[hard_mask, ["grade", "term", "int_rate", "fico_range_low", "dti", "annual_inc", "loan_amnt", "purpose"]].copy()
    for m in KEY_MODELS:
        hard[f"p_{m}"] = probs[m][hard_mask]
    hard = hard.sort_values("p_stack_lr").head(25).reset_index()
    return overlap_df, hard, counts_summary


def compute_error_confidence(test_scores: dict) -> pd.DataFrame:
    rows = []
    edges = np.linspace(0.0, 0.7, 15)
    for m in ["stack_lr", "catboost_platt"]:
        y, p, _ = test_scores[m]
        rej = p >= THR_10
        err = (rej != (y == 1)).astype(float)
        bins = np.clip(np.digitize(p, edges) - 1, 0, len(edges) - 2)
        for b in range(len(edges) - 1):
            mask = bins == b
            if not mask.any():
                continue
            rows.append({
                "model": m,
                "bin_low": float(edges[b]),
                "bin_high": float(edges[b + 1]),
                "n": int(mask.sum()),
                "mean_p": float(p[mask].mean()),
                "actual_default_rate": float(y[mask].mean()),
                "decision_error_rate": float(err[mask].mean()),
            })
    return pd.DataFrame(rows)


def compute_analysis_b(test_df: pd.DataFrame, test_scores: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # 1. Predicted vs actual by grade
    grade_rows = []
    vintage_rows = []
    cm_rows = []
    grades = test_df["grade"].astype(str)
    months = pd.to_datetime(test_df["issue_d"]).dt.strftime("%Y-%m")

    for m in ALL_VIS_MODELS:
        y, p, _ = test_scores[m]
        df_tmp = pd.DataFrame({"y": y, "p": p, "grade": grades.values, "month": months.values})
        for g, grp in df_tmp.groupby("grade"):
            grade_rows.append({"model": m, "grade": g, "n": len(grp),
                               "actual_dr": float(grp["y"].mean()), "mean_pred_p": float(grp["p"].mean())})
        for mo, grp in df_tmp.groupby("month"):
            vintage_rows.append({"model": m, "issue_month": mo, "n": len(grp),
                                 "actual_dr": float(grp["y"].mean()), "mean_pred_p": float(grp["p"].mean())})
        rej = p >= THR_10
        tp = int(np.sum(rej & (y == 1)))
        fp = int(np.sum(rej & (y == 0)))
        fn = int(np.sum(~rej & (y == 1)))
        tn = int(np.sum(~rej & (y == 0)))
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        cm_rows.append({
            "model": m, "threshold": THR_10, "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "approval_rate": float((~rej).mean()),
            "precision_reject": float(prec),
            "recall_default": float(rec),
            "f1_reject": float(2 * prec * rec / max(prec + rec, 1e-12)),
            "specificity_good": float(tn / max(tn + fp, 1)),
        })

    return pd.DataFrame(grade_rows), pd.DataFrame(vintage_rows), pd.DataFrame(cm_rows)


def plot_all(test_scores: dict, seg_df: pd.DataFrame, overlap_df: pd.DataFrame,
             counts_summary: dict, conf_df: pd.DataFrame, grade_df: pd.DataFrame,
             vintage_df: pd.DataFrame, cm_df: pd.DataFrame):
    # 1. Error segments (grade & income_band for stack_lr)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
    sub = seg_df[(seg_df["model"] == "stack_lr") & (seg_df["dimension"] == "grade")].sort_values("segment")
    x = np.arange(len(sub))
    axes[0].bar(x - 0.18, 100 * sub["fnr"], width=0.35, label="FNR (missed defaults / defaults)", color="#d62728")
    axes[0].bar(x + 0.18, 100 * sub["fpr"], width=0.35, label="FPR (declined good / good)", color="#1f77b4")
    axes[0].set_xticks(x, sub["segment"])
    axes[0].set(xlabel="LC Grade", ylabel="Rate (%)", title="stack_lr Error Rates by Grade (Test, 10:1 cutoff)")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.25)

    sub2 = seg_df[(seg_df["model"] == "stack_lr") & (seg_df["dimension"] == "purpose")].sort_values("fnr", ascending=False)
    x2 = np.arange(len(sub2))
    axes[1].bar(x2 - 0.18, 100 * sub2["fnr"], width=0.35, label="FNR", color="#d62728")
    axes[1].bar(x2 + 0.18, 100 * sub2["fpr"], width=0.35, label="FPR", color="#1f77b4")
    axes[1].set_xticks(x2, sub2["segment"], rotation=25, ha="right", fontsize=8)
    axes[1].set(xlabel="Loan Purpose", ylabel="Rate (%)", title="stack_lr Error Rates by Purpose")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(P5 / "plots" / "error_segments.png", dpi=150)
    plt.close(fig)

    # 2. FN overlap
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    piv = overlap_df.pivot(index="model_a", columns="model_b", values="jaccard").loc[KEY_MODELS, KEY_MODELS]
    im = axes[0].imshow(piv.values, cmap="Blues", vmin=0.4, vmax=1.0)
    axes[0].set_xticks(range(len(KEY_MODELS)), KEY_MODELS, rotation=30, ha="right", fontsize=8)
    axes[0].set_yticks(range(len(KEY_MODELS)), KEY_MODELS, fontsize=8)
    for i in range(len(KEY_MODELS)):
        for j in range(len(KEY_MODELS)):
            axes[0].text(j, i, f"{piv.values[i, j]:.2f}", ha="center", va="center", fontsize=8,
                         color="white" if piv.values[i, j] > 0.75 else "black")
    axes[0].set_title("Pairwise Jaccard Overlap of False Negatives")
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    ks = [k for k in range(1, len(KEY_MODELS) + 1)]
    vals = [counts_summary[f"missed_by_{k}_models"] for k in ks]
    axes[1].bar([f"{k} models" for k in ks], vals, color="#ff7f0e")
    for i, v in enumerate(vals):
        axes[1].text(i, v + 20, str(v), ha="center", fontsize=8.5)
    axes[1].set(ylabel="Count of Defaulted Borrowers", title=f"FN Consensus Across 5 Models (Total Defaults={counts_summary['total_defaults']})")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(P5 / "plots" / "error_fn_overlap.png", dpi=150)
    plt.close(fig)

    # 3. Error vs confidence
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for m, col in [("stack_lr", "#1f77b4"), ("catboost_platt", "#d62728")]:
        d = conf_df[conf_df["model"] == m]
        ax.plot(d["mean_p"], 100 * d["decision_error_rate"], marker="o", lw=1.8, color=col, label=f"{m} decision error %")
    ax.axvline(THR_10, color="grey", ls="--", label=f"10:1 threshold ({THR_10:.3f})")
    ax.set(xlabel="Mean Predicted Default Probability", ylabel="Decision Error Rate (%)",
           title="Decision Error Rate vs Predicted Probability (Peaks Near Threshold)")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(P5 / "plots" / "error_confidence.png", dpi=150)
    plt.close(fig)

    # 4. Analysis B: Probability histograms
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5))
    for ax, m in zip(axes.flat, ALL_VIS_MODELS):
        y, p, _ = test_scores[m]
        bins = np.linspace(0, 0.7, 36)
        ax.hist(p[y == 0], bins=bins, density=True, alpha=0.55, label="Non-default (y=0)", color="#1f77b4")
        ax.hist(p[y == 1], bins=bins, density=True, alpha=0.55, label="Default (y=1)", color="#d62728")
        ax.axvline(THR_10, color="black", ls="--", lw=1)
        ax.set_title(m, fontsize=9.5)
        ax.set_xlim(0, 0.7)
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Analysis B: Predicted Probability Distributions by True Class (Test)", fontsize=11)
    fig.tight_layout()
    fig.savefig(P5 / "plots" / "pred_histograms.png", dpi=150)
    plt.close(fig)

    # 5. Analysis B: Cumulative Gain & Lift
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    pct = np.linspace(0.01, 1.0, 100)
    for m in ALL_VIS_MODELS:
        y, p, _ = test_scores[m]
        order = np.argsort(-p)
        y_ord = y[order]
        cum_def = np.cumsum(y_ord) / y.sum()
        idx = np.clip((pct * len(y)).astype(int) - 1, 0, len(y) - 1)
        gain = cum_def[idx]
        lift = gain / pct
        axes[0].plot(100 * pct, 100 * gain, lw=1.7, label=m)
        axes[1].plot(100 * pct, lift, lw=1.7, label=m)
    axes[0].plot([0, 100], [0, 100], ls="--", color="grey", label="Random")
    axes[1].axhline(1.0, ls="--", color="grey", label="Random")
    axes[0].set(xlabel="% Riskiest Applicants Flagged", ylabel="% of All Defaults Captured", title="Cumulative Gain Chart (Test)")
    axes[1].set(xlabel="% Riskiest Applicants Flagged", ylabel="Lift", title="Lift Chart (Test)")
    axes[0].legend(fontsize=7.5)
    axes[1].legend(fontsize=7.5)
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(P5 / "plots" / "gain_lift.png", dpi=150)
    plt.close(fig)

    # 6. Analysis B: Predicted vs Actual by Grade & Vintage
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    act_g = grade_df[grade_df["model"] == "stack_lr"].sort_values("grade")
    axes[0].plot(act_g["grade"], 100 * act_g["actual_dr"], "ko-", lw=2.2, label="Actual Default Rate")
    for m in ["stack_lr", "catboost_platt", "xgb_hist_platt", "logreg_baseline"]:
        d = grade_df[grade_df["model"] == m].sort_values("grade")
        axes[0].plot(d["grade"], 100 * d["mean_pred_p"], marker="s", ms=4, lw=1.5, label=m)
    axes[0].set(xlabel="LC Grade", ylabel="Default Rate (%)", title="Predicted vs Actual Default Rate by Grade (Test)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)

    act_v = vintage_df[vintage_df["model"] == "stack_lr"].sort_values("issue_month")
    axes[1].plot(act_v["issue_month"], 100 * act_v["actual_dr"], "ko-", lw=2.2, label="Actual Default Rate")
    for m in ["stack_lr", "catboost_platt", "xgb_hist_platt", "logreg_baseline"]:
        d = vintage_df[vintage_df["model"] == m].sort_values("issue_month")
        axes[1].plot(d["issue_month"], 100 * d["mean_pred_p"], marker="s", ms=4, lw=1.5, label=m)
    axes[1].set(xlabel="Issue Month (Test)", ylabel="Default Rate (%)", title="Predicted vs Actual Default Rate by Vintage")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(P5 / "plots" / "pred_vs_actual_grade_vintage.png", dpi=150)
    plt.close(fig)

    # 7. Analysis B: Side-by-side Confusion Matrices
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2))
    for ax, (_, row) in zip(axes.flat, cm_df.iterrows()):
        mat = np.array([[row["TN"], row["FP"]], [row["FN"], row["TP"]]], dtype=int)
        ax.imshow(mat, cmap="Blues")
        labels = [["TN\n(App Good)", "FP\n(Decl Good)"], ["FN\n(App Def)", "TP\n(Decl Def)"]]
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{labels[i][j]}\n{mat[i, j]:,}", ha="center", va="center",
                        fontsize=8.5, color="white" if mat[i, j] > mat.max() * 0.55 else "black")
        ax.set_xticks([0, 1], ["Approve", "Decline"])
        ax.set_yticks([0, 1], ["Paid (0)", "Default (1)"])
        ax.set_title(f"{row['model']}\nApp={100*row['approval_rate']:.1f}% | Rec={100*row['recall_default']:.1f}%", fontsize=9)
    fig.suptitle("Analysis B: Side-by-Side Confusion Matrices at 10:1 Cost Threshold (Test n=45,000)", fontsize=11)
    fig.tight_layout()
    fig.savefig(P5 / "plots" / "confusion_matrices.png", dpi=150)
    plt.close(fig)


def run(cfg=None):
    cfg = cfg or config()
    (P5 / "tables").mkdir(parents=True, exist_ok=True)
    (P5 / "plots").mkdir(parents=True, exist_ok=True)

    test_df = load(cfg, names=("test",))["test"].sort_index()
    test_scores = load_calibrated_scores("test")

    prof_df = compute_error_profiles(test_df, test_scores)
    prof_df.to_csv(P5 / "tables" / "error_profiles.csv", index=False)

    seg_df = compute_segment_errors(test_df, test_scores)
    seg_df.to_csv(P5 / "tables" / "error_by_segment.csv", index=False)

    overlap_df, hard_df, counts_summary = compute_fn_overlap(test_df, test_scores)
    overlap_df.to_csv(P5 / "tables" / "error_fn_overlap.csv", index=False)
    hard_df.to_csv(P5 / "tables" / "error_hardest_examples.csv", index=False)

    conf_df = compute_error_confidence(test_scores)
    conf_df.to_csv(P5 / "tables" / "error_by_confidence.csv", index=False)

    grade_df, vintage_df, cm_df = compute_analysis_b(test_df, test_scores)
    grade_df.to_csv(P5 / "tables" / "prediction_by_grade.csv", index=False)
    vintage_df.to_csv(P5 / "tables" / "prediction_by_vintage.csv", index=False)
    cm_df.to_csv(P5 / "tables" / "confusion_matrices.csv", index=False)

    plot_all(test_scores, seg_df, overlap_df, counts_summary, conf_df, grade_df, vintage_df, cm_df)
    return counts_summary, cm_df


if __name__ == "__main__":
    summary, cm = run()
    print("FN Consensus:", summary)
    print("\nConfusion Matrices:\n", cm.to_string(index=False))
