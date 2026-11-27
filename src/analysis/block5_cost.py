"""Block 5 — Cost-sensitive threshold study, per-loan dollar rule, policies, and sensitivity (Steps 10.2-10.5)."""
from pathlib import Path
import json
import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.data.splits import load  # noqa: E402
from src.evaluation.cost_sensitive import curve, select  # noqa: E402
from src.utils.io import config, write_json  # noqa: E402

P3 = Path("artifacts/p3")
P4 = Path("artifacts/p4")
P5 = Path("artifacts/p5")

FUNDING_COST_RATE = 0.04  # ~2.5% annualized * ~1.6y avg duration on Lending Club


def load_calibrated_scores(split: str) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return {model_label: (y_true, y_prob, loan_ids)} for key P5 models on a split ('val' or 'test')."""
    manifest = json.loads((P3 / "model_manifest.json").read_text())
    specs = [
        ("stack_lr", "stack_lr", None),
        ("catboost_platt", "catboost", P4 / "calibration" / "catboost_platt.joblib"),
        ("xgb_hist_platt", "xgb_hist", P4 / "calibration" / "xgb_hist_platt.joblib"),
        ("lgbm_native_flags_platt", "lgbm_native_flags", P4 / "calibration" / "lgbm_native_flags_platt.joblib"),
        ("logreg_baseline", "logreg_baseline", None),
        ("grade_only", "grade_only", None),
    ]
    out = {}
    for label, base_name, cal_path in specs:
        if split == "val":
            pred_path = Path(manifest["models"][base_name]["val_pred_path"])
        else:
            pred_path = P3 / "predictions" / f"{base_name}_test.parquet"
        df = pd.read_parquet(pred_path).set_index("loan_id").sort_index()
        p = df["y_prob"].to_numpy(dtype=float)
        if cal_path is not None:
            p = joblib.load(cal_path).transform(p)
        out[label] = (df["y_true"].to_numpy(dtype=int), p, df.index.to_numpy())
    return out


def compute_data_driven_costs(cfg) -> dict:
    """Estimate LGD, interest margin, and cost ratio strictly on Train rows using audit.parquet."""
    train_ids = pd.read_parquet(Path(cfg["paths"]["splits"]) / "train_ids.parquet")["id"]
    sample = pd.read_parquet(cfg["paths"]["sample"], columns=["id", "target", "loan_amnt"]).set_index("id")
    audit = pd.read_parquet(
        cfg["paths"]["diagnostics"],
        columns=["id", "funded_amnt", "total_rec_prncp", "recoveries", "total_rec_int"],
    ).set_index("id")
    df = sample.loc[train_ids].join(audit, how="inner")
    funded = df["funded_amnt"].replace(0, np.nan)

    defaulters = df[df["target"] == 1]
    paid = df[df["target"] == 0]

    lgd_series = ((defaulters["funded_amnt"] - defaulters["total_rec_prncp"] - defaulters["recoveries"]) / funded.loc[defaulters.index]).clip(0.0, 1.0)
    gross_margin_series = (paid["total_rec_int"] / funded.loc[paid.index]).clip(0.0, 2.0)

    lgd = float(lgd_series.mean())
    gross_margin = float(gross_margin_series.mean())
    net_margin = max(gross_margin - FUNDING_COST_RATE, 0.01)

    r_gross = lgd / gross_margin
    r_net = lgd / net_margin
    avg_loan = float(df["loan_amnt"].mean())

    return {
        "train_rows": int(len(df)),
        "train_default_rate": float(df["target"].mean()),
        "avg_loan_amnt": avg_loan,
        "lgd_mean": lgd,
        "lgd_median": float(lgd_series.median()),
        "gross_margin_mean": gross_margin,
        "funding_cost_assumed": FUNDING_COST_RATE,
        "net_margin_mean": net_margin,
        "cost_ratio_gross": float(r_gross),
        "cost_ratio_net": float(r_net),
        "bayes_thr_gross": float(1.0 / (1.0 + r_gross)),
        "bayes_thr_net": float(1.0 / (1.0 + r_net)),
        "bayes_thr_10": float(1.0 / 11.0),
    }


def _policy_metrics(name: str, approved: np.ndarray, y: np.ndarray, loan_amnt: np.ndarray,
                    lgd: float, margin: float, r_ref: float = 10.0) -> dict:
    approved = np.asarray(approved, dtype=bool)
    rejected = ~approved
    tp = int(np.sum(rejected & (y == 1)))
    fp = int(np.sum(rejected & (y == 0)))
    fn = int(np.sum(approved & (y == 1)))
    tn = int(np.sum(approved & (y == 0)))
    n = len(y)
    app_rate = float(approved.mean())
    dr_app = float(y[approved].mean()) if approved.any() else 0.0

    unit_cost_10 = float((fp + 10.0 * fn) * 1000.0 / n)
    unit_cost_ref = float((fp + r_ref * fn) * 1000.0 / n)

    # Realized / expected dollar profit under LGD & margin
    pnl_per_loan = np.where(
        approved,
        np.where(y == 0, margin * loan_amnt, -lgd * loan_amnt),
        0.0,
    )
    total_profit = float(pnl_per_loan.sum())
    profit_per_app = float(pnl_per_loan.mean())
    capital_deployed = float(loan_amnt[approved].sum())
    roi_pct = float(100.0 * total_profit / capital_deployed) if capital_deployed > 0 else 0.0

    return {
        "policy": name,
        "approval_rate": app_rate,
        "approved_n": int(approved.sum()),
        "default_rate_approved": dr_app,
        "TP_declined_defaulter": tp,
        "FP_declined_good": fp,
        "FN_approved_defaulter": fn,
        "TN_approved_good": tn,
        "unit_cost_per_1k_R10": unit_cost_10,
        "unit_cost_per_1k_Rdata": unit_cost_ref,
        "net_profit_total_usd": total_profit,
        "net_profit_per_applicant_usd": profit_per_app,
        "capital_deployed_usd": capital_deployed,
        "roi_pct": roi_pct,
    }


def evaluate_policies(cfg, cost_info: dict, val_scores: dict, test_scores: dict) -> pd.DataFrame:
    """Compare lending policies on Test using thresholds locked on Validation."""
    frames = load(cfg, names=("val", "test"))
    val_df = frames["val"].sort_index()
    test_df = frames["test"].sort_index()

    lgd = cost_info["lgd_mean"]
    margin = cost_info["net_margin_mean"]
    r_data = cost_info["cost_ratio_net"]

    rows = []
    # 1 & 2: Non-model policies
    y_test = test_df["target"].to_numpy(dtype=int)
    amnt_test = test_df["loan_amnt"].to_numpy(dtype=float)
    grades_test = test_df["grade"].astype(str).to_numpy()

    rows.append(_policy_metrics("approve_all", np.ones(len(y_test), dtype=bool), y_test, amnt_test, lgd, margin, r_data))
    rows.append(_policy_metrics("grades_A_to_C_only", np.isin(grades_test, ["A", "B", "C"]), y_test, amnt_test, lgd, margin, r_data))

    for model_name in ["stack_lr", "catboost_platt"]:
        yv, pv, id_v = val_scores[model_name]
        yt, pt, id_t = test_scores[model_name]
        assert np.array_equal(id_t, test_df.index.to_numpy())
        assert np.array_equal(id_v, val_df.index.to_numpy())

        t_val_10 = select(yv, pv, [10.0])["10.0"]
        t_val_data = select(yv, pv, [r_data])[str(float(r_data))]
        t_bayes_10 = 1.0 / 11.0
        t_bayes_data = 1.0 / (1.0 + r_data)

        policy_defs = [
            (f"{model_name}|naive_050", pt < 0.50),
            (f"{model_name}|bayes_10_to_1", pt < t_bayes_10),
            (f"{model_name}|val_tuned_10_to_1", pt < t_val_10),
            (f"{model_name}|bayes_data_driven", pt < t_bayes_data),
            (f"{model_name}|val_tuned_data_driven", pt < t_val_data),
            (f"{model_name}|dollar_expected_profit", (1.0 - pt) * margin * amnt_test > pt * lgd * amnt_test),
        ]
        for q in (0.70, 0.80, 0.90):
            t_q = float(np.quantile(pv, q))
            policy_defs.append((f"{model_name}|fixed_approval_{int(q*100)}", pt < t_q))

        for pname, app_mask in policy_defs:
            rows.append(_policy_metrics(pname, app_mask, yt, amnt_test, lgd, margin, r_data))

    return pd.DataFrame(rows)


def sensitivity_sweep(val_scores: dict, test_scores: dict, ratios=range(2, 21)) -> pd.DataFrame:
    """Sweep cost ratio R from 2:1 to 20:1 on Validation and evaluate on Test."""
    rows = []
    for model_name in ["stack_lr", "catboost_platt", "xgb_hist_platt"]:
        yv, pv, _ = val_scores[model_name]
        yt, pt, _ = test_scores[model_name]
        for r in ratios:
            r_f = float(r)
            t_bayes = 1.0 / (1.0 + r_f)
            t_val = select(yv, pv, [r_f])[str(r_f)]

            for rule, thr in [("bayes", t_bayes), ("val_tuned", t_val)]:
                rej = pt >= thr
                fp = int(np.sum(rej & (yt == 0)))
                fn = int(np.sum(~rej & (yt == 1)))
                cost_1k = float((fp + r_f * fn) * 1000.0 / len(yt))
                app_rate = float((~rej).mean())
                dr_app = float(yt[~rej].mean()) if (~rej).any() else 0.0
                rows.append({
                    "model": model_name,
                    "cost_ratio": int(r),
                    "rule": rule,
                    "threshold": float(thr),
                    "test_cost_per_1k": cost_1k,
                    "test_approval_rate": app_rate,
                    "test_default_rate_approved": dr_app,
                })
    return pd.DataFrame(rows)


def plot_sensitivity(sens: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))
    sub = sens[sens["model"].isin(["stack_lr", "catboost_platt"])]
    styles = {
        ("stack_lr", "bayes"): ("#1f77b4", "-", "stack_lr (Bayes 1/(1+R))"),
        ("stack_lr", "val_tuned"): ("#1f77b4", "--", "stack_lr (Val-tuned)"),
        ("catboost_platt", "bayes"): ("#d62728", "-", "catboost_platt (Bayes)"),
        ("catboost_platt", "val_tuned"): ("#d62728", "--", "catboost_platt (Val-tuned)"),
    }
    for (m, r), (c, ls, lbl) in styles.items():
        d = sub[(sub["model"] == m) & (sub["rule"] == r)]
        axes[0].plot(d["cost_ratio"], d["threshold"], color=c, ls=ls, lw=1.8, label=lbl)
        axes[1].plot(d["cost_ratio"], d["test_cost_per_1k"], color=c, ls=ls, lw=1.8, label=lbl)
        axes[2].plot(d["cost_ratio"], 100 * d["test_approval_rate"], color=c, ls=ls, lw=1.8, label=lbl)

    axes[0].set(xlabel="Cost Ratio (C_FN : C_FP)", ylabel="Decision Threshold", title="Optimal Threshold vs Cost Ratio")
    axes[1].set(xlabel="Cost Ratio (C_FN : C_FP)", ylabel="Test Expected Cost / 1k", title="Minimum Cost vs Cost Ratio")
    axes[2].set(xlabel="Cost Ratio (C_FN : C_FP)", ylabel="Test Approval Rate (%)", title="Approval Rate vs Cost Ratio")
    for ax in axes:
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("Cost-Sensitive Threshold Sensitivity (2:1 to 20:1, Test n=45,000)", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_policies(policies_df: pd.DataFrame, out_path: Path):
    keep = [
        "approve_all",
        "grades_A_to_C_only",
        "stack_lr|naive_050",
        "stack_lr|fixed_approval_80",
        "stack_lr|bayes_data_driven",
        "stack_lr|dollar_expected_profit",
        "stack_lr|bayes_10_to_1",
        "catboost_platt|bayes_10_to_1",
    ]
    d = policies_df[policies_df["policy"].isin(keep)].copy()
    d["short"] = d["policy"].str.replace("stack_lr|", "stack: ").str.replace("catboost_platt|", "cb_platt: ")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    y_pos = np.arange(len(d))
    colors = ["#7f7f7f" if "approve_all" in p or "grades" in p else "#1f77b4" if "stack" in p else "#d62728" for p in d["policy"]]
    axes[0].barh(y_pos, d["net_profit_per_applicant_usd"], color=colors)
    axes[0].axvline(0, color="black", lw=0.8)
    axes[0].set_yticks(y_pos, d["short"], fontsize=8.5)
    axes[0].set(xlabel="Realized Net Profit per Applicant ($)", title="Test Net Profit per Applicant (Data-Driven LGD & Margin)")
    axes[0].grid(axis="x", alpha=0.25)

    axes[1].scatter(100 * d["approval_rate"], 100 * d["default_rate_approved"], s=90, c=colors)
    for _, row in d.iterrows():
        axes[1].annotate(row["short"], (100 * row["approval_rate"], 100 * row["default_rate_approved"]),
                         fontsize=7.5, xytext=(4, 3), textcoords="offset points")
    axes[1].set(xlabel="Approval Rate (%)", ylabel="Default Rate Among Approved (%)", title="Approval vs Portfolio Default Rate Trade-Off")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run(cfg=None):
    cfg = cfg or config()
    (P5 / "tables").mkdir(parents=True, exist_ok=True)
    (P5 / "plots").mkdir(parents=True, exist_ok=True)

    cost_info = compute_data_driven_costs(cfg)
    write_json(P5 / "tables" / "cost_data_driven.json", cost_info)
    pd.DataFrame([cost_info]).to_csv(P5 / "tables" / "cost_data_driven.csv", index=False)

    val_scores = load_calibrated_scores("val")
    test_scores = load_calibrated_scores("test")

    policies_df = evaluate_policies(cfg, cost_info, val_scores, test_scores)
    policies_df.to_csv(P5 / "tables" / "cost_policies_test.csv", index=False)

    sens_df = sensitivity_sweep(val_scores, test_scores)
    sens_df.to_csv(P5 / "tables" / "cost_sensitivity.csv", index=False)

    plot_sensitivity(sens_df, P5 / "plots" / "cost_sensitivity.png")
    plot_policies(policies_df, P5 / "plots" / "cost_policies.png")
    return cost_info, policies_df, sens_df


if __name__ == "__main__":
    info, pol, sens = run()
    print("Data-driven costs:", json.dumps(info, indent=2))
    print("\nPolicies:\n", pol[["policy", "approval_rate", "default_rate_approved", "unit_cost_per_1k_R10", "net_profit_per_applicant_usd"]].to_string(index=False))
