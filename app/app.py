"""5-Page Streamlit UI for Loan Default Prediction (Person 5 — integrated with src/ & artifacts/)."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from app.scoring import (
    P3,
    P4,
    P5,
    UI_MODELS,
    default_applicant,
    explain_single,
    feature_order,
    score_dataframe,
)

st.set_page_config(page_title="Loan Default Risk & Decision Engine", layout="wide")


@st.cache_data
def load_table(path: str) -> pd.DataFrame | None:
    p = Path(path)
    return pd.read_csv(p) if p.exists() else None


def risk_band(p: float) -> str:
    if p < 0.08:
        return "Low Risk"
    if p < 0.18:
        return "Moderate Risk"
    if p < 0.30:
        return "Elevated Risk"
    return "High Risk"


def main():
    st.title("Consumer Credit Default Risk & Cost-Sensitive Decision Engine")
    st.caption("Lending Club (300k borrowers, out-of-time split) + Prosper Cross-Lender Transfer | Built on frozen P1–P5 pipelines")

    page = st.sidebar.radio(
        "Navigation",
        [
            "1. Single Applicant Scoring",
            "2. Batch CSV Scoring",
            "3. Model Comparison Dashboard",
            "4. Cost-Sensitive Policy Explorer",
            "5. Insights, Fairness & Cross-Lender",
        ],
    )
    model_key = st.sidebar.selectbox("Active Model", list(UI_MODELS.keys()), index=0)
    cost_ratio = st.sidebar.slider("Cost Ratio (C_FN : C_FP)", min_value=2.0, max_value=20.0, value=10.0, step=0.5)
    threshold = 1.0 / (1.0 + cost_ratio)
    st.sidebar.metric("Bayes Decision Threshold", f"{threshold:.4f}", help="Approve if P(default) < 1 / (1 + R)")

    # ------------------------------------------------------------------ Page 1
    if page.startswith("1."):
        st.header("Single Applicant Underwriting, SHAP Explanation & What-If")
        defaults = default_applicant()

        with st.form("applicant_form"):
            c1, c2, c3, c4 = st.columns(4)
            loan_amnt = c1.number_input("Loan Amount ($)", 1000.0, 40000.0, float(defaults["loan_amnt"]), 500.0)
            term = c2.selectbox("Term (months)", [36, 60], index=0 if int(defaults["term"]) == 36 else 1)
            int_rate = c3.number_input("Interest Rate (%)", 5.0, 32.0, float(defaults["int_rate"]), 0.25)
            annual_inc = c4.number_input("Annual Income ($)", 5000.0, 500000.0, float(defaults["annual_inc"]), 2000.0)

            dti = c1.number_input("DTI (%)", 0.0, 60.0, float(defaults["dti"]), 0.5)
            fico = c2.number_input("FICO Range Low", 600.0, 850.0, float(defaults["fico_range_low"]), 5.0)
            revol_util = c3.number_input("Revolving Utilization (%)", 0.0, 120.0, float(defaults["revol_util"]), 1.0)
            emp_length = c4.number_input("Employment Length (yrs)", 0.0, 10.0, float(defaults["emp_length"]), 1.0)

            grade = c1.selectbox("LC Grade", ["A", "B", "C", "D", "E", "F", "G"], index=2)
            sub_grade = c2.selectbox("LC Sub-Grade", [f"{g}{i}" for g in "ABCDEFG" for i in range(1, 6)], index=10)
            home_ownership = c3.selectbox("Home Ownership", ["MORTGAGE", "RENT", "OWN"], index=0)
            purpose = c4.selectbox("Purpose", ["debt_consolidation", "credit_card", "home_improvement", "major_purchase", "small_business", "other"], index=0)

            open_acc = c1.number_input("Open Accounts", 1.0, 50.0, float(defaults["open_acc"]), 1.0)
            total_acc = c2.number_input("Total Accounts", 1.0, 100.0, float(defaults["total_acc"]), 1.0)
            inq_6m = c3.number_input("Inquiries (Last 6m)", 0.0, 10.0, float(defaults["inq_last_6mths"]), 1.0)
            addr_state = c4.text_input("State (2-letter)", str(defaults["addr_state"]))

            submitted = st.form_submit_button("Score Applicant", type="primary")

        row = dict(defaults)
        r_m = (int_rate / 100.0) / 12.0
        inst = loan_amnt * (r_m * (1 + r_m) ** term) / ((1 + r_m) ** term - 1)
        row.update({
            "loan_amnt": loan_amnt, "term": term, "int_rate": int_rate, "installment": round(inst, 2),
            "annual_inc": annual_inc, "dti": dti, "fico_range_low": fico, "revol_util": revol_util,
            "emp_length": emp_length, "grade": grade, "sub_grade": sub_grade,
            "home_ownership": home_ownership, "purpose": purpose, "open_acc": open_acc,
            "total_acc": total_acc, "inq_last_6mths": inq_6m, "addr_state": addr_state.upper().strip(),
        })
        row_df = pd.DataFrame([row])

        if submitted or "last_scored" not in st.session_state:
            st.session_state["last_scored"] = row_df

        active_df = st.session_state["last_scored"]
        p_def = float(score_dataframe(active_df, model_key, apply_platt=True)[0])
        decision = "APPROVE" if p_def < threshold else "DECLINE"

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Predicted Default Prob", f"{100 * p_def:.2f}%")
        m2.metric("Decision", decision, delta=f"Cutoff {100 * threshold:.2f}%", delta_color="normal" if decision == "APPROVE" else "inverse")
        m3.metric("Risk Band", risk_band(p_def))
        m4.metric("Est. Monthly Installment", f"${float(active_df['installment'].iloc[0]):,.2f}")

        col_l, col_r = st.columns([1.2, 1.0])
        with col_l:
            st.subheader("Local SHAP Attribution (Log-Odds Drivers)")
            try:
                shap_s, base_v = explain_single(active_df, model_key)
                top = shap_s.head(12).iloc[::-1]
                fig, ax = plt.subplots(figsize=(6.5, 4.2))
                colors = ["#d62728" if v > 0 else "#1f77b4" for v in top.values]
                ax.barh(top.index, top.values, color=colors)
                ax.axvline(0, color="black", lw=0.8)
                ax.set(xlabel="SHAP Contribution (log-odds; + increases risk)", title=f"Top 12 Drivers (base logit={base_v:.3f})")
                ax.grid(axis="x", alpha=0.25)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            except Exception as exc:
                st.warning(f"SHAP explanation unavailable for this selection: {exc}")

        with col_r:
            st.subheader("What-If Sensitivity Explorer")
            wf_feat = st.selectbox("Feature to Sweep", ["int_rate", "dti", "fico_range_low", "loan_amnt", "revol_util"])
            ranges = {
                "int_rate": np.linspace(6.0, 28.0, 15),
                "dti": np.linspace(2.0, 45.0, 15),
                "fico_range_low": np.linspace(620.0, 820.0, 15),
                "loan_amnt": np.linspace(2000.0, 35000.0, 15),
                "revol_util": np.linspace(5.0, 95.0, 15),
            }
            sweep_vals = ranges[wf_feat]
            sweep_df = pd.concat([active_df] * len(sweep_vals), ignore_index=True)
            sweep_df[wf_feat] = sweep_vals
            sweep_p = score_dataframe(sweep_df, model_key, apply_platt=True)
            fig, ax = plt.subplots(figsize=(5.5, 4.2))
            ax.plot(sweep_vals, 100 * sweep_p, marker="o", ms=4, color="#1f77b4")
            ax.axhline(100 * threshold, color="#d62728", ls="--", label=f"Cutoff ({100*threshold:.1f}%)")
            ax.axvline(float(active_df[wf_feat].iloc[0]), color="grey", ls=":", label="Current")
            ax.set(xlabel=wf_feat, ylabel="Predicted Default Prob (%)", title=f"What-If: {wf_feat}")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.25)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        export = active_df.copy()
        export["model"] = model_key
        export["p_default"] = p_def
        export["threshold"] = threshold
        export["decision"] = decision
        export["risk_band"] = risk_band(p_def)
        st.download_button("Download Decision Report (CSV)", export.to_csv(index=False), "applicant_decision.csv", "text/csv")

    # ------------------------------------------------------------------ Page 2
    elif page.startswith("2."):
        st.header("Batch CSV Scoring & Schema Validation")
        st.write(f"Upload a CSV containing the **{len(feature_order())}** underwriting columns (or download the 5-row template below).")
        sample_df = pd.DataFrame([default_applicant()] * 5)
        st.download_button("Download Sample Input CSV (5 rows)", sample_df.to_csv(index=False), "sample_applicants.csv", "text/csv")

        up = st.file_uploader("Upload Applicants CSV", type=["csv"])
        if up is not None:
            try:
                raw_df = pd.read_csv(up)
                probs = score_dataframe(raw_df, model_key, apply_platt=True)
                out_df = raw_df.copy()
                out_df["p_default"] = probs
                out_df["decision"] = np.where(probs < threshold, "APPROVE", "DECLINE")
                out_df["risk_band"] = [risk_band(float(x)) for x in probs]
                st.success(f"Scored {len(out_df):,} rows | Approval rate: {100 * (probs < threshold).mean():.1f}% | Mean PD: {100 * probs.mean():.2f}%")
                st.dataframe(out_df[["loan_amnt", "int_rate", "grade", "dti", "fico_range_low", "p_default", "decision", "risk_band"]].head(50))
                st.download_button("Download Scored CSV", out_df.to_csv(index=False), "scored_applicants.csv", "text/csv")
            except Exception as exc:
                st.error(f"Validation / scoring error: {exc}")

    # ------------------------------------------------------------------ Page 3
    elif page.startswith("3."):
        st.header("Frozen Model Benchmark Dashboard (Test n=45,000)")
        bench = load_table(str(P3 / "tables" / "benchmark.csv"))
        eff = load_table(str(P3 / "tables" / "efficiency.csv"))
        cal_t = load_table(str(P4 / "tables" / "calibration_test.csv"))
        if bench is not None:
            cols_show = [c for c in ["model", "roc_auc", "pr_auc", "ks", "brier", "ece", "cost_per_1k_10", "dr_at_80_approval"] if c in bench.columns]
            st.dataframe(bench[cols_show].sort_values("pr_auc", ascending=False), use_container_width=True)
        c1, c2 = st.columns(2)
        if (P5 / "plots" / "gain_lift.png").exists():
            c1.image(str(P5 / "plots" / "gain_lift.png"), caption="Cumulative Gain & Lift (Test)")
        if (P5 / "plots" / "confusion_matrices.png").exists():
            c2.image(str(P5 / "plots" / "confusion_matrices.png"), caption="Side-by-Side Confusion Matrices (10:1 Cutoff)")
        if cal_t is not None:
            st.subheader("Post-Hoc Calibration Impact (Block 4 Test Read-Out)")
            st.dataframe(cal_t, use_container_width=True)
        if eff is not None:
            st.subheader("Efficiency & Latency Profile (Analysis C)")
            st.dataframe(eff, use_container_width=True)

    # ------------------------------------------------------------------ Page 4
    elif page.startswith("4."):
        st.header("Cost-Sensitive Lending Policy Explorer")
        cost_info_path = P5 / "tables" / "cost_data_driven.json"
        cost_info = json.loads(cost_info_path.read_text()) if cost_info_path.exists() else {"lgd_mean": 0.558, "net_margin_mean": 0.153}

        c1, c2, c3 = st.columns(3)
        lgd_in = c1.slider("Loss Given Default (LGD)", 0.20, 0.90, float(cost_info["lgd_mean"]), 0.01)
        margin_in = c2.slider("Net Interest Margin (Paid Loans)", 0.05, 0.35, float(cost_info["net_margin_mean"]), 0.01)
        implied_r = lgd_in / margin_in
        c3.metric("Implied Data-Driven Cost Ratio", f"{implied_r:.2f} : 1", f"Bayes Cutoff = {1/(1+implied_r):.4f}")

        pol = load_table(str(P5 / "tables" / "cost_policies_test.csv"))
        if pol is not None:
            st.subheader("Test Policy Comparison (n=45,000)")
            st.dataframe(pol, use_container_width=True)
        col1, col2 = st.columns(2)
        if (P5 / "plots" / "cost_policies.png").exists():
            col1.image(str(P5 / "plots" / "cost_policies.png"), caption="Policy Net Profit & Approval vs Default Rate")
        if (P5 / "plots" / "cost_sensitivity.png").exists():
            col2.image(str(P5 / "plots" / "cost_sensitivity.png"), caption="Sensitivity Sweep (2:1 to 20:1)")

    # ------------------------------------------------------------------ Page 5
    else:
        st.header("Insights, Error Analysis, Fairness & Cross-Lender Generalization")
        t1, t2, t3, t4 = st.tabs(["Ablation & SHAP (P4)", "Error Analysis (Analysis E)", "Fairness & Mitigation", "Cross-Lender (Prosper)"])
        with t1:
            c1, c2 = st.columns(2)
            if (P4 / "plots" / "ablation.png").exists():
                c1.image(str(P4 / "plots" / "ablation.png"), caption="Analysis A: Feature-Group Ablation")
            if (P4 / "plots" / "shap_beeswarm_stack_lr.png").exists():
                c2.image(str(P4 / "plots" / "shap_beeswarm_stack_lr.png"), caption="Analysis F: Champion SHAP Summary")
        with t2:
            c1, c2 = st.columns(2)
            if (P5 / "plots" / "error_segments.png").exists():
                c1.image(str(P5 / "plots" / "error_segments.png"), caption="Error Rates by Grade & Purpose")
            if (P5 / "plots" / "error_fn_overlap.png").exists():
                c2.image(str(P5 / "plots" / "error_fn_overlap.png"), caption="Cross-Model False Negative Overlap")
            hard = load_table(str(P5 / "tables" / "error_hardest_examples.csv"))
            if hard is not None:
                st.subheader("Hardest False Negatives (Missed by All 5 Models)")
                st.dataframe(hard.head(15), use_container_width=True)
        with t3:
            c1, c2 = st.columns(2)
            if (P5 / "plots" / "fairness_groups.png").exists():
                c1.image(str(P5 / "plots" / "fairness_groups.png"), caption="Fairness Metrics by Group")
            if (P5 / "plots" / "fairness_mitigation.png").exists():
                c2.image(str(P5 / "plots" / "fairness_mitigation.png"), caption="Fairlearn ThresholdOptimizer Trade-Off")
            mit = load_table(str(P5 / "tables" / "fairness_mitigation.csv"))
            if mit is not None:
                st.dataframe(mit, use_container_width=True)
        with t4:
            cl = load_table(str(P5 / "tables" / "cross_lender.csv"))
            if (P5 / "plots" / "cross_lender.png").exists():
                st.image(str(P5 / "plots" / "cross_lender.png"), caption="Lending Club -> Prosper Transfer")
            if cl is not None:
                st.dataframe(cl, use_container_width=True)


if __name__ == "__main__":
    main()
