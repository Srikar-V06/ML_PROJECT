"""Reviewed point-in-time inventory for the accepted-loans tape. New fields fail closed."""
CLEAN = set("""
loan_amnt term int_rate installment grade sub_grade emp_title emp_length home_ownership annual_inc
verification_status purpose addr_state dti delinq_2yrs earliest_cr_line fico_range_low inq_last_6mths
mths_since_last_delinq mths_since_last_record open_acc pub_rec revol_bal revol_util total_acc
initial_list_status collections_12_mths_ex_med mths_since_last_major_derog application_type
annual_inc_joint dti_joint verification_status_joint acc_now_delinq tot_coll_amt tot_cur_bal
open_acc_6m open_act_il open_il_12m open_il_24m mths_since_rcnt_il total_bal_il il_util
open_rv_12m open_rv_24m max_bal_bc all_util total_rev_hi_lim inq_fi total_cu_tl inq_last_12m
acc_open_past_24mths avg_cur_bal bc_open_to_buy bc_util chargeoff_within_12_mths delinq_amnt
mo_sin_old_il_acct mo_sin_old_rev_tl_op mo_sin_rcnt_rev_tl_op mo_sin_rcnt_tl mort_acc
mths_since_recent_bc mths_since_recent_bc_dlq mths_since_recent_inq mths_since_recent_revol_delinq
num_accts_ever_120_pd num_actv_bc_tl num_actv_rev_tl num_bc_sats num_bc_tl num_il_tl num_op_rev_tl
num_rev_accts num_rev_tl_bal_gt_0 num_sats num_tl_120dpd_2m num_tl_30dpd num_tl_90g_dpd_24m
num_tl_op_past_12m pct_tl_nvr_dlq percent_bc_gt_75 pub_rec_bankruptcies tax_liens tot_hi_cred_lim
total_bal_ex_mort total_bc_limit total_il_high_credit_limit revol_bal_joint sec_app_fico_range_low
sec_app_earliest_cr_line sec_app_inq_last_6mths sec_app_mort_acc sec_app_open_acc sec_app_revol_util
sec_app_open_act_il sec_app_num_rev_accts sec_app_chargeoff_within_12_mths sec_app_collections_12_mths_ex_med
sec_app_mths_since_last_major_derog
""".split())
ADMIN = set("id member_id url zip_code desc title issue_d disbursement_method".split())
REDUNDANT = set("fico_range_high sec_app_fico_range_high policy_code".split())
POST = set("""funded_amnt funded_amnt_inv pymnt_plan recoveries collection_recovery_fee last_credit_pull_d
deferral_term payment_plan_start_date orig_projected_additional_accrued_interest""".split())
PREFIX = ("out_prncp", "total_pymnt", "total_rec_", "last_pymnt_", "next_pymnt_", "last_fico_",
          "hardship_", "debt_settlement_", "settlement_")


def select_inputs(train):
    """Keep core underwriting fields and fill the 50-field budget by training coverage."""
    core = """loan_amnt term int_rate installment grade sub_grade emp_length home_ownership
              annual_inc verification_status purpose dti delinq_2yrs earliest_cr_line fico_range_low
              inq_last_6mths open_acc pub_rec revol_bal revol_util total_acc""".split()
    extra = sorted(CLEAN.difference(core))
    numeric = train[extra].select_dtypes(include="number")
    varying = numeric.loc[:, numeric.nunique() > 1]
    ranked = varying.notna().sum().sort_values(ascending=False, kind="stable")
    selected = core + ranked.index[:50 - len(core)].tolist()
    assert len(selected) == len(set(selected)) == 50 and set(selected) <= CLEAN
    return selected


def classify(name):
    if name in CLEAN:
        return "origination_time", "Application, original loan terms or credit bureau attributes at underwriting."
    if name == "loan_status":
        return "target", "Snapshot repayment status defines the outcome."
    if name in ADMIN:
        return "identifier_or_admin", "Identifier, split metadata, free-text listing metadata or disbursement administration."
    if name in REDUNDANT:
        return "redundant_or_constant", "Policy code or upper endpoint of a retained score band."
    if name in POST or name.startswith(PREFIX):
        return "post_origination_leakage", "Funding, servicing, subsequent credit pull or resolution event."
    raise ValueError(f"Unreviewed input column: {name}")
