import pandas as pd
from pathlib import Path

from src.utils.io import artifact, cli_config, write_json
from src.utils.logging import log

# Multipliers put continuous values in the units used on the Lending Club tape.
MAPPING = [
    ("loan_amnt", "LoanOriginalAmount", 1, "Original principal in USD"),
    ("term", "Term", 1, "Contract months"),
    ("int_rate", "BorrowerRate", 100, "Nominal rate in percentage points"),
    ("annual_inc", "StatedMonthlyIncome", 12, "Annualized stated income in USD"),
    ("dti", "DebtToIncomeRatio", 100, "Percentage points; underwriting definitions can differ"),
    ("fico_range_low", "CreditScoreRangeLower", 1, "Lower bureau score band; score vendors may differ"),
    ("inq_last_6mths", "InquiriesLast6Months", 1, "Recent credit inquiries"),
    ("open_acc", "OpenCreditLines", 1, "Open credit lines"),
    ("acc_now_delinq", "CurrentDelinquencies", 1, "Active bureau delinquencies at listing"),
    ("delinq_amnt", "AmountDelinquent", 1, "Outstanding delinquent bureau balance in USD"),
    ("revol_bal", "RevolvingCreditBalance", 1, "Revolving credit balance in USD"),
    ("revol_util", "BankcardUtilization", 100, "Bankcard utilization proxy in percentage points"),
    ("emp_length", "EmploymentStatusDuration", 1/12, "Months in employment status converted to years; tenure proxy"),
    ("addr_state", "BorrowerState", None, "State code"),
    ("home_ownership", "IsBorrowerHomeowner", None, "Boolean owner flag; collapse LC OWN and MORTGAGE for transfer"),
    ("grade", "ProsperRating (Alpha)", None, "Platform-native rating with earlier CreditGrade fallback; scales differ"),
    ("earliest_cr_line", "FirstRecordedCreditLine", None, "First recorded credit line date"),
]


def run(cfg):
    raw = pd.read_csv(cfg["paths"]["prosper"], low_memory=False)
    assert len(raw) >= 100000
    target = raw.LoanStatus.map(cfg["prosper_target"])
    frame = raw.loc[target.notna()].copy()
    # Prosper's file repeats some listings across rating records. Retain one per loan.
    repeated = int(frame.LoanKey.duplicated().sum())
    frame = frame.drop_duplicates("LoanKey")
    out = pd.DataFrame({"loan_id": frame.LoanKey, "issue_d": pd.to_datetime(frame.LoanOriginationDate),
                        "target": target.loc[frame.index].astype("int8")})
    for name, source, scale, _ in MAPPING:
        value = frame[source]
        if scale is not None:
            value = value * scale
        elif name == "earliest_cr_line":
            value = pd.to_datetime(value)
        elif name == "grade":
            value = value.fillna(frame.CreditGrade)
        elif name == "home_ownership":
            value = value.map({True: "OWNER", False: "NON_OWNER"})
        out[name] = value
    Path(cfg["paths"]["prosper_common"]).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cfg["paths"]["prosper_common"], index=False)
    pd.DataFrame(MAPPING, columns=["lc_feature", "prosper_feature", "multiplier", "definition"]).to_csv(
        artifact(cfg, "tables/prosper_feature_mapping.csv"), index=False)
    summary = dict(raw=len(raw), terminal_rows=int(target.notna().sum()), deduplicated=repeated,
                   aligned=len(out), features=len(MAPPING), statuses=raw.LoanStatus.value_counts().to_dict())
    write_json(artifact(cfg, "tables/prosper_population.json"), summary)
    log(cfg, "prosper", len(out), len(MAPPING), **summary)


if __name__ == "__main__":
    run(cli_config())
