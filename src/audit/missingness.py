import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from src.audit.schema import CLEAN
from src.utils.io import artifact


def category(col):
    if "joint" in col or col.startswith(("sec_app_", "mths_since_")):
        return "Structural / Not Applicable"
    if col.startswith(("emp_", "annual_inc", "dti")):
        return "Self-Reported / Behavioral (MNAR candidate)"
    if col.startswith(("num_", "mo_", "open_", "total_", "bc_", "inq_", "tot_", "il_", "revol_")) or col in {
            "mort_acc", "all_util", "pub_rec_bankruptcies", "tax_liens", "avg_cur_bal", "pct_tl_nvr_dlq",
            "max_bal_bc", "percent_bc_gt_75", "acc_open_past_24mths"}:
        return "Bureau / Credit History Coverage (MAR)"
    return "Incidental / Sparse (MCAR candidate)"


def study(train, cfg):
    rows = []
    for col in sorted(CLEAN.intersection(train.columns)):
        missing = train[col].isna()
        n, m = int(missing.sum()), int((~missing).sum())
        a = train.target[missing].mean()
        b = train.target[~missing].mean()
        p = train.target.mean()
        testable = min(n, m) > 0 and missing.mean() > .005 and 0 < p < 1
        z = (a-b)/np.sqrt(p*(1-p)*(1/n+1/m)) if testable else np.nan
        flag = bool(missing.mean() > .005 and m > 0)
        rows.append(dict(feature_name=col, missing_pct=100*missing.mean(), missing_type=category(col),
                         default_rate_missing=a, default_rate_present=b, rate_diff=a-b, p_value=2*norm.sf(abs(z)),
                         linear_model_handling="Train median or explicit missing category",
                         tree_model_handling="Preserve native missing values", add_missing_indicator=flag))
    table = pd.DataFrame(rows)
    table.to_csv(artifact(cfg, "missing_value_table.csv"), index=False)
    top = table.dropna(subset=["p_value"]).sort_values("rate_diff", key=abs).tail(12)
    if len(top):
        top.plot.barh(x="feature_name", y=["default_rate_missing", "default_rate_present"], figsize=(9, 5))
        plt.tight_layout()
        plt.savefig(artifact(cfg, "plots/missingness_default_rates.png"))
        plt.close()
    return table
