PC1 explains 15.5% of training variance. Its largest signed axis weights are numeric__num_op_rev_tl (+0.30), numeric__num_sats (+0.29), numeric__open_acc (+0.28), numeric__num_bc_sats (+0.27). The bureau group contributes the largest share (91.7% of squared axis weights). Positive weights move the score up as that encoded input increases, and negative weights move it down. The overall axis sign is arbitrary.

PC2 explains 7.6% of training variance. Its largest signed axis weights are numeric__revol_util (+0.37), numeric__bc_util (+0.36), numeric__percent_bc_gt_75 (+0.33), numeric__loan_to_income (+0.25). The bureau group contributes the largest share (59.2% of squared axis weights). Positive weights move the score up as that encoded input increases, and negative weights move it down. The overall axis sign is arbitrary.

PC3 explains 7.0% of training variance. Its largest signed axis weights are numeric__fico_range_low (+0.26), numeric__loan_amnt (+0.26), numeric__total_bc_limit (+0.25), numeric__installment (+0.24). The bureau group contributes the largest share (70.7% of squared axis weights). Positive weights move the score up as that encoded input increases, and negative weights move it down. The overall axis sign is arbitrary.

lightgbm validation AUC changes relative to its matched control were 80% -0.0094, 90% -0.0110, 95% -0.0098, 99% -0.0113.

lr validation AUC changes relative to its matched control were 80% -0.0014, 90% -0.0003, 95% +0.0006, 99% -0.0006.

random_forest validation AUC changes relative to its matched control were 80% -0.0121, 90% -0.0109, 95% -0.0096, 99% -0.0086.

PCA retains high-variance directions without using the default label. A useful income or delinquency cutoff can span several rotated axes, requiring more tree splits. The matched validation changes show whether this affected this cohort.
