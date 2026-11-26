# Monotone-constraint sign check (Train only)

Each candidate feature is cut into Train deciles (`pd.qcut`, ties dropped; missing values excluded and reported separately). The sign is kept only if the Spearman correlation between decile rank and decile default rate has the expected sign with magnitude >= 0.5; otherwise the constraint is set to 0. Engineered ratios use P2's `UnderwritingFeatures` (zero/negative income -> missing). `fico_range_low` is the only FICO field in the 50-field set. All other features are unconstrained (0). Validation and Test were not used.

| Feature | Expected | Bins | First-bin default | Last-bin default | Spearman rho | Missing share | Missing default | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `dti` | +1 | 10 | 0.127 | 0.251 | +1.000 | 0.00% | — | **keep** |
| `int_rate` | +1 | 10 | 0.045 | 0.352 | +1.000 | 0.00% | — | **keep** |
| `revol_util` | +1 | 10 | 0.126 | 0.209 | +0.988 | 0.06% | 0.208 | **keep** |
| `loan_to_income` | +1 | 10 | 0.112 | 0.270 | +1.000 | 0.00% | — | **keep** |
| `payment_to_income` | +1 | 10 | 0.110 | 0.230 | +0.988 | 0.00% | — | **keep** |
| `fico_range_low` | -1 | 10 | 0.230 | 0.083 | -1.000 | 0.00% | — | **keep** |
| `annual_inc` | -1 | 10 | 0.212 | 0.130 | -1.000 | 0.00% | — | **keep** |

## Decile default rates

- `dti`: 0.127, 0.138, 0.145, 0.157, 0.167, 0.177, 0.186, 0.199, 0.221, 0.251
- `int_rate`: 0.045, 0.080, 0.110, 0.126, 0.158, 0.179, 0.213, 0.242, 0.278, 0.352
- `revol_util`: 0.126, 0.154, 0.161, 0.177, 0.181, 0.186, 0.186, 0.194, 0.196, 0.209
- `loan_to_income`: 0.112, 0.127, 0.133, 0.143, 0.159, 0.171, 0.190, 0.222, 0.244, 0.270
- `payment_to_income`: 0.110, 0.134, 0.140, 0.153, 0.167, 0.183, 0.197, 0.214, 0.241, 0.230
- `fico_range_low`: 0.230, 0.219, 0.211, 0.206, 0.189, 0.177, 0.162, 0.140, 0.118, 0.083
- `annual_inc`: 0.212, 0.209, 0.198, 0.190, 0.187, 0.177, 0.167, 0.155, 0.143, 0.130
## Fit and verification (Val only)

- Same tuned params and native-NaN/no-flag inputs as `lgbm_native`; `monotone_constraints_method='advanced'`; 7 constrained model columns.
- Train-tail early stopping: best_iteration=1297, refit_rounds=1441 on 210,000 Train rows.
- Val ROC AUC 0.7502 vs 0.7512 unconstrained (Δ -0.0010); PR AUC 0.4366 vs 0.4392 (Δ -0.0026).

### ICE sweep (2,000 Val rows, 25-point Train-quantile grid)

| Feature | Level | Sign | Grid | Violating rows | Max violation |
|---|---|---:|---:|---:|---:|
| `dti` | model_input | +1 | 25 | 0/2000 | 0.00e+00 |
| `int_rate` | model_input | +1 | 25 | 0/2000 | 0.00e+00 |
| `revol_util` | model_input | +1 | 25 | 0/2000 | 0.00e+00 |
| `loan_to_income` | model_input | +1 | 25 | 0/2000 | 0.00e+00 |
| `payment_to_income` | model_input | +1 | 25 | 0/2000 | 0.00e+00 |
| `fico_range_low` | model_input | -1 | 17 | 0/2000 | 0.00e+00 |
| `annual_inc` | model_input | -1 | 25 | 70/2000 | 5.79e-04 |
| `annual_inc` | raw_input | -1 | 25 | 1770/2000 | 5.06e-02 |

`model_input` varies one constrained column with all other model columns fixed; this is the guarantee LightGBM enforces. `raw_input` varies raw `annual_inc`, which also moves `loan_to_income`, `payment_to_income` and the unconstrained `revolving_to_income`, so raw-level monotonicity in income is not guaranteed by construction.
