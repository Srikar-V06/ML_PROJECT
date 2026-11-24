# Block 1 interfaces

Run commands from the repository root with Python 3.12. The shipped configuration uses seed 42. All probabilities represent the probability of the adverse terminal outcome.

## Evaluation

`evaluate(y_true, y_prob, split_name, model_name, out_dir, cost_ratios=(5.0, 10.0, 15.0), val_thresholds=None, n_bootstrap=1000, seed=42)` in `src.evaluation.metrics` consumes aligned one-dimensional arrays with both binary classes and finite probabilities in [0, 1]. When `out_dir` is supplied, it writes `<model_name>_<split_name>.json`. The CLI passes None and writes a combined `{"val": ..., "test": ...}` report.

Each split record contains model and split strings, integer rows, float roc_auc, pr_auc (average precision), ks (maximum absolute class-CDF separation), brier, log_loss and ece (ten equal-width bins). Calibration uses an unpenalized intercept and slope on clipped log-odds. Both are JSON null for constant scores because they cannot be jointly identified. ci95 maps roc_auc, pr_auc, brier and ks to two percentile endpoints. The bootstrap resamples loans with replacement and redraws single-class samples.

thresholds maps string keys "5.0", "10.0" and "15.0" to numeric cutoffs. Pass the validation mapping unchanged to Test. A probability equal to the threshold triggers rejection. cost_per_1000 maps each ratio to threshold, optimized, naive_050, approve_all and approval_rate. Costs use one unit for a false positive. The optimizer scans whole tie groups and prefers the higher threshold at equal cost.

`compare_models(y_true, y_prob_a, y_prob_b, n_bootstrap=1000, seed=42)` in `src.evaluation.stats_tests` reports A minus B. delong contains difference, z, p_value and ci95. A nonzero difference with zero estimated variance has null z. paired_bootstrap contains roc_auc, pr_auc and brier, each with difference, p_value and ci95. Both models use the same resampled indices. The bootstrap test uses a null-centered two-sided tail with a finite-replication correction. Negative Brier differences favor A.

## Model registry and predictions

`register_model(name)` returns a decorator. The registered callable takes `(X, flags, cfg)` and returns an estimator supporting `fit(X, y)` and `predict_proba(X)[:, 1]`. X has the audited columns, with datetime columns expressed as days since 1970-01-01, numeric columns as float and missing strings represented by NaN. flags contains the Train-derived indicator column names. `get_model(name)` returns the factory and raises KeyError for unknown names. `list_models()` returns the sorted registered names. Import a new model module inside list_models so each fresh CLI process discovers it.

`artifacts/predictions/<model>_<split>.parquet` has loan_id (int64, unique per file), y_true (int8), y_prob (float64) and split (string, val or test). Use loan_id to join predictions and assert identical labels before a paired comparison. The test cohort contains 30,000 records.

## Frozen files

`splits/manifest.json` has version (integer), order (list of sort keys), cv_folds (integer), sample_sha256 (hex digest of the clean parquet), sha256 (filename to hex digest) and splits (train, val, test). Each split record has file, rows, start, end and default_rate. Dates are inclusive ISO dates at monthly source precision. The three membership parquets have one int64 id column. train_expanding_cv.parquet has int64 id, integer fold (1 through 4) and string role (train or val). IDs recur across expanding folds by design.

`src.data.splits.load(cfg)` verifies all five file digests and returns a dictionary of data frames indexed by loan id. Each has target, issue_d and 99 origination columns. Only rows with decision=keep in feature_audit.csv enter model factories. Model features exclude id, issue_d and target. Audit-only servicing data live under data/processed/audit_only.parquet.

## Input provenance

Lending Club uses the supplied accepted-loans gzip. The dictionary and historical Prosper export resolve from the URLs in config/base.yaml when the files are absent. Prosper's aligned parquet has loan_id (LoanKey), issue_d, target and the 17 underwriting fields described in artifacts/tables/prosper_feature_mapping.csv. Homeowner and grade values require platform-specific harmonization before external scoring.
