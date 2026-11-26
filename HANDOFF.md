# Handoff to P3

Use `config/base.yaml` and the `p1p2_300k` directories. The split is 210,000 Train, 45,000 Validation and 45,000 Test. The consolidated `excluded_loan_ids.parquet` records prior experiment IDs and is required to reproduce the draw. Seed 42 fixes sampling. Issue month and loan ID establish chronological order.

| Item | Location |
|---|---|
| Checksums, split IDs and four expanding Train folds | [Frozen manifest](splits/p1p2_300k/manifest.json) and adjacent parquets |
| Processed data | data/processed/p1p2_300k/sample.parquet; borrower linkage and excluded servicing fields in audit.parquet |
| Audited fields and missing handling | [Feature inventory](artifacts/p1p2_300k/feature_audit.csv) · [Missingness ledger](artifacts/p1p2_300k/missing_value_table.csv) |
| Baseline models and results | artifacts/p1p2_300k/models, metrics, tables and predictions |
| Scorecard | [Bin points](artifacts/p1p2_300k/tables/credit_scorecard_points.csv) · [Information value](artifacts/p1p2_300k/tables/woe_iv_summary.csv) |
| Tuned DT/RF and 15 PCA/control pipelines | artifacts/p1p2_300k/block2/models |
| Booster settings and fold AUCs | `artifacts/p1p2_300k/block2/search_MODEL.json`. The local `search.sqlite3` is needed to resume the original trials. |
| Prosper transfer preparation | data/processed/p1p2_300k/prosper_common.parquet · [Mapping](artifacts/p1p2_300k/tables/prosper_feature_mapping.csv) |

P1/P2 validation predictions for every fitted model share the same 45,000 IDs. The five booster families have tuned settings and inner-fold scores. P3 owns their final fits, early stopping, model-set freeze and single Test evaluation. Baselines use 250 bootstrap replicates for course-time validation estimates. The scorecard comparisons use paired borrower resampling and paired DeLong. Reuse the evaluator on final scores after freezing models and operating thresholds.

P2 ran three Optuna trials per family, with up to 3,000 fitting rows per expanding fold and the entire inner validation window. These are lightweight search results, not evidence of global optimality. The main decision tree and forest then use all 210,000 training borrowers. The PCA experiment uses the same 20,000 training borrowers and all 45,000 validation borrowers across its three controls and twelve variants. Reuse the saved pipelines to reproduce their predictions.

On a fresh clone, regenerate the local raw/processed files using README.md. The output directories include individual fixed-size models, validation predictions and evidence for handoff. No account key is needed for the local run. The single later Test evaluation must use frozen models and validation-selected cutoffs.
