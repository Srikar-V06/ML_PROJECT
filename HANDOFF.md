# Block 1 handoff to P2

Start with `splits/manifest.json`, `artifacts/feature_audit.csv` and `artifacts/missing_value_table.csv`. The production sample contains 200,000 accepted loans. Use the 140,000 Train IDs for estimation and the four expanding folds inside Train for engineering decisions. Val and Test each contain 30,000 IDs. Branch membership is recoverable from the committed ID parquets.

## Reproduce the run

1. Install Python 3.12 and create a virtual environment. Install `python -m pip install -r requirements.txt`.
2. Place the supplied gzip at `data/raw/accepted_2007_to_2018Q4.csv.gz`. For an offline production run, also place `data/raw/LCDataDictionary.xlsx` and `data/raw/prosperLoanData.csv` at those exact paths. The configured companion URLs are recorded in config/base.yaml. See the source digest in artifacts/tables/population.json.
3. Run `python -m src.block1 --config config/base.yaml`. The pipeline prepares the clean and audit-only parquets, checks the frozen membership, fits the four models and writes predictions, plots, metrics and Prosper alignment. Use a fresh output directory when deliberately versioning a different sample. The pinned reference run uses the built-in monotonic WoE fallback. Installing optional optbinning can change the scorecard.
4. Run `python -m pytest -v`. CI builds temporary outputs from the committed synthetic 500-row fixture and verifies shipped audit and split evidence. The offline fixture has no real borrower records.

## Add a P2 model

Implement a factory with `@register_model("my_model")` accepting the three arguments documented in docs/interfaces.md. Import the module in list_models. Then run `python -m src.train --model my_model --config config/base.yaml`. The CLI fits on Train, locks cutoffs on Val and scores Test. Use the four expanding Train folds for hyperparameter selection before this command. Compare paired predictions by loan_id. The Test results for Block 1 are already disclosed. Repeated model selection against them consumes the holdout.

## Data contract

- Use the 99 audited features as the input inventory. Preserve sparse fields for P2. The ledger includes Train missing percentages, event rates, mechanism hypotheses and indicator recommendations. Refit transformations inside each CV training fold.
- Numeric linear inputs use Train median imputation and scaling. Categorical inputs learn a missing level and a Train-only vocabulary. Trees can use native missing values. A completely empty Train feature has no estimable effect. Inspect later coverage when interpreting drift.
- Loan id verifies loan-level uniqueness. All member_id values are missing. Distinct people and borrower-level recurrence across splits are unobservable.
- The cohort includes accepted, resolved loans through 2016 Q1. Selection into approval, residual 60-month censoring and changes in underwriting limit generalization. Issue dates have monthly precision and adjacent partitions may share a month. Targets are final snapshot outcomes. A production as-of-label backtest would require historical outcome-availability dates.
- Grade, subgrade, rate and installment assume scoring after original pricing terms are known. The feature map distinguishes historical bureau collections at application from recoveries on the subject loan. Keep data/processed/audit_only.parquet out of every modeling workflow except the leakage demonstration.
- The administrative group comprises loan ID, member ID, listing URL, masked ZIP, listing title and description, issue month and disbursement method. The two free-text listing fields were treated as listing metadata. The archived dictionary contains 80 worksheet rows on its LoanStats tab and predates some columns in this tape. The complete 151-field timing inventory is in src/audit/schema.py.
- Evaluation uses relative FN and FP costs and one observation per loan. Choose production cost weights from exposure, recoveries, foregone margin and the decision context before interpreting dollar savings. Prosper has distinct product definitions and rating scales.

The full results are in [the Block 1 report](docs/block1_report.md). The operating thresholds are in [the risk memo](docs/head_of_risk_memo.md).

The companion files used for the reference run are identified below.

| Input | SHA-256 |
| --- | --- |
| [Lending Club dictionary archived at DePaul](https://bigblue.depaul.edu/jlee141/econdata/LendingClub_LoanData/LC_DataDictionary.xlsx) | `3cf0ac7ac903238ea6e7ee1eb42b28c5409f66ff6c4fe9b3004c7d649a145f0a` |
| [Historical Prosper export hosted by Udacity](https://s3.amazonaws.com/udacity-hosted-downloads/ud651/prosperLoanData.csv) | `71c1c1931f65bc4d2e3a8f3603d865e074e327b2271590f3a06829197b9cce0f` |

## Block 2 handoff to P3

The P2 runner tunes models inside the frozen Train folds and writes validation predictions in P1's loan-ID format. Its production input is the clean parquet whose SHA-256 appears in the frozen manifest. The repository fixture exercises the workflow on synthetic loans. For the commands, preprocessing choices, artifact paths and instructions for loading a fitted pipeline, see the [Block 2 guide](docs/block2.md).
