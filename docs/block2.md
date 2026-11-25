# Block 2 | Features and classical models

## Run from the repository root

Use Python 3.12 and the project requirements. The input is P1's checksum-matched `data/processed/lending_club_200k.parquet`. If the shared copy lives elsewhere, add `--sample "path/to/lending_club_200k.parquet"` to a command. The split IDs, feature audit and missingness ledger come from P1.

```powershell
python -m pip install -r requirements.txt
python -m pytest -q tests/test_block2.py
python -m src.block2 --stage boosters --trials 12
python -m src.block2 --stage classical --trials 12
python -m src.block2 --stage eda
```

The booster command is the overnight job. Retrying it fills each study up to the requested total of completed trials. A study is scoped to the model, frozen manifest, seed and fitting-row limit. The search uses a reproducible sample of up to 20,000 loans from each inner fitting partition to control runtime. Every candidate sees the same rows and every inner validation partition is scored in full. Use `--fit-row-limit 0` for full-fold fitting. The final classical and PCA models use every Train loan. The P2 runner scores Val. P3 can use the saved pipelines for the final benchmark.

## From an application to a prediction

1. **Select inputs.** The loader verifies the parquet and membership hashes. The runner selects P1's keep-list and normalizes columns using the same `model_frame` function as the existing baselines.
2. **Build features.** Loan-to-income measures loan principal relative to one year's reported income. Payment-to-income measures the fraction of annual income consumed by twelve scheduled payments. Revolving-balance-to-income measures a second debt burden. Open accounts divided by total accounts measures the share still open. Missing operands or a nonpositive denominator yield a missing ratio.
3. **Fit the preprocessor.** Numeric gaps remain available to NaN-capable trees. Linear models, PCA, AdaBoost and classic gradient boosting use fitting-set medians. Linear/PCA numeric inputs are standardized. Categorical inputs have a missing token, with a bounded one-hot vocabulary for dense matrices. Unseen categories yield zeros. P1's ledger selects extra missing flags for the final fit. Each inner fold recomputes the same rule, which requires more than 0.5% missing and at least one observed value.
4. **Tune on earlier loans.** Optuna maximizes mean ROC AUC across P1's four expanding Train folds. Each candidate gets a new preprocessor in each fold. The booster families are AdaBoost, classic gradient boosting, XGBoost, LightGBM and CatBoost.
5. **Explain the classical models.** The decision tree tunes maximum depth, minimum leaf size and pruning strength. Its validation pruning curve describes sensitivity after selection. A separate depth-three fit gives readable rules with training counts and default fractions. The forest tunes tree count, depth, leaf size and feature subsampling. The OOB curve uses only trees that omitted each loan from their bootstrap draw. Impurity importance measures reductions at fitted splits. Validation permutation importance measures the loss of ROC AUC when one transformed column is shuffled. Correlated predictors can substitute for each other.
6. **Test the PCA hypothesis.** Fit PCA on imputed training inputs. Keep the smallest number of components reaching each of 80%, 90%, 95% and 99% cumulative variance. Evaluate logistic regression, random forest and LightGBM at each cutoff alongside three controls using the same preprocessing and classifier settings. A split on one original field may require several splits on rotated axes. Use the measured AUC changes to decide whether this mattered.

## Outputs

| Output under `artifacts/` | Use |
| --- | --- |
| `block2/search.sqlite3`, `block2/search_<model>.json` | Resumable trials, selected estimator parameters, inner-fold scores and fitting budget |
| `block2/models/<name>.joblib` | Complete fitted pipeline with feature engineering and classifier |
| `predictions/<name>_val.parquet` | Validation loan ID, observed label and default probability for each model |
| `tables/block2_classical.csv`, `tables/block2_pca.csv` | Validation ROC AUC and average precision, including PCA changes against matched controls |
| `tables/block2_tree_rules.csv`, `block2_pruning.csv`, `block2_oob.csv`, `block2_rf_importance.csv` in the same directory | Leaf conditions and counts, pruning curve, OOB coverage and both importance measures. `block2/tree_findings.md` interprets the highest-default leaf. |
| `tables/block2_pc_weights.csv`, `block2/pca_findings.md` | First three signed component axes, leading contributors, explained variance and measured changes |
| `plots/block2_*.png`, `block2/eda_findings.md` | Model diagnostics and Train EDA with data-derived takeaways for segments, distributions, deciles, correlation, quarterly drift and 2D PCA/t-SNE |

For an ablation, drop the raw columns in the chosen `FEATURE_GROUPS` set and remove them from the flags passed to the preprocessor. Ratios are built only when both source columns remain. Pass `engineered=False` to omit all ratios or `flags=[]` to omit all flags.

To score with a fitted pipeline, pass the scoring cohort through P1's model frame using the audit keep-list.

```python
import joblib
from src.models.baselines import model_frame

model = joblib.load("artifacts/block2/models/random_forest.joblib")
probability = model.predict_proba(model_frame(frame[columns]))[:, 1]
```

P3 can also pass the selected JSON `params` dictionary as `cfg["model_params"][name]` to a registry factory for a fresh fit. The existing `src.train` command scores both Val and Test. Use the P2 commands above during model development.
