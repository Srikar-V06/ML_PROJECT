# Interfaces for the 300k handoff

Read config/base.yaml from the repository root. Every saved probability represents the adverse terminal outcome. All current model evidence uses Validation. Load only the partitions required by the current experiment.

~~~python
from pathlib import Path
import joblib
import pandas as pd
from src.data.splits import load
from src.models.baselines import model_frame
from src.utils.io import config

cfg = config()
root = Path(cfg["paths"]["artifacts"])
val = load(cfg, names=["val"])["val"]
audit = pd.read_csv(root / "feature_audit.csv")
fields = audit.loc[audit.decision.eq("keep"), "feature_name"].tolist()
x = model_frame(val[fields])
forest = joblib.load(root / "block2/models/random_forest.joblib")
p = forest.predict_proba(x)[:, 1]
~~~

- **Inputs.** sample.parquet contains target, issue month, loan ID and 99 audited origination fields. The 50 keep fields are in source order. Reserved fields remain available for later studies. The separately stored member ID is checked for uniqueness by the loader. Identifiers and servicing fields stay outside prediction pipelines.
- **Splits.** load verifies the sample and split digests in the manifest. Its frames have a unique loan ID index. Split membership parquets have an id column. The CV parquet adds fold (1–4) and role (train or val). Multiple appearances across expanding folds are expected.
- **Preprocessing.** model_frame normalizes pandas dtypes. build_preprocessor in src/features/block2.py accepts tree or linear, a missing-flag list and an engineered switch. For AdaBoost and classic gradient boosting pass impute_tree=True. FEATURE_GROUPS supplies raw field groups. Construct any future ablation before computing ratios and selecting flags.
- **Model factories.** get_model(name)(x, flags, cfg) returns an unfitted estimator or pipeline. Configure selected parameters through cfg["model_params"][name] for P2 families. Search files store parameters in the params field. P1 single-model training uses python -m src.train --model NAME; it saves the fitted object and scores the splits specified in the configuration.
- **Predictions.** Each predictions/NAME_val.parquet contains unique loan_id, y_true, y_prob, split. Join by ID and verify labels before computing differences. Load the full fitted model with joblib and pass the same ordered raw model frame. P3 can evaluate the serialized versions directly.
- **Evaluation.** evaluate(y_true, y_prob, split_name, model_name, out_dir, cost_ratios, val_thresholds, n_bootstrap, seed) returns discrimination, probability, calibration and threshold-based metrics. Supply validation thresholds when scoring Test. compare_models(y, p_a, p_b, n_bootstrap, seed) returns A minus B, paired confidence intervals and p-values. Only compare aligned predictions on identical borrowers.
- **Prosper.** LoanKey identifies the selected loan and MemberKey defines one row per identified borrower. The processed table includes loan_id, issue_d, target and 17 mapped fields. For P5 select the intersection with the Lending Club schema, align units and categories, train on the common input representation, and document platform differences.
