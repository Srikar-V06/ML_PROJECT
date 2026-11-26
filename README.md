# Loan default prediction • P1 and P2

The project uses **300,000 distinct borrowers and 50 selected origination fields**. P1 prepares data, baselines and evaluation. P2 hands over engineered features, tuned classical models, PCA experiments, and booster search results.

| Deliverable | Open |
|---|---|
| Data and cohort | [Frozen splits](splits/p1p2_300k/manifest.json) · [Feature audit](artifacts/p1p2_300k/feature_audit.csv) |
| Block 1 results | [Baseline metrics](artifacts/p1p2_300k/tables/model_metrics.csv) · [Paired comparisons](artifacts/p1p2_300k/tables/model_comparisons.csv) |
| Block 2 results | [Tree and forest metrics](artifacts/p1p2_300k/tables/block2_classical.csv) · [PCA comparison](artifacts/p1p2_300k/tables/block2_pca.csv) · [Booster searches](artifacts/p1p2_300k/block2/) |
| P3 starting point | [Handoff](HANDOFF.md) · [Interfaces](docs/interfaces.md) · [Configuration](config/base.yaml) |

## Reproduce

Use Python 3.12 from the repository root. Install the pinned requirements. Put `accepted_2007_to_2018Q4.csv.gz` and `lc_loan.csv.zip` in `data/raw`. Block 1 downloads the dictionary and Prosper CSV if needed. Rebuilding uses the frozen split files and the exclusion IDs under `splits/p1p2_300k/`. The modeling data is generated at `data/processed/p1p2_300k/sample.parquet`.

~~~powershell
python -m pip install -r requirements.txt
python -m src.block1
python -m src.block2 --stage classical
python -m src.block2 --stage boosters
python -m src.block2 --stage eda
python -m pytest -q
~~~

For an existing checkout with saved predictions and models, use the handoff directly. To intentionally change the cohort, use versioned output directories and a new manifest.
