from pathlib import Path

import pandas as pd
import pytest

from src.audit.leakage import run as audit
from src.data.prep_lending_club import run as prepare
from src.data.splits import freeze, load
from src.utils.io import config


@pytest.fixture(scope="session")
def fixture_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("block1")
    cfg = config()
    # Exercises the full evaluation API on independent synthetic rows.
    cfg["paths"].pop("borrower_mapping", None)
    cfg["paths"].pop("excluded_loan_ids", None)
    for key in ("evaluation_splits", "model_features", "pca_fit_rows"):
        cfg.pop(key, None)
    raw = Path(__file__).parent / "fixtures/accepted_500.csv"
    data = pd.read_csv(raw)
    assert len(data) == 500
    # Give the synthetic fixture explicit borrower identities for the valid-cohort tests.
    data["member_id"] = [f"fixture-member-{i}" for i in range(len(data))]
    raw = root / "fixture_with_members.csv"
    data.to_csv(raw, index=False)
    cfg["paths"].update(lending_club=str(raw), sample=str(root/"sample.parquet"),
                        diagnostics=str(root/"audit_only.parquet"), splits=str(root/"splits"),
                        artifacts=str(root/"artifacts"))
    # A census of this small mature fixture makes production PSI bounds meaningful in CI.
    cfg["sample_size"] = int(data.loan_status.isin(cfg["target"]).sum())
    cfg["bootstrap_iterations"] = 12
    prepare(cfg)
    freeze(pd.read_parquet(cfg["paths"]["sample"]), cfg)
    audit(cfg)
    return cfg, load(cfg)
