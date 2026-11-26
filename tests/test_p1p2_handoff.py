"""The current handoff must keep the future P3 holdout separate and unscored."""
import json
from pathlib import Path

import pandas as pd

from src.models.block2 import BOOSTERS
from src.utils.io import config, digest


def test_p1p2_handoff():
    cfg = config()
    root = Path(cfg["paths"]["splits"])
    artifacts = Path(cfg["paths"]["artifacts"])
    manifest = json.loads((root/"manifest.json").read_text())
    for filename, checksum in manifest["sha256"].items():
        assert digest(root / filename) == checksum
    names = ("train", "val", "test")
    frames = [pd.read_parquet(root / f"{name}_ids.parquet") for name in names]
    assert [len(frame) for frame in frames] == [manifest["splits"][name]["rows"] for name in names]
    assert [len(frame) for frame in frames] == [210_000, 45_000, 45_000]
    cohort = pd.concat(frames).id
    assert len(cohort) == cfg["sample_size"] and cohort.is_unique
    exclusions = pd.read_parquet(cfg["paths"]["excluded_loan_ids"]).id
    assert exclusions.is_unique and not cohort.isin(exclusions).any()
    audit = pd.read_csv(artifacts/"feature_audit.csv")
    selected = audit.query("decision == 'keep'")
    assert len(selected) == 50 and selected.stage.eq("origination_time").all()
    assert audit.reason.notna().all() and audit.reason.str.strip().ne("").all()
    assert audit.feature_name.is_unique
    assert selected.loc[~selected.feature_name.isin(cfg["auc_allowlist"]), "single_feature_auc"].le(.85).all()
    pred = artifacts/"predictions"
    assert not list(pred.glob("*_test.parquet"))
    files = list(pred.glob("*_val.parquet"))
    assert len(files) == 21
    for path in files:
        rows = pd.read_parquet(path)
        assert rows.loan_id.equals(frames[1].id.rename("loan_id"))
        assert rows.y_prob.between(0,1).all() and rows.split.eq("val").all()
    for name in BOOSTERS:
        result = json.loads((artifacts/f"block2/search_{name}.json").read_text())
        assert result["completed_trials"] >= 3 and len(result["fold_auc"]) == 4
