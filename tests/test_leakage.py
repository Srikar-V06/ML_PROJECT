import pandas as pd
import pytest

from src.audit.schema import classify
from src.utils.io import artifact


def test_complete_point_in_time_inventory(fixture_run):
    cfg, splits = fixture_run
    expected = set(pd.read_csv(cfg["paths"]["lending_club"], nrows=0))
    table = pd.read_csv(artifact(cfg, "feature_audit.csv"))
    assert set(table.feature_name) == expected and table.feature_name.is_unique
    assert table.reason.str.strip().ne("").all() and table.reason.notna().all()
    kept = table.query("decision == 'keep'")
    assert set(kept.feature_name) == set(splits["train"]) - {"issue_d", "target"}
    assert len(kept) >= 50 and kept.stage.eq("origination_time").all()
    assert kept.loc[~kept.feature_name.isin(cfg["auc_allowlist"]), "single_feature_auc"].le(.85).all()
    assert all(classify(c)[0] == "origination_time" for c in kept.feature_name)
    assert not set(kept.feature_name).intersection({"id", "issue_d", "loan_status", "last_credit_pull_d",
                                                   "funded_amnt", "recoveries", "total_pymnt"})
    with pytest.raises(ValueError):
        classify("unknown_future_attribute")


def test_train_only_missingness(fixture_run):
    cfg, frames = fixture_run
    table = pd.read_csv(artifact(cfg, "missing_value_table.csv")).set_index("feature_name")
    train = frames["train"]
    assert table.loc["annual_inc_joint", "missing_type"].startswith("Structural")
    assert table.loc["annual_inc_joint", "missing_pct"] == pytest.approx(100*train.annual_inc_joint.isna().mean())
    assert table.loc["annual_inc_joint", "default_rate_missing"] == pytest.approx(train.target[train.annual_inc_joint.isna()].mean())
