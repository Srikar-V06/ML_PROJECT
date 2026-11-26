from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from src.data.prep_lending_club import linked_members, stratify
from src.data.splits import verify_borrowers
from src.utils.io import digest


@pytest.mark.parametrize("second_member", [None, "  ", " A "])
def test_reject_unverified_borrowers(tmp_path, second_member):
    path = tmp_path / "keys.parquet"
    pd.DataFrame({"id": [1, 2], "member_id": ["A", second_member]}).to_parquet(path)
    with pytest.raises(ValueError, match="distinct borrower"):
        verify_borrowers([1, 2], {"paths": {"diagnostics": path}})


def test_link_borrowers_by_verified_loan(tmp_path):
    path = tmp_path / "old.csv"
    old = pd.DataFrame(dict(id=[2, 1], member_id=["A", "B"], issue_d=["Jan-2015"]*2,
                            loan_amnt=[200, 100]))
    old.to_csv(path, index=False)
    full = pd.DataFrame(dict(id=[1, 2, 3], loan_amnt=[100, 200, 100],
                             issue_d=pd.to_datetime(["2015-01-01"]*3)))
    keys = linked_members(full, path)
    assert keys.iloc[:2].tolist() == ["B", "A"] and pd.isna(keys.iloc[2])
    full.loc[0, "loan_amnt"] = 101
    with pytest.raises(ValueError, match="principal"):
        linked_members(full, path)


def check_manifest(root):
    manifest = json.loads((root/"manifest.json").read_text())
    for name, checksum in manifest["sha256"].items():
        assert digest(root/name) == checksum
    return manifest


def test_frozen_membership(fixture_run):
    cfg, frames = fixture_run
    root = Path(cfg["paths"]["splits"])
    manifest = check_manifest(root)
    train, val, test = (frames[k] for k in ("train", "val", "test"))
    assert train.issue_d.max() <= val.issue_d.min() <= val.issue_d.max() <= test.issue_d.min()
    assert not (set(train.index) & set(val.index) or set(val.index) & set(test.index) or set(train.index) & set(test.index))
    assert sum(map(len, frames.values())) == cfg["sample_size"]
    cv = pd.read_parquet(root/"train_expanding_cv.parquet")
    previous = set()
    for _, fold in cv.groupby("fold"):
        a = fold.loc[fold.role == "train", "id"]
        b = fold.loc[fold.role == "val", "id"]
        assert previous.issubset(a) and set(a).isdisjoint(b)
        assert train.loc[a].issue_d.max() <= train.loc[b].issue_d.min()
        previous = set(a) | set(b)
    assert cv.fold.nunique() == manifest["cv_folds"] == 4


def test_deterministic_stratification(fixture_run):
    _, frames = fixture_run
    pool = pd.concat(frames.values()).reset_index()
    pool["quarter"] = pool.issue_d.dt.to_period("Q").astype(str)
    a, _ = stratify(pool, 250, 42)
    b, _ = stratify(pool, 250, 42)
    assert len(a) == a.id.nunique() == 250 and np.array_equal(a.id, b.id)
