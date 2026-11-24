from pathlib import Path
import json

import numpy as np
import pandas as pd

from src.data.prep_lending_club import stratify
from src.utils.io import digest


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


def test_production_split_manifest():
    root = Path("splits")
    manifest = check_manifest(root)
    ids = []
    for name, count in [("train", 140000), ("val", 30000), ("test", 30000)]:
        part = manifest["splits"][name]
        subset = pd.read_parquet(root/part["file"]).id
        assert len(subset) == part["rows"] == count
        ids.extend(subset)
    assert len(set(ids)) == 200000


def test_deterministic_stratification(fixture_run):
    _, frames = fixture_run
    pool = pd.concat(frames.values()).reset_index()
    pool["quarter"] = pool.issue_d.dt.to_period("Q").astype(str)
    a, _ = stratify(pool, 250, 42)
    b, _ = stratify(pool, 250, 42)
    assert len(a) == a.id.nunique() == 250 and np.array_equal(a.id, b.id)
