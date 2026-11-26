import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.utils.io import cli_config, digest, write_json
from src.utils.logging import log


def verify_borrowers(ids, cfg):
    keys = pd.read_parquet(cfg["paths"]["diagnostics"], columns=["id", "member_id"])
    members = keys.set_index("id", verify_integrity=True).member_id.reindex(ids).astype("string").str.strip()
    if members.isna().any() or members.eq("").any() or not members.is_unique:
        raise ValueError("Each sampled loan must map to a distinct borrower with a populated member_id.")


def freeze(data, cfg):
    verify_borrowers(data.id, cfg)
    root = Path(cfg["paths"]["splits"])
    root.mkdir(parents=True, exist_ok=True)
    data = data.sort_values(["issue_d", "id"]).reset_index(drop=True)
    fractions = cfg["split_fractions"]
    assert len(fractions) == 3 and np.isclose(sum(fractions), 1) and min(fractions) > 0
    assert data.id.is_unique and data.issue_d.notna().all()
    cut = np.rint(np.cumsum(fractions)[:2] * len(data)).astype(int)
    manifest = {"version": 1, "order": ["issue_d", "id"], "splits": {}, "sha256": {},
                "cv_folds": cfg["cv_folds"], "sample_sha256": digest(cfg["paths"]["sample"])}
    for name, subset in zip(("train", "val", "test"), (data.iloc[:cut[0]], data.iloc[cut[0]:cut[1]], data.iloc[cut[1]:])):
        filename = f"{name}_ids.parquet"
        manifest["splits"][name] = dict(rows=len(subset), default_rate=float(subset.target.mean()),
            start=str(subset.issue_d.min().date()), end=str(subset.issue_d.max().date()), file=filename)
        save_frozen(root, filename, subset[["id"]], manifest)
    train = data.iloc[:cut[0]]
    folds = []
    for i, (fit, valid) in enumerate(TimeSeriesSplit(n_splits=cfg["cv_folds"]).split(train), 1):
        for role, indices in (("train", fit), ("val", valid)):
            folds.append(train.iloc[indices][["id"]].assign(fold=i, role=role))
    save_frozen(root, "train_expanding_cv.parquet", pd.concat(folds), manifest)
    path = root / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Existing manifest differs. Use a new splits directory to version a new sample.")
    write_json(path, manifest)
    log(cfg, "freeze", len(data), len(data.columns))
    return manifest


def save_frozen(root, name, frame, manifest):
    path = root / name
    if path.exists():
        pd.testing.assert_frame_equal(pd.read_parquet(path), frame.reset_index(drop=True))
    else:
        frame.to_parquet(path, index=False)
    manifest["sha256"][name] = digest(path)


def load(cfg, names=None):
    root = Path(cfg["paths"]["splits"])
    manifest = json.loads((root / "manifest.json").read_text())
    if digest(cfg["paths"]["sample"]) != manifest["sample_sha256"]:
        raise ValueError("Sample checksum mismatch")
    for name, expected in manifest["sha256"].items():
        if digest(root / name) != expected:
            raise ValueError(f"Split checksum mismatch: {name}")
    data = pd.read_parquet(cfg["paths"]["sample"]).set_index("id", verify_integrity=True)
    verify_borrowers(data.index, cfg)
    return {name: data.loc[pd.read_parquet(root / info["file"]).id]
            for name, info in manifest["splits"].items() if names is None or name in names}


if __name__ == "__main__":
    cfg = cli_config()
    freeze(pd.read_parquet(cfg["paths"]["sample"], columns=["id", "target", "issue_d"]), cfg)
