import json

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from src.models.baselines import model_frame, sanity
from src.models.registry import get_model, list_models
from src.models.scorecard import OFFSET, FACTOR
from src.train import run
from src.utils.io import artifact


def test_overfit_1000(fixture_run):
    cfg, frames = fixture_run
    data = pd.concat([frames["train"]]*4, ignore_index=True).iloc[:1000]
    assert len(data) == 1000
    x = model_frame(data.drop(columns=["target", "issue_d"]))
    assert sanity(x, data.target, 42)["train_auc"] > .99


@pytest.mark.parametrize("name", ["constant", "grade_only", "logreg_baseline", "woe_scorecard"])
def test_train_entrypoint(name, fixture_run):
    cfg, frames = fixture_run
    assert name in list_models()
    run(name, cfg)
    report = json.loads(artifact(cfg, f"metrics/{name}.json").read_text())
    assert report["test"]["thresholds"] == report["val"]["thresholds"]
    pred = pd.read_parquet(artifact(cfg, f"predictions/{name}_test.parquet"))
    assert list(pred) == ["loan_id", "y_true", "y_prob", "split"]
    assert np.array_equal(pred.loan_id, frames["test"].index)
    assert np.array_equal(pred.y_true, frames["test"].target)
    if name == "constant":
        assert np.allclose(pred.y_prob, frames["train"].target.mean())


def test_score_points_match_probability(fixture_run):
    cfg, frames = fixture_run
    x = model_frame(frames["train"].drop(columns=["target", "issue_d"]))
    model = get_model("woe_scorecard")(x, [], cfg).fit(x, frames["train"].target)
    x = x.iloc[:12]
    table = pd.read_csv(artifact(cfg, "tables/credit_scorecard_points.csv"))
    points = np.full(len(x), table.loc[table.feature == "(base)", "points"].iloc[0])
    for col in model.features:
        mapping = table.loc[table.feature == col].set_index("bin").points
        points += model.binners[col].bins(x[col]).map(mapping).fillna(0.).to_numpy()
    assert np.allclose(points, model.points(x))
    assert np.allclose(expit((OFFSET-points)/FACTOR), model.predict_proba(x)[:, 1])
