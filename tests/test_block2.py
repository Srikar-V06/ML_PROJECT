import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src.block2 import run
from src.features.block2 import FEATURE_GROUPS, UnderwritingFeatures, build_preprocessor
from src.models.baselines import model_frame
from src.models.block2 import BOOSTERS
from src.utils.io import artifact


def test_underwriting_inputs_and_ablation():
    x = pd.DataFrame(dict(loan_amnt=[120., 300.], installment=[10., 15.], annual_inc=[600., 0.]))
    out = UnderwritingFeatures().fit_transform(x)
    assert out.payment_to_income.iloc[0] == out.loan_to_income.iloc[0] == .2
    assert out.payment_to_income.isna().iloc[1]
    assert "payment_to_income" not in UnderwritingFeatures().fit_transform(x.drop(columns="annual_inc"))
    pd.testing.assert_frame_equal(UnderwritingFeatures(False).fit_transform(x), x)
    assert set(out.columns) - set(x.columns) <= FEATURE_GROUPS["engineered"]
    with pytest.raises(ValueError, match="Unreviewed"):
        UnderwritingFeatures().fit(x.assign(total_pymnt=1.))


@pytest.mark.parametrize("kind", ["tree", "linear"])
def test_fold_preprocessing(kind):
    x = pd.DataFrame(dict(annual_inc=[0., 2., np.nan], grade=["A", np.nan, "B"]))
    val = pd.DataFrame(dict(annual_inc=[np.nan, 100.], grade=["unseen", "A"]))
    prep = build_preprocessor(kind, ["annual_inc"]).fit(x)
    v = pd.DataFrame(prep.transform(val), columns=prep.get_feature_names_out())
    assert v["missing__missingindicator_annual_inc"].tolist() == [1., 0.]
    assert (v.filter(like="category__").iloc[0] == 0).all()
    if kind == "tree":
        assert np.isnan(v.numeric__annual_inc.iloc[0])
        assert v.numeric__annual_inc.iloc[1] == 100.
    else:
        assert v.numeric__annual_inc.iloc[0] == pytest.approx(0.)
        assert np.isfinite(v).all().all()


def test_block2_handoff(fixture_run, tmp_path):
    cfg, frames = fixture_run
    # Isolate from P1 tests sharing the prepared fixture.
    source = cfg
    cfg = {**source, "paths": {**source["paths"], "artifacts": str(tmp_path)}}
    for file in ("feature_audit.csv", "missing_value_table.csv"):
        artifact(cfg, file).write_bytes(artifact(source, file).read_bytes())
    run(cfg, "all", trials=1, fit_row_limit=100)
    run(cfg, "boosters", trials=1, fit_row_limit=100)
    for name in BOOSTERS:
        record = json.loads(artifact(cfg, f"block2/search_{name}.json").read_text())
        assert record["completed_trials"] == 1
        assert len(record["fold_auc"]) == cfg["cv_folds"]
    val = frames["val"]
    table = pd.read_csv(artifact(cfg, "tables/block2_pca.csv"))
    assert len(table) == 15
    assert set(table.representation) == {"control", "80", "90", "95", "99"}
    cols = pd.read_csv(artifact(cfg, "feature_audit.csv")).query("decision == 'keep'").feature_name
    probe = model_frame(val[cols])
    for name in ["decision_tree", "random_forest", *table.model]:
        model = joblib.load(artifact(cfg, f"block2/models/{name}.joblib"))
        pred = pd.read_parquet(artifact(cfg, f"predictions/{name}_val.parquet"))
        np.testing.assert_array_equal(pred.loan_id, val.index)
        np.testing.assert_array_equal(pred.y_true, val.target)
        assert pred.split.eq("val").all()
        np.testing.assert_allclose(pred.y_prob, model.predict_proba(probe)[:, 1])
    assert not list((tmp_path / "predictions").glob("*_test.parquet"))
