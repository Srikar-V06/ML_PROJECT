"""Unit tests for Block 3 (P3) scaffolding on the 500-row synthetic fixture."""
import json
import numpy as np
import pandas as pd
import pytest

from src.block3 import (
    CatBoostNativePreprocessor,
    DEFAULT_BOOSTER_PARAMS,
    P3_BOOSTERS,
    chronological_train_tail_split,
    extract_keep_flags,
    fit_with_train_tail_early_stopping,
    save_benchmark_table,
    save_p3_artifacts,
    stratified_train_subsample,
    tune_booster_p3,
    verify_manifest_and_acquire_test_lock,
    write_model_manifest,
)
from src.models.baselines import model_frame
from src.models.registry import get_model, list_models
from src.utils.io import artifact


def _fixture_train_val(fixture_run):
    cfg, frames = fixture_run
    audit = pd.read_csv(artifact(cfg, "feature_audit.csv"))
    cols = audit.loc[audit.decision.eq("keep"), "feature_name"].tolist()
    missing = pd.read_csv(artifact(cfg, "missing_value_table.csv"))
    flags = extract_keep_flags(missing, cols)
    tr, va = frames["train"], frames["val"]
    return cfg, cols, flags, tr, va


def test_p3_models_registered():
    registered = set(list_models())
    assert set(P3_BOOSTERS) <= registered


def test_catboost_native_preprocessor_and_subsample(fixture_run, tmp_path):
    cfg, cols, flags, tr, va = _fixture_train_val(fixture_run)
    X_tr = model_frame(tr[cols])
    prep = CatBoostNativePreprocessor(flags=flags, engineered=True).fit(X_tr)
    Xt = prep.transform(model_frame(va[cols]))
    assert isinstance(Xt, pd.DataFrame)
    for col in prep.cat_cols_:
        assert Xt[col].dtype == object or pd.api.types.is_string_dtype(Xt[col])
        assert not Xt[col].isna().any()

    sub_path = tmp_path / "gb_sub.parquet"
    ids1, n1 = stratified_train_subsample(X_tr, tr.target, tr.issue_d, n_rows=100, seed=42, out_path=sub_path)
    ids2, n2 = stratified_train_subsample(X_tr, tr.target, tr.issue_d, n_rows=100, seed=42)
    assert n1 == n2 == 100
    assert ids1.equals(ids2)
    saved = pd.read_parquet(sub_path)
    assert saved["loan_id"].tolist() == ids1.tolist()


@pytest.mark.parametrize("name", P3_BOOSTERS)
def test_train_tail_early_stopping_all_boosters(fixture_run, name):
    cfg, cols, flags, tr, va = _fixture_train_val(fixture_run)
    X_tr = model_frame(tr[cols])
    y_tr = tr.target
    X_va = model_frame(va[cols])

    fit_idx, tail_idx = chronological_train_tail_split(X_tr, y_tr, issue_d=tr.issue_d, tail_fraction=0.10)
    assert len(fit_idx) + len(tail_idx) == len(X_tr)
    assert tr.loc[fit_idx, "issue_d"].max() <= tr.loc[tail_idx, "issue_d"].min()

    factory = get_model(name)
    model, meta = fit_with_train_tail_early_stopping(
        factory,
        X_tr,
        y_tr,
        flags,
        cfg,
        issue_d=tr.issue_d,
        tail_fraction=0.10,
        patience=5,
        max_rounds=25,
        adaboost_max_rounds=15,
    )
    assert meta["probe_fit_rows"] + meta["probe_tail_rows"] == meta["fit_rows"] == len(X_tr)
    assert meta["refit_rounds"] == max(1, round(meta["best_iteration"] / 0.9))

    probs = model.predict_proba(X_va)[:, 1]
    assert probs.shape == (len(X_va),)
    assert np.all(np.isfinite(probs))
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_optuna_tuner_and_recovery_fallback(fixture_run, tmp_path):
    cfg, cols, flags, tr, _ = _fixture_train_val(fixture_run)
    X_tr = model_frame(tr[cols])
    y_tr = tr.target

    params, rec = tune_booster_p3(
        "lightgbm",
        X_tr,
        y_tr,
        cfg,
        trials=1,
        timeout_seconds=60,
        fit_row_limit=100,
        max_rounds=20,
        patience=5,
        p3_root=tmp_path,
    )
    assert rec["status"] == "complete"
    assert rec["completed_trials"] == 1
    assert "learning_rate" in params

    # Fallback when 0 trials complete
    fb_params, fb_rec = tune_booster_p3(
        "hgb",
        X_tr,
        y_tr,
        cfg,
        trials=0,
        timeout_seconds=60,
        fit_row_limit=100,
        p3_root=tmp_path,
    )
    assert fb_rec["status"] == "fallback_default"
    assert fb_params == DEFAULT_BOOSTER_PARAMS["hgb"]
    recovery = pd.read_csv(tmp_path / "tables" / "recovery.csv")
    assert set(recovery["family"]) == {"lightgbm", "hgb"}


def test_monotone_vector_and_ensembles(fixture_run):
    from sklearn.linear_model import LogisticRegression
    from src.block3 import DEFAULT_MONOTONE_SIGNS, monotone_vector
    from src.block3_ensembles import AveragingEnsemble, StackedEnsemble, to_logit

    names = ["numeric__dti", "numeric__fico_range_low", "numeric__loan_amnt", "category__grade_A", "missing__x_dti"]
    assert monotone_vector(names, DEFAULT_MONOTONE_SIGNS) == [1, -1, 0, 0, 0]

    cfg, cols, flags, tr, va = _fixture_train_val(fixture_run)
    X_tr, X_va = model_frame(tr[cols]), model_frame(va[cols])
    members = [(n, get_model(n)(X_tr, flags, cfg).fit(X_tr, tr.target)) for n in ("lgbm_native", "logreg_baseline")]
    avg = AveragingEnsemble(members)
    expected = np.mean([m.predict_proba(X_va)[:, 1] for _, m in members], axis=0)
    assert np.allclose(avg.predict_proba(X_va)[:, 1], expected)

    Z = np.column_stack([to_logit(m.predict_proba(X_tr)[:, 1]) for _, m in members])
    stack = StackedEnsemble(members, LogisticRegression().fit(Z, tr.target))
    p = stack.predict_proba(X_va)
    assert p.shape == (len(X_va), 2) and np.allclose(p.sum(axis=1), 1.0)


def test_dual_export_and_test_lock_guard(fixture_run, tmp_path):
    cfg, cols, flags, tr, va = _fixture_train_val(fixture_run)
    X_tr = model_frame(tr[cols])
    X_va = model_frame(va[cols])
    model = get_model("lgbm_native")(X_tr, flags, cfg).fit(X_tr, tr.target)
    p = model.predict_proba(X_va)[:, 1]

    pred_df = pd.DataFrame({
        "loan_id": va.index.to_numpy(dtype="int64"),
        "y_true": va.target.to_numpy(dtype="int8"),
        "y_prob": p.astype("float64"),
        "split": "val",
    })
    p3_root = tmp_path / "artifacts" / "p3"
    brief_root = tmp_path / "brief"
    paths = save_p3_artifacts(
        "lgbm_native",
        model,
        pred_df,
        "val",
        run_meta={"roc_auc": 0.70},
        p3_root=p3_root,
        brief_root=brief_root,
        timestamp="20261126T150000Z",
    )
    assert paths["p3_model"].exists() and paths["brief_model"].exists()
    canon = pd.read_parquet(paths["p3_pred"])
    brief = pd.read_parquet(paths["brief_pred"])
    assert list(canon.columns) == ["loan_id", "y_true", "y_prob", "split"]
    assert list(brief.columns) == ["id", "y_true", "p_default"]
    assert brief["id"].equals(canon["loan_id"])
    assert np.allclose(brief["p_default"], canon["y_prob"])

    bm_df = pd.DataFrame([{"model": "lgbm_native", "roc_auc": 0.70}])
    p_a, p_b = save_benchmark_table(bm_df, p3_root=p3_root, brief_root=brief_root)
    assert p_a.exists() and p_b.exists()

    # Manifest & Test-lock guard
    manifest_path = p3_root / "model_manifest.json"
    lock_path = p3_root / "test_lock.json"
    write_model_manifest({"lgbm_native": paths["p3_model"]}, cols, manifest_path, champion="lgbm_native")

    # Feature order mismatch must fail before locking
    with pytest.raises(ValueError, match="Feature column order mismatch"):
        verify_manifest_and_acquire_test_lock(manifest_path, lock_path, expected_feature_order=cols[::-1])
    assert not lock_path.exists()

    # Valid run acquires lock
    loaded = verify_manifest_and_acquire_test_lock(manifest_path, lock_path, expected_feature_order=cols)
    assert loaded["champion"] == "lgbm_native"
    assert lock_path.exists()

    # Second run must refuse
    with pytest.raises(RuntimeError, match="Test lock already exists"):
        verify_manifest_and_acquire_test_lock(manifest_path, lock_path, expected_feature_order=cols)

    # Tampered model SHA must fail if lock is absent
    lock_path.unlink()
    paths["p3_model"].write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_manifest_and_acquire_test_lock(manifest_path, lock_path, expected_feature_order=cols)
