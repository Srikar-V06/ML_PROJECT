"""Verification tests for Block 5 (Person 5) outputs and lock integrity."""
from pathlib import Path
import pandas as pd


P5 = Path("artifacts/p5")


def test_block5_outputs_exist():
    expected_tables = [
        "cost_data_driven.csv",
        "cost_policies_test.csv",
        "cost_sensitivity.csv",
        "error_profiles.csv",
        "error_by_segment.csv",
        "error_fn_overlap.csv",
        "error_hardest_examples.csv",
        "error_by_confidence.csv",
        "prediction_by_grade.csv",
        "prediction_by_vintage.csv",
        "confusion_matrices.csv",
        "fairness_audit.csv",
        "fairness_mitigation.csv",
        "cross_lender.csv",
    ]
    for name in expected_tables:
        p = P5 / "tables" / name
        assert p.exists(), f"Missing {p}"
        assert len(pd.read_csv(p)) > 0, f"Empty table {p}"

    expected_plots = [
        "cost_sensitivity.png",
        "cost_policies.png",
        "error_segments.png",
        "error_fn_overlap.png",
        "error_confidence.png",
        "pred_histograms.png",
        "gain_lift.png",
        "pred_vs_actual_grade_vintage.png",
        "confusion_matrices.png",
        "fairness_groups.png",
        "fairness_mitigation.png",
        "cross_lender.png",
    ]
    for name in expected_plots:
        p = P5 / "plots" / name
        assert p.exists() and p.stat().st_size > 1000, f"Missing or tiny plot {p}"

    assert Path("reports/memo_head_of_risk.md").exists()
    assert Path("artifacts/p3/test_lock.json").exists()
    assert Path("artifacts/p4/calibration_lock.json").exists()
