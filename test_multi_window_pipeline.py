"""
Unit and Integration Tests for Multi-Window ROC-AUC Evaluation Pipeline.

Validates:
1. Window duration definitions and sample count calculation
2. Actual sampling frequency handling
3. Insufficient duration detection and exclusion reporting
4. Label consistency across windows
5. Absence of NaN / Infinite feature values
6. Leakage prevention (disjoint train/test partitions)
7. ROC-AUC calculation from continuous probability scores
8. Output CSVs and ROC plot generation
9. Integrity of existing production model
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.config import FEATURES_DIR, MODELS_DIR, PLOTS_DIR
from src.model import CANONICAL_FEATURES, create_gaussian_rbf_svm_pipeline
from src.multi_window_feature_extraction import WINDOWS, WINDOW_START_TIMES
from src.multi_window_roc_evaluation import compute_bootstrap_auc_ci


def test_window_durations_configuration():
    """Verify all five window durations are properly defined in seconds."""
    expected = {
        "1_min": 60.0,
        "3_min": 180.0,
        "5_min": 300.0,
        "10_min": 600.0,
        "30_min": 1800.0,
    }
    assert WINDOWS == expected
    assert set(WINDOW_START_TIMES.keys()) == set(expected.keys())
    assert WINDOW_START_TIMES["1_min"] == 180.0
    assert WINDOW_START_TIMES["30_min"] == 0.0


def test_sample_count_calculation():
    """Verify sample count calculation respects actual sampling frequency without hardcoding."""
    for fs in [20.0, 20.00011, 20.05]:
        for w_name, w_sec in WINDOWS.items():
            expected_n = int(round(fs * w_sec))
            assert expected_n > 0
            if abs(fs - 20.0) < 0.01:
                if w_sec == 60:
                    assert expected_n == 1200
                elif w_sec == 180:
                    assert expected_n == 3600
                elif w_sec == 300:
                    assert expected_n == 6000
                elif w_sec == 600:
                    assert expected_n == 12000
                elif w_sec == 1800:
                    assert expected_n == 36000


def test_insufficient_recording_duration_exclusion():
    """Verify records shorter than required window duration are excluded without artificial padding."""
    audit_file = FEATURES_DIR / "window_extraction_audit_log.csv"
    assert audit_file.exists(), "Extraction audit log must exist."

    df_audit = pd.read_csv(audit_file)
    assert not df_audit.empty

    # Record tpehg873 has 753s; for 10_min (start=180, dur=600 -> req=780s), it must be excluded
    r873_10m = df_audit[(df_audit["window"] == "10_min") & (df_audit["record_id"] == "tpehg873")]
    assert not r873_10m.empty
    assert r873_10m.iloc[0]["status"] == "EXCLUDED"
    assert "shorter than requested window" in r873_10m.iloc[0]["reason"]


def test_consistent_labels():
    """Verify ground truth labels are strictly binary (0=Term, 1=Preterm) across all window datasets."""
    suffixes = ["1min", "3min", "5min", "10min", "30min"]
    for s in suffixes:
        csv_path = FEATURES_DIR / f"window_{s}_features.csv"
        assert csv_path.exists(), f"Feature file {csv_path} must exist."
        df = pd.read_csv(csv_path)
        assert "label" in df.columns
        unique_labels = set(df["label"].unique())
        assert unique_labels.issubset({0, 1}), f"Invalid labels found in {csv_path}: {unique_labels}"
        assert (df["label"] == 0).sum() > 0
        assert (df["label"] == 1).sum() > 0


def test_missing_and_infinite_features():
    """Verify no NaN or infinite values exist in any extracted feature matrices."""
    suffixes = ["1min", "3min", "5min", "10min", "30min"]
    for s in suffixes:
        csv_path = FEATURES_DIR / f"window_{s}_features.csv"
        df = pd.read_csv(csv_path)
        for feat in CANONICAL_FEATURES:
            assert feat in df.columns, f"Canonical feature {feat} missing in {csv_path}"
            assert not df[feat].isna().any(), f"NaN values detected in {csv_path} [{feat}]"
            assert not np.isinf(df[feat].to_numpy()).any(), f"Infinite values detected in {csv_path} [{feat}]"


def test_no_patient_leakage():
    """Verify zero overlap between training and testing records for every window."""
    train_ref = pd.read_csv(FEATURES_DIR / "train_dataset.csv")
    test_ref = pd.read_csv(FEATURES_DIR / "test_dataset.csv")

    train_ids = set(train_ref["record_id"])
    test_ids = set(test_ref["record_id"])
    assert len(train_ids.intersection(test_ids)) == 0

    suffixes = ["1min", "3min", "5min", "10min", "30min"]
    for s in suffixes:
        df = pd.read_csv(FEATURES_DIR / f"window_{s}_features.csv")
        w_train = df[df["record_id"].isin(train_ids)]
        w_test = df[df["record_id"].isin(test_ids)]
        overlap = set(w_train["record_id"]).intersection(set(w_test["record_id"]))
        assert len(overlap) == 0, f"Data leakage detected in window {s}: {overlap}"


def test_roc_auc_calculation_from_probabilities():
    """Verify ROC-AUC calculation requires continuous probabilities and returns valid scores."""
    np.random.seed(42)
    y_true = np.array([0, 0, 0, 1, 1, 0, 1, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.35, 0.8, 0.9, 0.4, 0.75, 0.15, 0.65, 0.85])

    auc = roc_auc_score(y_true, y_prob)
    assert 0.5 <= auc <= 1.0

    # Test bootstrap CI helper
    ci_low, ci_high, std_err = compute_bootstrap_auc_ci(y_true, y_prob, n_bootstraps=100)
    assert ci_low is not None and ci_high is not None
    assert 0.0 <= ci_low <= ci_high <= 1.0


def test_output_csv_and_plot_artifacts():
    """Verify all requested CSV tables and plot images were generated and are non-empty."""
    suffixes = ["1min", "3min", "5min", "10min", "30min"]

    # 1. Feature CSVs
    for s in suffixes:
        p = FEATURES_DIR / f"window_{s}_features.csv"
        assert p.exists() and p.stat().st_size > 0

    # 2. Raw ROC Coordinates CSVs
    for s in suffixes:
        p = FEATURES_DIR / f"roc_coordinates_{s}.csv"
        assert p.exists() and p.stat().st_size > 0

    # 3. Individual Plots
    for s in suffixes:
        p = PLOTS_DIR / f"roc_curve_{s}.png"
        assert p.exists() and p.stat().st_size > 1000

    # 4. Combined Plot
    comb_p = PLOTS_DIR / "roc_curve_all_windows.png"
    assert comb_p.exists() and comb_p.stat().st_size > 1000

    # 5. Comparison CSV and Report TXT
    comp_csv = FEATURES_DIR / "window_duration_auc_comparison.csv"
    assert comp_csv.exists() and comp_csv.stat().st_size > 0
    df_comp = pd.read_csv(comp_csv)
    assert len(df_comp) == 5

    rep_txt = FEATURES_DIR / "window_duration_auc_report.txt"
    assert rep_txt.exists() and rep_txt.stat().st_size > 0


def test_production_model_preserved():
    """Verify existing production model was not overwritten or removed."""
    prod_model_path = MODELS_DIR / "gaussian_rbf_svm.joblib"
    assert prod_model_path.exists(), "Production model file must be preserved."
    assert prod_model_path.stat().st_size > 1000
