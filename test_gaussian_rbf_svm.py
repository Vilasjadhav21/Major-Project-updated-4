"""
Unit and Integration Tests for Gaussian RBF-SVM Pipeline and Merged EHG Dataset.

Verifies:
    1. Loading the merged dataset (326 records).
    2. Correct label mapping (0=Term, 1=Preterm).
    3. Missing and infinite feature values checks.
    4. Canonical feature column validation and metadata stripping.
    5. Train/test separation with zero patient/record leakage.
    6. No oversampling or distortion of independent test data.
    7. Model training, prediction, and probability estimation.
    8. Saving and loading the trained Gaussian RBF-SVM model.
    9. Reproducibility under fixed random_state.
    10. Valid clinical evaluation metrics suite.
    11. Inference on arbitrary new single-record feature inputs.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

from src.config import FEATURES_DIR, MODELS_DIR
from src.model import (
    CANONICAL_FEATURES,
    EHGClassifier,
    create_gaussian_rbf_svm_pipeline,
)


@pytest.fixture
def merged_data():
    csv_path = FEATURES_DIR / "combined_features.csv"
    assert csv_path.exists(), f"Merged dataset missing: {csv_path}"
    return pd.read_csv(csv_path)


@pytest.fixture
def train_test_data():
    train_path = FEATURES_DIR / "train_dataset.csv"
    test_path = FEATURES_DIR / "test_dataset.csv"
    assert train_path.exists(), f"Train dataset missing: {train_path}"
    assert test_path.exists(), f"Test dataset missing: {test_path}"
    return pd.read_csv(train_path), pd.read_csv(test_path)


def test_load_merged_dataset(merged_data):
    """Verify merged dataset dimensions, source composition, and label mappings."""
    assert len(merged_data) == 326
    assert set(CANONICAL_FEATURES).issubset(set(merged_data.columns))
    assert "dataset_version" in merged_data.columns
    assert "record_id" in merged_data.columns
    assert "label" in merged_data.columns

    # Verify provenance breakdown
    v101_count = (merged_data["dataset_version"] == "v1.0.1").sum()
    v100_count = (merged_data["dataset_version"] == "v1.0.0").sum()
    assert v101_count == 300
    assert v100_count == 26

    # Verify label mapping: Term = 0, Preterm = 1
    unique_labels = set(merged_data["label"].unique())
    assert unique_labels == {0, 1}
    assert (merged_data["label"] == 0).sum() == 275
    assert (merged_data["label"] == 1).sum() == 51

    # Verify 5 non-pregnant control records were excluded
    assert not merged_data["record_id"].str.contains("tpehgt_n").any()


def test_no_missing_or_infinite_features(merged_data):
    """Verify clean dataset has zero NaN, null, or infinite feature values."""
    feature_vals = merged_data[CANONICAL_FEATURES].to_numpy()
    assert not np.isnan(feature_vals).any(), "Found NaN in feature matrix"
    assert not np.isinf(feature_vals).any(), "Found infinite values in feature matrix"


def test_feature_column_validation():
    """Verify that EHGClassifier enforces the 5 canonical features and excludes metadata."""
    clf = EHGClassifier()

    # 1. Missing required canonical feature should raise ValueError
    invalid_df = pd.DataFrame({
        "sampen_imf1": [0.8],
        "median_freq_imf1_hz": [2.8],
        # Missing rms_imf1, rms_imf2, hilbert_par_imf1
    })
    with pytest.raises(ValueError, match="Missing required canonical feature"):
        clf._validate_X(invalid_df, require_canonical=True)

    # 2. DataFrame with metadata and extra non-feature columns should have them safely stripped
    df_with_meta = pd.DataFrame({
        "record_id": ["rec_001"],
        "dataset_version": ["v1.0.1"],
        "label": [0],
        "sampen_imf1": [0.8],
        "median_freq_imf1_hz": [2.8],
        "rms_imf1": [0.002],
        "rms_imf2": [0.003],
        "hilbert_par_imf1": [10.5],
    })
    X_val, feat_names = clf._validate_X(df_with_meta)
    assert X_val.shape == (1, 5)
    assert feat_names == CANONICAL_FEATURES


def test_train_test_separation(train_test_data):
    """Verify strict disjointness between train and test splits (no data leakage)."""
    train_df, test_df = train_test_data
    assert len(train_df) == 260
    assert len(test_df) == 66

    train_ids = set(train_df["record_id"])
    test_ids = set(test_df["record_id"])
    overlap = train_ids.intersection(test_ids)
    assert len(overlap) == 0, f"Data leakage: overlapping records {overlap}"


def test_no_oversampling_of_test_data(train_test_data):
    """Verify the test set is completely untouched and not resampled."""
    _, test_df = train_test_data
    assert len(test_df) == 66
    assert (test_df["label"] == 0).sum() == 56
    assert (test_df["label"] == 1).sum() == 10
    # No artificial duplicates
    assert test_df["record_id"].nunique() == 66


def test_model_training_and_prediction(train_test_data):
    """Verify Gaussian RBF-SVM pipeline training, prediction, and probability properties."""
    train_df, test_df = train_test_data

    clf = EHGClassifier(C=1.0, gamma="scale")
    clf.fit(train_df, train_df["label"])

    assert clf.is_fitted_
    preds = clf.predict(test_df)
    probas = clf.predict_proba(test_df)

    assert len(preds) == len(test_df)
    assert set(preds).issubset({0, 1})
    assert probas.shape == (len(test_df), 2)
    # Probabilities must sum to 1
    np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-5)


def test_save_and_load_trained_model(tmp_path, train_test_data):
    """Verify trained pipeline serialization, persistence, and identical reload predictions."""
    train_df, test_df = train_test_data

    clf = EHGClassifier(C=10.0, gamma=0.01)
    clf.fit(train_df, train_df["label"])

    save_path = tmp_path / "test_gaussian_rbf_svm.joblib"
    clf.save(save_path)
    assert save_path.exists()

    reloaded_clf = EHGClassifier.load(save_path)
    orig_preds = clf.predict(test_df)
    reloaded_preds = reloaded_clf.predict(test_df)

    np.testing.assert_array_equal(orig_preds, reloaded_preds)


def test_reproducibility(train_test_data):
    """Verify deterministic training results across multiple runs with identical random seed."""
    train_df, test_df = train_test_data

    clf1 = EHGClassifier(C=1.0, gamma="scale")
    clf1.fit(train_df, train_df["label"])
    preds1 = clf1.predict(test_df)

    clf2 = EHGClassifier(C=1.0, gamma="scale")
    clf2.fit(train_df, train_df["label"])
    preds2 = clf2.predict(test_df)

    np.testing.assert_array_equal(preds1, preds2)


def test_valid_evaluation_metrics(train_test_data):
    """Verify test metrics computation produces mathematically valid results."""
    train_df, test_df = train_test_data

    clf = EHGClassifier()
    clf.fit(train_df, train_df["label"])
    metrics = clf.evaluate(test_df, test_df["label"])

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["balanced_accuracy"] <= 1.0
    assert 0.0 <= metrics["sensitivity"] <= 1.0
    assert 0.0 <= metrics["specificity"] <= 1.0
    assert metrics["tp"] + metrics["fp"] + metrics["tn"] + metrics["fn"] == len(test_df)
    assert metrics["prediction_time_sec"] > 0.0


def test_inference_on_new_sample():
    """Verify loading the production model checkpoint and predicting on a new unseen sample."""
    model_path = MODELS_DIR / "gaussian_rbf_svm.joblib"
    assert model_path.exists(), f"Production model missing: {model_path}"

    model = EHGClassifier.load(model_path)

    # Provide a representative new 1-minute EHG record vector
    sample_record = pd.DataFrame([{
        "sampen_imf1": 0.825,
        "median_freq_imf1_hz": 2.85,
        "rms_imf1": 0.0024,
        "rms_imf2": 0.0031,
        "hilbert_par_imf1": 5.95,
    }])

    pred = model.predict(sample_record)
    prob = model.predict_proba(sample_record)

    assert len(pred) == 1
    assert pred[0] in [0, 1]
    assert prob.shape == (1, 2)
    assert 0.0 <= prob[0, 1] <= 1.0
    print(f"Sample prediction: {'Preterm' if pred[0] == 1 else 'Term'} (P(Preterm)={prob[0, 1]:.4f})")
