"""
Unit test suite for Objective 4: EHG Rapid Triage Pipeline.
"""

import os
import tempfile
import numpy as np
import pandas as pd
import pytest

from src.config import PipelineConfig, SignalConfig, ensure_directories
from src.data_loader import TPEHGDataLoader
from src.metadata import TPEHGMetadataParser
from src.preprocessing import EHGPreprocessor
from src.emd_processing import EMDProcessor
from src.feature_extraction import EHGFeatureExtractor
from src.feature_selection import FeatureSelector
from src.model import EHGClassifier
from src.evaluation import EHGEvaluator


def test_config_initialization():
    """Verify configuration parameters and directory creation."""
    config = PipelineConfig()
    assert config.signal.sampling_rate == 20.0
    assert config.signal.bandpass_low == 0.1
    assert config.signal.bandpass_high == 4.0
    ensure_directories()
    assert config.data_dir.exists()


def test_synthetic_data_generation():
    """Verify synthetic EHG signal generation for preterm and term cases."""
    signals_p, meta_p = TPEHGDataLoader.generate_synthetic_ehg(duration_sec=30.0, is_preterm=True)
    signals_t, meta_t = TPEHGDataLoader.generate_synthetic_ehg(duration_sec=30.0, is_preterm=False)

    assert signals_p.shape == (600, 3)  # 30s * 20Hz = 600 samples
    assert meta_p['is_preterm'] is True
    assert meta_t['is_preterm'] is False


def test_metadata_parsing():
    """Verify metadata parser extracts attributes from comments."""
    comments = [
        "Gestation: 31.5 weeks",
        "Delivery: 33.0 weeks",
        "Age: 26",
        "Parity: 1"
    ]
    meta = TPEHGMetadataParser.parse_header_comments(comments)
    assert meta['gestation_rec_weeks'] == 31.5
    assert meta['gestation_del_weeks'] == 33.0
    assert meta['is_preterm'] is True
    assert meta['maternal_age'] == 26
    assert meta['parity'] == 1


def test_preprocessing():
    """Verify bandpass filtering and signal normalization."""
    raw_signal, _ = TPEHGDataLoader.generate_synthetic_ehg(duration_sec=20.0)
    preprocessor = EHGPreprocessor()

    filtered = preprocessor.bandpass_filter(raw_signal, lowcut=0.1, highcut=4.0, fs=20.0)
    assert filtered.shape == raw_signal.shape

    normalized = preprocessor.normalize(filtered, method='zscore')
    assert np.abs(np.mean(normalized[:, 0])) < 1e-5
    assert np.abs(np.std(normalized[:, 0]) - 1.0) < 1e-3


def test_emd_decomposition():
    """Verify EMD processing and IMF energy calculation."""
    raw_signal, _ = TPEHGDataLoader.generate_synthetic_ehg(duration_sec=10.0)
    emd_proc = EMDProcessor()

    imfs = emd_proc.decompose(raw_signal[:, 0])
    assert imfs.ndim == 2
    assert len(imfs) > 0

    energies = emd_proc.compute_imf_energies(imfs)
    assert len(energies) == len(imfs)
    assert np.isclose(np.sum(energies), 1.0)


def test_feature_extraction():
    """Verify feature extractor output schema and values."""
    raw_signal, _ = TPEHGDataLoader.generate_synthetic_ehg(duration_sec=10.0)
    extractor = EHGFeatureExtractor()

    feats = extractor.extract_all_features(raw_signal[:, 0], channel_prefix="ch1")
    assert 'ch1_rms' in feats
    assert 'ch1_median_frequency' in feats
    assert 'ch1_sample_entropy' in feats
    assert isinstance(feats['ch1_rms'], float)


def test_feature_selection():
    """Verify feature selection and RFE filtering."""
    np.random.seed(42)
    n_samples = 40
    X_data = {
        'f1': np.random.randn(n_samples),
        'f2': np.random.randn(n_samples),
        'f3': np.random.randn(n_samples) * 0.01,
        'f4_dup': np.random.randn(n_samples)
    }
    X = pd.DataFrame(X_data)
    X['f4_dup'] = X['f1'] * 0.999  # Highly collinear with f1
    y = np.array([1 if i < 20 else 0 for i in range(n_samples)])

    selector = FeatureSelector()
    X_clean = selector.remove_collinear_features(X, threshold=0.90)
    assert 'f4_dup' not in X_clean.columns

    X_selected = selector.fit_transform(X, y, top_k=2)
    assert X_selected.shape[1] <= 2


def test_model_training_and_eval():
    """Verify end-to-end classification training, prediction, and evaluation."""
    np.random.seed(42)
    n_samples = 30
    X = pd.DataFrame({
        'feat1': np.random.randn(n_samples),
        'feat2': np.random.randn(n_samples) + np.array([2.0 if i < 15 else 0.0 for i in range(n_samples)])
    })
    y = np.array([1 if i < 15 else 0 for i in range(n_samples)])

    clf = EHGClassifier()
    clf.fit(X, y)

    preds = clf.predict(X)
    probas = clf.predict_proba(X)

    assert len(preds) == n_samples
    assert probas.shape == (n_samples, 2)

    metrics = EHGEvaluator.compute_metrics(y, preds, probas)
    assert 'sensitivity' in metrics
    assert 'specificity' in metrics
    assert 'roc_auc' in metrics
    assert 0.0 <= metrics['roc_auc'] <= 1.0


def test_model_serialization(tmp_path):
    """Verify model saving and loading."""
    X = np.random.randn(20, 3)
    y = np.random.randint(0, 2, 20)

    clf = EHGClassifier()
    clf.fit(X, y)

    save_path = tmp_path / "model.joblib"
    clf.save(save_path)

    loaded_clf = EHGClassifier.load(save_path)
    loaded_preds = loaded_clf.predict(X)

    np.testing.assert_array_equal(clf.predict(X), loaded_preds)
