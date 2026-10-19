"""
Unit tests for single-record feature extraction module.
"""

import numpy as np
import pytest
from src.feature_extraction import (
    sample_entropy,
    median_frequency,
    rms,
    hilbert_peak_to_average_power_ratio,
    extract_features
)


def test_rms():
    signal = np.array([3.0, 4.0, 0.0, 0.0])
    # sqrt((9 + 16 + 0 + 0) / 4) = sqrt(25/4) = 2.5
    assert np.isclose(rms(signal), 2.5)


def test_hilbert_par():
    # Constant amplitude sine wave has PAR approx 1.0 (or low constant ratio)
    t = np.linspace(0, 1, 1000)
    signal = np.sin(2 * np.pi * 5 * t)
    par = hilbert_peak_to_average_power_ratio(signal)
    assert isinstance(par, float)
    assert par >= 1.0


def test_median_frequency():
    fs = 20.0
    t = np.linspace(0, 60, 1200, endpoint=False)
    signal = np.sin(2 * np.pi * 2.0 * t)  # 2 Hz sine wave
    med_freq = median_frequency(signal, sampling_frequency=fs)
    assert isinstance(med_freq, float)
    assert np.isclose(med_freq, 2.0, atol=0.1)


def test_sample_entropy():
    np.random.seed(42)
    signal = np.random.randn(1200)
    samp_en = sample_entropy(signal, m=2, r_ratio=0.2)
    assert isinstance(samp_en, float)
    assert samp_en > 0.0


def test_extract_features_valid():
    np.random.seed(42)
    imf1 = np.random.randn(1200)
    imf2 = np.random.randn(1200)
    feats = extract_features(imf1, imf2, sampling_frequency=20.0)

    expected_keys = {
        'sampen_imf1',
        'median_frequency_imf1_hz',
        'rms_imf1',
        'rms_imf2',
        'hilbert_par_imf1'
    }
    assert set(feats.keys()) == expected_keys
    for val in feats.values():
        assert isinstance(val, float)
        assert np.isfinite(val)


def test_extract_features_invalid_length():
    imf1 = np.random.randn(1000)
    imf2 = np.random.randn(1200)
    with pytest.raises(ValueError, match="IMF-1 length must be 1200"):
        extract_features(imf1, imf2, sampling_frequency=20.0)


def test_extract_features_nan():
    imf1 = np.random.randn(1200)
    imf1[10] = np.nan
    imf2 = np.random.randn(1200)
    with pytest.raises(ValueError, match="IMF signal contains NaN values"):
        extract_features(imf1, imf2, sampling_frequency=20.0)


def test_extract_features_sampling_rates():
    """Verify feature extraction accepts both 20.0 Hz and 20.00011 Hz representations."""
    np.random.seed(42)
    imf1 = np.random.randn(1200)
    imf2 = np.random.randn(1200)

    # 1. Test nominal 20.0 Hz
    feats_20 = extract_features(imf1, imf2, sampling_frequency=20.0)
    assert isinstance(feats_20, dict)
    assert len(feats_20) == 5

    # 2. Test 20.00011 Hz (TPEHGDB header representation)
    feats_20_00011 = extract_features(imf1, imf2, sampling_frequency=20.00011)
    assert isinstance(feats_20_00011, dict)
    assert len(feats_20_00011) == 5

    # Verify feature values are virtually identical
    for k in feats_20:
        assert np.isclose(feats_20[k], feats_20_00011[k], rtol=1e-3)

    # 3. Test invalid sampling rate outside tolerance (e.g. 10.0 Hz, 50.0 Hz)
    with pytest.raises(ValueError, match="Sampling frequency must be 20 Hz"):
        extract_features(imf1, imf2, sampling_frequency=10.0)
    with pytest.raises(ValueError, match="Sampling frequency must be 20 Hz"):
        extract_features(imf1, imf2, sampling_frequency=50.0)

