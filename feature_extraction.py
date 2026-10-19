"""
Feature extraction module for EHG signal analysis.

Provides reusable functions and pipeline classes for extracting time-domain,
frequency-domain, entropy, and Hilbert features from decomposed IMFs.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.signal import hilbert, welch
from scipy.spatial.distance import cdist
from scipy.stats import kurtosis, skew

try:
    from src.config import ProjectConfig, DatasetConfig, EMDConfig
except ImportError:
    pass

from src.emd_processing import EMDProcessor


@dataclass
class SignalConfig:
    sampling_rate: float = 20.0
    bandpass_low: float = 0.08
    bandpass_high: float = 4.0


@dataclass
class FeatureConfig:
    emd_features: List[str] = field(default_factory=lambda: ["imf_energy_ratio", "imf_mean_inst_freq", "imf_mean_inst_amp"])


def sample_entropy(signal: np.ndarray, m: int = 2, r_ratio: float = 0.2) -> float:
    """
    Calculate Sample Entropy (SampEn) for a 1D signal.

    Parameters
    ----------
    signal : np.ndarray
        1D time-series signal array.
    m : int, default=2
        Embedding dimension.
    r_ratio : float, default=0.2
        Tolerance factor as a fraction of signal standard deviation.

    Returns
    -------
    float
        Sample entropy value.

    Raises
    ------
    ValueError
        If signal is invalid, length is insufficient, std is zero, or log(0)/division by zero occurs.
    """
    signal = np.asarray(signal, dtype=np.float64)
    N = len(signal)

    if N < m + 1:
        raise ValueError(f"Insufficient samples ({N}) for sample entropy with m={m}.")

    std_val = float(np.std(signal))
    if std_val < 1e-12 or np.isnan(std_val):
        raise ValueError(f"Standard deviation of signal is zero or invalid ({std_val}), cannot compute sample entropy.")

    r = r_ratio * std_val

    # Construct template vectors of length m and m+1 for indices 0..N-m-1
    # Both A and B use the N - m vectors starting at i = 0 .. N-m-1
    n_templates = N - m
    x_m = np.array([signal[i:i + m] for i in range(n_templates)])
    x_m1 = np.array([signal[i:i + m + 1] for i in range(n_templates)])

    # Chebyshev distance matrices with chunking to prevent memory exhaustion on long windows
    chunk_size = 2000
    if n_templates <= chunk_size:
        d_m = cdist(x_m, x_m, metric='chebyshev')
        d_m1 = cdist(x_m1, x_m1, metric='chebyshev')
        np.fill_diagonal(d_m, np.inf)
        np.fill_diagonal(d_m1, np.inf)
        B = int(np.sum(d_m <= r))
        A = int(np.sum(d_m1 <= r))
    else:
        B, A = 0, 0
        for i in range(0, n_templates, chunk_size):
            end = min(i + chunk_size, n_templates)
            dm_chunk = cdist(x_m[i:end], x_m, metric='chebyshev')
            dm1_chunk = cdist(x_m1[i:end], x_m1, metric='chebyshev')
            for local_idx in range(end - i):
                dm_chunk[local_idx, i + local_idx] = np.inf
                dm1_chunk[local_idx, i + local_idx] = np.inf
            B += int(np.sum(dm_chunk <= r))
            A += int(np.sum(dm1_chunk <= r))

    if B == 0:
        raise ValueError(f"Sample entropy calculation failed: zero matching vector pairs found at length m={m} (B=0).")
    if A == 0:
        raise ValueError(f"Sample entropy calculation failed: zero matching vector pairs found at length m+1={m+1} (A=0).")

    samp_en = -np.log(A / B)
    return float(samp_en)


def median_frequency(signal: np.ndarray, sampling_frequency: float) -> float:
    """
    Calculate FFT-based median frequency of a signal, excluding the DC component.

    Parameters
    ----------
    signal : np.ndarray
        1D signal array.
    sampling_frequency : float
        Sampling frequency in Hz.

    Returns
    -------
    float
        Median frequency in Hz.

    Raises
    ------
    ValueError
        If signal is empty or spectral power is zero.
    """
    signal = np.asarray(signal, dtype=np.float64)
    if len(signal) == 0:
        raise ValueError("Cannot calculate median frequency of an empty signal.")

    freqs = np.fft.rfftfreq(len(signal), d=1.0 / sampling_frequency)
    spectrum = np.fft.rfft(signal)
    power = np.abs(spectrum) ** 2

    # Exclude DC component (0 Hz)
    freqs_no_dc = freqs[1:]
    power_no_dc = power[1:]

    total_power = np.sum(power_no_dc)
    if total_power <= 0 or np.isnan(total_power):
        raise ValueError("Total spectral power (excluding DC) is zero or invalid.")

    cumulative_power = np.cumsum(power_no_dc)
    med_idx = np.where(cumulative_power >= 0.5 * total_power)[0]

    if len(med_idx) == 0:
        raise ValueError("Could not locate 50% cumulative power threshold.")

    return float(freqs_no_dc[med_idx[0]])


def rms(signal: np.ndarray) -> float:
    """
    Calculate Root Mean Square (RMS) of a signal.

    Parameters
    ----------
    signal : np.ndarray
        1D signal array.

    Returns
    -------
    float
        RMS value.
    """
    signal = np.asarray(signal, dtype=np.float64)
    if len(signal) == 0:
        raise ValueError("Cannot calculate RMS of an empty signal.")
    return float(np.sqrt(np.mean(signal ** 2)))


def hilbert_peak_to_average_power_ratio(signal: np.ndarray) -> float:
    """
    Calculate Hilbert-transform Peak-to-Average Power Ratio (PAR) of a signal.

    Parameters
    ----------
    signal : np.ndarray
        1D signal array.

    Returns
    -------
    float
        Peak-to-Average Power Ratio.

    Raises
    ------
    ValueError
        If mean instantaneous power is zero.
    """
    signal = np.asarray(signal, dtype=np.float64)
    if len(signal) == 0:
        raise ValueError("Cannot calculate Hilbert PAR of an empty signal.")

    analytic_signal = hilbert(signal)
    inst_amplitude = np.abs(analytic_signal)
    inst_power = inst_amplitude ** 2

    mean_power = np.mean(inst_power)
    if mean_power <= 0 or np.isnan(mean_power):
        raise ValueError("Mean instantaneous power is zero or invalid.")

    peak_power = np.max(inst_power)
    return float(peak_power / mean_power)


def extract_canonical_features(
    imf1: np.ndarray,
    imf2: np.ndarray,
    sampling_frequency: float,
    expected_samples: Optional[int] = None,
) -> Dict[str, float]:
    """
    Extract the 5 canonical EHG features for rapid triage across any valid window length.

    Parameters
    ----------
    imf1 : np.ndarray
        Decomposed IMF-1 signal array.
    imf2 : np.ndarray
        Decomposed IMF-2 signal array.
    sampling_frequency : float
        Actual sampling frequency in Hz from signal header.
    expected_samples : Optional[int]
        If provided, validates that signal length strictly matches expected_samples.

    Returns
    -------
    Dict[str, float]
        Dictionary with keys:
        - sampen_imf1
        - median_freq_imf1_hz (and median_frequency_imf1_hz alias)
        - rms_imf1
        - rms_imf2
        - hilbert_par_imf1
    """
    imf1 = np.asarray(imf1, dtype=np.float64)
    imf2 = np.asarray(imf2, dtype=np.float64)

    if len(imf1) != len(imf2):
        raise ValueError(f"IMF-1 length ({len(imf1)}) does not match IMF-2 length ({len(imf2)}).")

    if expected_samples is not None and len(imf1) != expected_samples:
        raise ValueError(f"IMF length must be {expected_samples}, got {len(imf1)}.")

    if len(imf1) < 10:
        raise ValueError(f"Insufficient samples ({len(imf1)}) for feature extraction.")

    if np.isnan(imf1).any() or np.isnan(imf2).any():
        raise ValueError("IMF signal contains NaN values.")
    if np.isinf(imf1).any() or np.isinf(imf2).any():
        raise ValueError("IMF signal contains Infinite values.")

    sampen_val = sample_entropy(imf1, m=2, r_ratio=0.2)
    med_freq_val = median_frequency(imf1, sampling_frequency)
    rms_imf1_val = rms(imf1)
    rms_imf2_val = rms(imf2)
    hilbert_par_val = hilbert_peak_to_average_power_ratio(imf1)

    return {
        'sampen_imf1': sampen_val,
        'median_freq_imf1_hz': med_freq_val,
        'median_frequency_imf1_hz': med_freq_val,
        'rms_imf1': rms_imf1_val,
        'rms_imf2': rms_imf2_val,
        'hilbert_par_imf1': hilbert_par_val
    }


def extract_features(imf1: np.ndarray, imf2: np.ndarray, sampling_frequency: float) -> Dict[str, float]:
    """
    Validate signals and extract the 5 target numerical features for EHG single-record validation.

    Parameters
    ----------
    imf1 : np.ndarray
        IMF-1 signal array (must be 1200 samples).
    imf2 : np.ndarray
        IMF-2 signal array (must be 1200 samples).
    sampling_frequency : float
        Sampling frequency in Hz (must be 20 Hz).

    Returns
    -------
    Dict[str, float]
        Dictionary containing sampen_imf1, median_frequency_imf1_hz, rms_imf1, rms_imf2, hilbert_par_imf1.
    """
    imf1 = np.asarray(imf1, dtype=np.float64)
    imf2 = np.asarray(imf2, dtype=np.float64)

    # Validation checks
    if len(imf1) != 1200:
        raise ValueError(f"IMF-1 length must be 1200, got {len(imf1)}.")
    if len(imf2) != 1200:
        raise ValueError(f"IMF-2 length must be 1200, got {len(imf2)}.")
    if np.isnan(imf1).any() or np.isnan(imf2).any():
        raise ValueError("IMF signal contains NaN values.")
    if np.isinf(imf1).any() or np.isinf(imf2).any():
        raise ValueError("IMF signal contains Infinite values.")
    fs_tolerance = 0.01  # Documented tolerance in Hz for TPEHGDB v1.0.1 (nominal 20 Hz vs 20.00011 Hz headers)
    if abs(float(sampling_frequency) - 20.0) > fs_tolerance:
        raise ValueError(f"Sampling frequency must be 20 Hz (+/- {fs_tolerance} Hz tolerance for TPEHGDB headers), got {sampling_frequency} Hz.")

    feats = extract_canonical_features(imf1, imf2, sampling_frequency, expected_samples=1200)

    return {
        'sampen_imf1': feats['sampen_imf1'],
        'median_frequency_imf1_hz': feats['median_freq_imf1_hz'],
        'rms_imf1': feats['rms_imf1'],
        'rms_imf2': feats['rms_imf2'],
        'hilbert_par_imf1': feats['hilbert_par_imf1']
    }


class EHGFeatureExtractor:
    """Feature extraction pipeline for raw signals and decomposed IMFs."""

    def __init__(
        self,
        signal_config: Optional[SignalConfig] = None,
        feature_config: Optional[FeatureConfig] = None
    ):
        self.signal_config = signal_config or SignalConfig()
        self.feature_config = feature_config or FeatureConfig()
        self.emd_processor = EMDProcessor()

    # --- Time Domain Features ---
    @staticmethod
    def extract_time_features(signal: np.ndarray) -> Dict[str, float]:
        """Extract standard time-domain statistical metrics."""
        rms_val = rms(signal)
        std = float(np.std(signal))
        var = float(np.var(signal))
        mean_val = float(np.mean(signal))
        peak_amp = float(np.max(np.abs(signal)))
        skew_val = float(skew(signal))
        kurt_val = float(kurtosis(signal))
        p2p = float(np.ptp(signal))

        # Zero-Crossing Rate
        zero_crossings = np.where(np.diff(np.signbit(signal)))[0]
        zcr = float(len(zero_crossings) / len(signal))

        return {
            'mean': mean_val,
            'std': std,
            'var': var,
            'rms': rms_val,
            'peak_amplitude': peak_amp,
            'skewness': skew_val,
            'kurtosis': kurt_val,
            'zero_crossing_rate': zcr,
            'peak_to_peak': p2p
        }

    # --- Frequency Domain Features ---
    def extract_freq_features(
        self, signal: np.ndarray, fs: Optional[float] = None
    ) -> Dict[str, float]:
        """Extract spectral features using Welch Power Spectral Density."""
        fs = fs or self.signal_config.sampling_rate
        nperseg = min(len(signal), int(fs * 4))
        freqs, psd = welch(signal, fs=fs, nperseg=nperseg)

        # Restrict to bandpass range (0.1 - 4.0 Hz)
        valid_idx = (freqs >= self.signal_config.bandpass_low) & (freqs <= self.signal_config.bandpass_high)
        freqs = freqs[valid_idx]
        psd = psd[valid_idx]

        if len(psd) == 0 or np.sum(psd) == 0:
            return {
                'median_frequency': 0.0,
                'peak_frequency': 0.0,
                'mean_frequency': 0.0,
                'spectral_entropy': 0.0,
                'total_power': 0.0
            }

        total_power = float(np.sum(psd))
        cumulative_power = np.cumsum(psd)
        med_idx = np.where(cumulative_power >= total_power / 2.0)[0]
        median_freq = float(freqs[med_idx[0]]) if len(med_idx) > 0 else float(freqs[0])

        peak_freq = float(freqs[np.argmax(psd)])
        mean_freq = float(np.sum(freqs * psd) / total_power)

        # Spectral Entropy
        psd_norm = psd / total_power
        spectral_entropy = float(-np.sum(psd_norm * np.log2(psd_norm + 1e-12)))

        return {
            'median_frequency': median_freq,
            'peak_frequency': peak_freq,
            'mean_frequency': mean_freq,
            'spectral_entropy': spectral_entropy,
            'total_power': total_power
        }

    # --- Entropy & Nonlinear Features ---
    @staticmethod
    def extract_sample_entropy(signal: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
        """Sample entropy wrapper."""
        try:
            return sample_entropy(signal, m=m, r_ratio=r_factor)
        except ValueError:
            return 0.0

    # --- EMD & IMF Features ---
    def extract_emd_features(self, signal: np.ndarray) -> Dict[str, float]:
        """Extract features across decomposed Intrinsic Mode Functions (IMFs)."""
        imfs = self.emd_processor.decompose(signal)
        n_imfs = len(imfs)

        energy_ratios = self.emd_processor.compute_imf_energies(imfs)
        inst_amp, inst_freq = self.emd_processor.compute_instantaneous_frequency(
            imfs, fs=self.signal_config.sampling_rate
        )

        features = {}
        for i in range(min(n_imfs, self.feature_config.emd_features.count('imf_energy_ratio') + 6)):
            features[f'imf_{i+1}_energy_ratio'] = float(energy_ratios[i]) if i < len(energy_ratios) else 0.0
            features[f'imf_{i+1}_mean_inst_freq'] = float(np.mean(inst_freq[i])) if i < len(inst_freq) else 0.0
            features[f'imf_{i+1}_mean_inst_amp'] = float(np.mean(inst_amp[i])) if i < len(inst_amp) else 0.0

        return features

    def extract_all_features(self, signal: np.ndarray, channel_prefix: str = "ch1") -> Dict[str, float]:
        """Extract complete combined feature vector for a signal channel."""
        time_feats = self.extract_time_features(signal)
        freq_feats = self.extract_freq_features(signal)
        samp_en = self.extract_sample_entropy(signal)
        emd_feats = self.extract_emd_features(signal)

        combined = {}
        for k, v in time_feats.items():
            combined[f"{channel_prefix}_{k}"] = v
        for k, v in freq_feats.items():
            combined[f"{channel_prefix}_{k}"] = v
        combined[f"{channel_prefix}_sample_entropy"] = samp_en
        for k, v in emd_feats.items():
            combined[f"{channel_prefix}_{k}"] = v

        return combined

    def extract_multichannel_features(
        self, signals: np.ndarray, channel_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Extract feature dictionary across multi-channel signals array (samples, channels)."""
        if signals.ndim == 1:
            signals = signals[:, np.newaxis]

        n_channels = signals.shape[1]
        channel_names = channel_names or [f"ch{i+1}" for i in range(n_channels)]

        all_feats = {}
        for ch in range(n_channels):
            ch_feats = self.extract_all_features(signals[:, ch], channel_prefix=channel_names[ch])
            all_feats.update(ch_feats)

        return all_feats

