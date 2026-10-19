"""
Empirical Mode Decomposition (EMD) module for EHG signal decomposition.

Decomposes non-stationary uterine EHG signals into Intrinsic Mode Functions (IMFs).
Integrates PyEMD package with robust pure-Python fallback.
"""

from typing import List, Optional, Tuple, Union
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import argrelextrema, hilbert

from src.config import EMDConfig

from PyEMD import EMD

try:
    HAS_PYEMD = True
except ImportError:
    HAS_PYEMD = False


def perform_emd(
    signal: np.ndarray,
    sampling_frequency: float = 20.0,
    expected_samples: Optional[int] = None,
) -> np.ndarray:
    """
    Perform Empirical Mode Decomposition (EMD) on a 1-D EHG signal array using PyEMD.

    Validation rules:
    1. Accept a 1-D NumPy array.
    2. Validate that the input contains sufficient samples (or exact expected_samples if specified).
    3. Validate that the signal contains no NaN values.
    4. Validate that the signal contains no infinite values.
    5. Perform Empirical Mode Decomposition using PyEMD.
    6. Return all extracted IMFs.
    """
    if not isinstance(signal, np.ndarray):
        signal = np.asarray(signal)

    if signal.ndim != 1:
        raise ValueError(f"Input signal must be a 1-D NumPy array. Got shape {signal.shape}")

    if expected_samples is not None and len(signal) != expected_samples:
        raise ValueError(f"Input signal must contain exactly {expected_samples} samples. Got {len(signal)} samples.")

    if len(signal) < 10:
        raise ValueError(f"Input signal is too short ({len(signal)} samples), cannot perform EMD.")

    if np.isnan(signal).any():
        raise ValueError("Input signal contains NaN values.")

    if np.isinf(signal).any():
        raise ValueError("Input signal contains infinite values.")

    emd = EMD()
    imfs = emd(signal)
    return imfs


class EMDProcessor:
    """EMD decomposition engine for multi-channel EHG signals."""

    def __init__(self, config: Optional[EMDConfig] = None):
        self.config = config or EMDConfig()

    def decompose(self, signal_1d: np.ndarray) -> np.ndarray:
        """
        Decompose a single 1D signal array into IMFs.

        Returns array of shape: (n_imfs, n_samples)
        """
        return perform_emd(signal_1d, sampling_frequency=self.config.sampling_frequency)

    def _fallback_emd(self, signal: np.ndarray, max_imfs: int = 6) -> np.ndarray:
        """
        Simplified Empirical Mode Decomposition sifting fallback algorithm.
        """
        n_samples = len(signal)
        residue = signal.copy().astype(float)
        imfs = []
        x = np.arange(n_samples)

        for _ in range(max_imfs):
            if np.std(residue) < 1e-6 or np.max(np.abs(residue)) < 1e-6:
                break

            h = residue.copy()
            for _ in range(15):  # Max sifting iterations
                maxima = argrelextrema(h, np.greater)[0]
                minima = argrelextrema(h, np.less)[0]

                if len(maxima) < 2 or len(minima) < 2:
                    break

                # Boundary conditions
                max_x = np.r_[0, maxima, n_samples - 1]
                max_y = np.r_[h[0], h[maxima], h[-1]]
                min_x = np.r_[0, minima, n_samples - 1]
                min_y = np.r_[h[0], h[minima], h[-1]]

                try:
                    upper_env = CubicSpline(max_x, max_y)(x)
                    lower_env = CubicSpline(min_x, min_y)(x)
                    mean_env = (upper_env + lower_env) / 2.0
                    h = h - mean_env
                except Exception:
                    break

            imfs.append(h)
            residue = residue - h

            if len(maxima) < 2 or len(minima) < 2:
                break

        imfs.append(residue)  # Final residue
        return np.array(imfs[:max_imfs])

    def compute_imf_energies(self, imfs: np.ndarray) -> np.ndarray:
        """Compute relative energy ratio for each extracted IMF."""
        energies = np.sum(imfs ** 2, axis=1)
        total_energy = np.sum(energies) + 1e-8
        return energies / total_energy

    def compute_instantaneous_frequency(
        self, imfs: np.ndarray, fs: float = 20.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Hilbert-Huang Transform instantaneous amplitude and frequency per IMF.

        Returns:
            Tuple of (instantaneous_amplitude, instantaneous_frequency)
            both of shape (n_imfs, n_samples)
        """
        n_imfs, n_samples = imfs.shape
        inst_amp = np.zeros((n_imfs, n_samples))
        inst_freq = np.zeros((n_imfs, n_samples))

        dt = 1.0 / fs

        for i in range(n_imfs):
            analytic_signal = hilbert(imfs[i])
            inst_amp[i] = np.abs(analytic_signal)
            phase = np.unwrap(np.angle(analytic_signal))
            # Derivative of unwrapped phase gives instantaneous frequency
            freq = np.diff(phase) / (2.0 * np.pi * dt)
            freq = np.pad(freq, (0, 1), mode='edge')
            inst_freq[i] = np.maximum(0.0, freq)

        return inst_amp, inst_freq
