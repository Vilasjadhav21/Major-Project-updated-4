"""
Preprocessing module for EHG signal processing.

Includes Butterworth digital bandpass filtering, baseline wander removal,
window segmentation, downsampling, and Z-score normalization.
"""

from typing import List, Optional, Tuple, Union
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, medfilt

from src.config import SignalConfig


class EHGPreprocessor:
    """Preprocessor for raw EHG uterine EMG signals."""

    def __init__(self, config: Optional[SignalConfig] = None):
        self.config = config or SignalConfig()

    def bandpass_filter(
        self,
        signal: np.ndarray,
        lowcut: Optional[float] = None,
        highcut: Optional[float] = None,
        fs: Optional[float] = None,
        order: Optional[int] = None
    ) -> np.ndarray:
        """
        Apply zero-phase Butterworth bandpass filter to 1D or 2D signals.
        """
        lowcut = lowcut or self.config.bandpass_low
        highcut = highcut or self.config.bandpass_high
        fs = fs or self.config.sampling_rate
        order = order or self.config.filter_order

        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        
        # Ensure cutoffs stay within valid Nyquist range
        low = max(0.001, min(low, 0.99))
        high = max(low + 0.001, min(high, 0.99))

        b, a = butter(order, [low, high], btype='band')

        if signal.ndim == 1:
            return filtfilt(b, a, signal)
        else:
            filtered = np.zeros_like(signal)
            for ch in range(signal.shape[1]):
                filtered[:, ch] = filtfilt(b, a, signal[:, ch])
            return filtered

    def remove_baseline_wander(
        self, signal: np.ndarray, kernel_size_sec: float = 2.0
    ) -> np.ndarray:
        """
        Remove baseline wander using median filtering.
        """
        kernel_samples = int(kernel_size_sec * self.config.sampling_rate)
        if kernel_samples % 2 == 0:
            kernel_samples += 1

        if signal.ndim == 1:
            baseline = medfilt(signal, kernel_size=kernel_samples)
            return signal - baseline
        else:
            cleaned = np.zeros_like(signal)
            for ch in range(signal.shape[1]):
                baseline = medfilt(signal[:, ch], kernel_size=kernel_samples)
                cleaned[:, ch] = signal[:, ch] - baseline
            return cleaned

    def notch_filter(self, signal: np.ndarray, freq: float = 50.0, quality: float = 30.0) -> np.ndarray:
        """Apply narrow notch filter to remove powerline interference."""
        nyq = 0.5 * self.config.sampling_rate
        norm_freq = freq / nyq

        if norm_freq >= 1.0:
            return signal  # Notch freq higher than Nyquist frequency

        b, a = iirnotch(norm_freq, quality)
        if signal.ndim == 1:
            return filtfilt(b, a, signal)
        else:
            notched = np.zeros_like(signal)
            for ch in range(signal.shape[1]):
                notched[:, ch] = filtfilt(b, a, signal[:, ch])
            return notched

    def normalize(self, signal: np.ndarray, method: str = 'zscore') -> np.ndarray:
        """Normalize signal values (zscore or minmax)."""
        if method == 'zscore':
            mean = np.mean(signal, axis=0, keepdims=True)
            std = np.std(signal, axis=0, keepdims=True) + 1e-8
            return (signal - mean) / std
        elif method == 'minmax':
            min_val = np.min(signal, axis=0, keepdims=True)
            max_val = np.max(signal, axis=0, keepdims=True)
            range_val = (max_val - min_val) + 1e-8
            return (signal - min_val) / range_val
        else:
            raise ValueError(f"Unknown normalization method: {method}")

    def segment_windows(
        self, signal: np.ndarray, window_sec: Optional[float] = None, overlap_sec: Optional[float] = None
    ) -> np.ndarray:
        """
        Segment continuous signal into overlapping windows.

        Returns array of shape: (n_windows, window_samples, n_channels)
        """
        window_sec = window_sec or self.config.window_length_sec
        overlap_sec = overlap_sec or self.config.window_overlap_sec
        fs = self.config.sampling_rate

        win_samples = int(window_sec * fs)
        step_samples = int((window_sec - overlap_sec) * fs)

        if signal.ndim == 1:
            signal = signal[:, np.newaxis]

        n_samples, n_channels = signal.shape

        if n_samples < win_samples:
            # Pad signal if shorter than window length
            pad_size = win_samples - n_samples
            signal = np.pad(signal, ((0, pad_size), (0, 0)), mode='reflect')
            n_samples = win_samples

        windows = []
        for start in range(0, n_samples - win_samples + 1, step_samples):
            windows.append(signal[start:start + win_samples, :])

        return np.array(windows)

    def preprocess_pipeline(self, raw_signal: np.ndarray) -> np.ndarray:
        """Run standard preprocessing workflow."""
        # 1. Bandpass filter (0.1 - 4.0 Hz)
        bp_signal = self.bandpass_filter(raw_signal)

        # 2. Optional baseline wander removal
        bw_signal = self.remove_baseline_wander(bp_signal)

        # 3. Z-score normalization
        norm_signal = self.normalize(bw_signal, method='zscore')

        return norm_signal
