"""
Window Extraction module for Objective 4: Rapid Triage (1-Minute Algorithm).

Extracts deterministic or custom time windows from EHG signals based on sampling rate,
channel names, and boundary trimming rules.
"""

from pathlib import Path
from typing import Dict, Tuple, Union
import numpy as np
import wfdb


class EHGWindowExtractor:
    """Extractor for 1-minute EHG window segments."""

    @staticmethod
    def resolve_channel_index(record: wfdb.Record, channel_name_or_idx: Union[str, int]) -> Tuple[int, str]:
        """
        Resolve channel name or index against actual WFDB record header.

        Returns:
            Tuple of (channel_index, canonical_channel_name)
        """
        if isinstance(channel_name_or_idx, int):
            if 0 <= channel_name_or_idx < record.n_sig:
                return channel_name_or_idx, record.sig_name[channel_name_or_idx]
            raise ValueError(f"Channel index {channel_name_or_idx} out of range (0-{record.n_sig - 1}).")

        target = str(channel_name_or_idx).strip()
        sig_names = record.sig_name

        # 1. Exact match
        if target in sig_names:
            idx = sig_names.index(target)
            return idx, sig_names[idx]

        # 2. Standardized mapping for S1 / S2 / S3 subbands
        normalized_target = target.replace("–", "-").replace(" ", "").lower()

        channel_alias_map = {
            "s1raw": "1",
            "s10.08-4hz": "1_DOCFILT-4-0.08-4",
            "s10.08-4": "1_DOCFILT-4-0.08-4",
            "s10.3-3hz": "1_DOCFILT-4-0.3-3",
            "s10.3-3": "1_DOCFILT-4-0.3-3",
            "s10.3-4hz": "1_DOCFILT-4-0.3-4",
            "s10.3-4": "1_DOCFILT-4-0.3-4",
            "s2raw": "2",
            "s20.08-4hz": "2_DOCFILT-4-0.08-4",
            "s20.08-4": "2_DOCFILT-4-0.08-4",
            "s20.3-3hz": "2_DOCFILT-4-0.3-3",
            "s20.3-3": "2_DOCFILT-4-0.3-3",
            "s20.3-4hz": "2_DOCFILT-4-0.3-4",
            "s20.3-4": "2_DOCFILT-4-0.3-4",
            "s3raw": "3",
            "s30.08-4hz": "3_DOCFILT-4-0.08-4",
            "s30.08-4": "3_DOCFILT-4-0.08-4",
            "s30.3-3hz": "3_DOCFILT-4-0.3-3",
            "s30.3-3": "3_DOCFILT-4-0.3-3",
            "s30.3-4hz": "3_DOCFILT-4-0.3-4",
            "s30.3-4": "3_DOCFILT-4-0.3-4",
        }

        canonical_name = channel_alias_map.get(normalized_target, target)
        if canonical_name in sig_names:
            idx = sig_names.index(canonical_name)
            return idx, sig_names[idx]

        # 3. Cross-version harmonization fallback:
        # If S1 filtered subband is requested on a v1.0.0 record containing EHG1_Butter-4-bi-0.08-5
        if "s1" in normalized_target or "1_docfilt" in normalized_target:
            if "EHG1_Butter-4-bi-0.08-5" in sig_names:
                idx = sig_names.index("EHG1_Butter-4-bi-0.08-5")
                return idx, sig_names[idx]
            elif "EHG1" in sig_names:
                idx = sig_names.index("EHG1")
                return idx, sig_names[idx]
        elif "s2" in normalized_target or "2_docfilt" in normalized_target:
            if "EHG2_Butter-4-bi-0.08-5" in sig_names:
                idx = sig_names.index("EHG2_Butter-4-bi-0.08-5")
                return idx, sig_names[idx]
            elif "EHG2" in sig_names:
                idx = sig_names.index("EHG2")
                return idx, sig_names[idx]
        elif "s3" in normalized_target or "3_docfilt" in normalized_target:
            if "EHG3_Butter-4-bi-0.08-5" in sig_names:
                idx = sig_names.index("EHG3_Butter-4-bi-0.08-5")
                return idx, sig_names[idx]
            elif "EHG3" in sig_names:
                idx = sig_names.index("EHG3")
                return idx, sig_names[idx]

        # 4. Partial matching search
        for idx, name in enumerate(sig_names):
            if normalized_target in name.lower().replace("-", ""):
                return idx, name

        raise ValueError(
            f"Could not resolve channel '{channel_name_or_idx}'. Available channels in record header: {sig_names}"
        )

    def extract_one_minute(
        self,
        record: wfdb.Record,
        channel_name: Union[str, int] = "S1 0.08-4 Hz",
        start_time: float = 180.0,
        window_duration: float = 60.0
    ) -> Dict[str, Union[np.ndarray, float, str, int]]:
        """
        Extract a 1-minute EHG window from a WFDB record.

        Args:
            record: Loaded wfdb.Record object.
            channel_name: Channel identifier (e.g., 'S1 0.08-4 Hz' or '1_DOCFILT-4-0.08-4').
            start_time: Start time of window in seconds (default: 180.0s).
            window_duration: Duration of window in seconds (default: 60.0s).

        Returns:
            Dictionary containing extracted signal array, sampling frequency,
            start_time, end_time, channel_name, channel_index, and window metadata.
        """
        fs = float(record.fs)
        ch_idx, canonical_ch_name = self.resolve_channel_index(record, channel_name)

        end_time = start_time + window_duration

        # Calculate sample indices programmatically
        start_sample = int(round(start_time * fs))
        end_sample = int(round(end_time * fs))

        total_samples = record.sig_len
        total_duration = total_samples / fs

        if start_sample < 0 or end_sample > total_samples:
            raise ValueError(
                f"Window [{start_time}s, {end_time}s] ({start_sample} to {end_sample} samples) "
                f"exceeds signal duration of {total_duration:.2f}s ({total_samples} samples)."
            )

        extracted_signal = record.p_signal[start_sample:end_sample, ch_idx]

        return {
            "signal": extracted_signal,
            "sampling_frequency": fs,
            "start_time": start_time,
            "end_time": end_time,
            "start_sample": start_sample,
            "end_sample": end_sample,
            "channel_name": canonical_ch_name,
            "channel_index": ch_idx,
            "total_samples": total_samples,
            "total_duration": total_duration,
            "extracted_samples": len(extracted_signal),
            "extracted_duration": len(extracted_signal) / fs
        }

    # Alias for general multi-window extraction
    extract_window = extract_one_minute


def extract_one_minute(
    record: wfdb.Record,
    channel_name: Union[str, int] = "S1 0.08-4 Hz",
    start_time: float = 180.0,
    window_duration: float = 60.0
) -> Tuple[np.ndarray, float, float, float, str]:
    """
    Convenience function returning (signal, sampling_frequency, start_time, end_time, channel_name).
    """
    extractor = EHGWindowExtractor()
    res = extractor.extract_one_minute(
        record=record,
        channel_name=channel_name,
        start_time=start_time,
        window_duration=window_duration
    )
    return (
        res["signal"],
        res["sampling_frequency"],
        res["start_time"],
        res["end_time"],
        res["channel_name"]
    )


def extract_window(
    record: wfdb.Record,
    channel_name: Union[str, int] = "S1 0.08-4 Hz",
    start_time: float = 180.0,
    window_duration: float = 60.0
) -> Dict[str, Union[np.ndarray, float, str, int]]:
    """
    Extract a specified time window from an EHG signal record.
    """
    extractor = EHGWindowExtractor()
    return extractor.extract_window(
        record=record,
        channel_name=channel_name,
        start_time=start_time,
        window_duration=window_duration
    )
