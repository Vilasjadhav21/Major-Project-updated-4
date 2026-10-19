"""
Verification script for Objective 4: 1-Minute EHG Signal Window Extraction.

Loads record tpehg1007, inspects channel header mapping, extracts deterministic 60-second window
from S1 0.08-4 Hz (180s to 240s), verifies statistical integrity, saves diagnostic plots, and
exports numerical signal CSV to outputs/features/tpehg1007_S1_1minute.csv.
"""

import sys
from pathlib import Path

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_loader import TPEHGDataLoader, load_record
from src.metadata import get_label
from src.window_extraction import EHGWindowExtractor


def verify_window_extraction():
    """Execute complete 1-minute EHG window extraction, verification, and plot generation."""
    data_dir = PROJECT_ROOT / "data" / "tpehgdb"
    plots_dir = PROJECT_ROOT / "outputs" / "plots"
    features_dir = PROJECT_ROOT / "outputs" / "features"
    plots_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data loader & select first record
    data_loader = TPEHGDataLoader(data_dir=data_dir)
    first_record_entry = data_loader.get_first_record_name()

    # 2. Get label metadata
    label_info = get_label(first_record_entry)
    record_label = label_info["label"]

    # 3. Load complete WFDB record
    record = load_record(first_record_entry)
    record_name = record.record_name
    fs = float(record.fs)
    total_samples = record.sig_len
    total_duration = total_samples / fs

    # 5, 7. Print complete channel mapping table
    print("=" * 75)
    print(f"             TPEHGDB RECORD '{record_name}' CHANNEL MAPPING             ")
    print("=" * 75)
    print(f"{'Index':<7} | {'Channel Name':<28} | {'Sampling Freq':<14} | {'Units':<8}")
    print("-" * 75)
    for idx, (name, unit) in enumerate(zip(record.sig_name, record.units)):
        print(f"{idx:<7} | {name:<28} | {fs:<14.1f} | {unit:<8}")
    print("=" * 75)

    # 8, 9, 10, 11, 12, 13. Perform 1-minute window extraction
    extractor = EHGWindowExtractor()
    target_channel_spec = "S1 0.08-4 Hz"
    start_time = 180.0
    window_duration = 60.0

    res = extractor.extract_one_minute(
        record=record,
        channel_name=target_channel_spec,
        start_time=start_time,
        window_duration=window_duration
    )

    signal = res["signal"]
    ch_index = res["channel_index"]
    ch_name = res["channel_name"]
    end_time = res["end_time"]
    start_sample = res["start_sample"]
    end_sample = res["end_sample"]
    extracted_samples = res["extracted_samples"]
    extracted_duration = res["extracted_duration"]

    # Valid filtered region boundary rule (ignore first and last 180s)
    trim_boundary_sec = 180.0
    valid_region_start = trim_boundary_sec
    valid_region_end = max(0.0, total_duration - trim_boundary_sec)
    valid_region_str = f"[{valid_region_start:.1f}s to {valid_region_end:.1f}s]"

    # 14. Compute numerical verification statistics
    mean_val = float(np.mean(signal))
    std_val = float(np.std(signal))
    min_val = float(np.min(signal))
    max_val = float(np.max(signal))
    nan_count = int(np.isnan(signal).sum())
    inf_count = int(np.isinf(signal).sum())

    # 15. Verify exactly 1200 samples for 20 Hz, 60-second window
    expected_samples = 1200
    assert extracted_samples == expected_samples, f"Expected {expected_samples} samples, got {extracted_samples}."

    # 30. Print complete summary with exact required key titles
    print("\n" + "=" * 75)
    print("        1-MINUTE EHG SIGNAL EXTRACTION VERIFICATION REPORT           ")
    print("=" * 75)
    print(f"Record:                 {record_name}")
    print(f"Label:                  {record_label}")
    print(f"Channel selected:       {ch_name}")
    print(f"Channel index:          {ch_index}")
    print(f"Sampling frequency:     {fs} Hz")
    print(f"Complete signal length: {total_samples} samples")
    print(f"Complete duration:      {total_duration:.2f} seconds")
    print(f"Valid filtered region:  {valid_region_str}")
    print(f"Window start:           {start_time:.1f} seconds (sample {start_sample})")
    print(f"Window end:             {end_time:.1f} seconds (sample {end_sample})")
    print(f"Extracted samples:      {extracted_samples}")
    print(f"Extracted duration:     {extracted_duration:.1f} seconds")
    print(f"Mean:                   {mean_val:.8f}")
    print(f"Standard deviation:     {std_val:.8f}")
    print(f"Minimum:                {min_val:.8f}")
    print(f"Maximum:                {max_val:.8f}")
    print(f"NaN count:              {nan_count}")
    print(f"Infinite count:         {inf_count}")
    print("=" * 75)

    # 17. Diagnostic Plot 1: 1-Minute Extracted Window
    plot1_path = plots_dir / f"{record_name}_S1_1minute.png"
    time_window_sec = np.arange(start_sample, end_sample) / fs

    plt.figure(figsize=(12, 4.5))
    plt.plot(time_window_sec, signal, color='#1f77b4', linewidth=1.0)
    plt.title(f"TPEHGDB {record_name} — S1 0.08–4 Hz — 1 Minute Window", fontsize=12, fontweight='bold')
    plt.xlabel("Time (seconds)", fontsize=10)
    plt.ylabel("EHG amplitude (mV)", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    print(f"\n[SUCCESS] Extracted 1-minute plot saved to: {plot1_path.resolve()}")

    # 18. Diagnostic Plot 2: Window Location in Complete Recording
    plot2_path = plots_dir / f"{record_name}_S1_window_location.png"
    full_s1_signal = record.p_signal[:, ch_index]
    time_full_sec = np.arange(total_samples) / fs

    plt.figure(figsize=(14, 5))
    plt.plot(time_full_sec, full_s1_signal, color='#a0a0a0', linewidth=0.5, label='Complete S1 0.08–4 Hz Recording')

    # Highlight 180s boundary lines
    plt.axvline(180.0, color='orange', linestyle='--', linewidth=1.2, label='180s Filter Boundary')
    plt.axvline(valid_region_end, color='orange', linestyle='--', linewidth=1.2)

    # Highlight extracted 1-minute window
    plt.axvspan(start_time, end_time, color='red', alpha=0.35, label='Selected 1-Min Window (180s - 240s)')
    plt.plot(time_window_sec, signal, color='red', linewidth=1.0)

    plt.title(f"TPEHGDB {record_name} — Complete S1 0.08–4 Hz Recording & 1-Minute Window Location", fontsize=12, fontweight='bold')
    plt.xlabel("Time (seconds)", fontsize=10)
    plt.ylabel("EHG amplitude (mV)", fontsize=10)
    plt.legend(loc='upper right', fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] Window location plot saved to: {plot2_path.resolve()}")

    # 19. Export extracted numerical signal to CSV
    csv_path = features_dir / f"{record_name}_S1_1minute.csv"
    df_window = pd.DataFrame({
        "time_seconds": time_window_sec,
        "ehg_amplitude": signal
    })
    df_window.to_csv(csv_path, index=False)
    print(f"[SUCCESS] Extracted numerical signal CSV saved to: {csv_path.resolve()}")


if __name__ == "__main__":
    verify_window_extraction()
