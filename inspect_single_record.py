"""
Script to load and inspect ONE real record from PhysioNet TPEHGDB v1.0.1.

Validates record loading via WFDB, prints header metadata and signal information,
and saves a diagnostic raw multi-channel signal plot to outputs/plots/first_record_channels.png.
"""

import sys
from pathlib import Path

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import wfdb

from src.data_loader import TPEHGDataLoader, load_record


def inspect_single_record():
    """Load first record from RECORDS, print header metadata, and plot raw signals."""
    data_loader = TPEHGDataLoader(data_dir="data/tpehgdb")

    # 1. Read record names from data/tpehgdb/RECORDS
    records_list = data_loader.read_records_list()
    print(f"Total records found in RECORDS file: {len(records_list)}")

    # 2. Select the FIRST record listed in RECORDS
    first_record_entry = data_loader.get_first_record_name()
    print(f"Selected first record entry from RECORDS: '{first_record_entry}'")

    # 3. Load record using wfdb.rdrecord via load_record()
    record = load_record(first_record_entry)

    # Extract header metadata & verification
    record_name = record.record_name
    fs = record.fs  # Sampling frequency verified from header
    num_samples = record.sig_len  # Number of samples verified from header
    num_channels = record.n_sig  # Number of channels verified from header
    sig_names = record.sig_name  # Signal names verified from header
    sig_units = record.units if record.units is not None else ['mV'] * num_channels
    signal_shape = record.p_signal.shape
    first_5_samples = record.p_signal[:5, :]
    header_comments = record.comments

    # 6, 7, 8. Print complete metadata verification
    print("=" * 75)
    print("                TPEHGDB SINGLE RECORD INSPECTION REPORT              ")
    print("=" * 75)
    print(f"Record Name:             {record_name}")
    print(f"Sampling Frequency (fs): {fs} Hz (Verified from WFDB header)")
    print(f"Number of Samples:       {num_samples}")
    print(f"Number of Channels:      {num_channels} (Verified from WFDB header)")
    print(f"Signal Names:            {sig_names}")
    print(f"Signal Units:            {sig_units}")
    print(f"Signal Array Shape:      {signal_shape}")
    print("-" * 75)
    print("First 5 Samples of Each Channel:")
    for ch_idx, (name, unit) in enumerate(zip(sig_names, sig_units)):
        samples_str = ", ".join([f"{val:.6f}" for val in first_5_samples[:, ch_idx]])
        print(f"  Channel {ch_idx+1:02d} [{name} ({unit})]: [{samples_str}]")
    print("-" * 75)
    print("Header Comments:")
    for comment in header_comments:
        print(f"  - {comment}")
    print("=" * 75)

    # 9. Save ONE diagnostic plot of the loaded channels
    plots_dir = PROJECT_ROOT / "outputs" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_filepath = plots_dir / "first_record_channels.png"

    time_sec = np.arange(num_samples) / fs

    fig, axes = plt.subplots(num_channels, 1, figsize=(14, 2.2 * num_channels), sharex=True)
    if num_channels == 1:
        axes = [axes]

    fig.suptitle(
        f"PhysioNet TPEHGDB Record: {record_name} (Raw Unprocessed Signals)",
        fontsize=13,
        fontweight='bold'
    )

    for i in range(num_channels):
        ax = axes[i]
        unit_str = sig_units[i] if i < len(sig_units) else "mV"
        ch_name = sig_names[i] if i < len(sig_names) else f"Ch {i+1}"

        ax.plot(time_sec, record.p_signal[:, i], linewidth=0.6, color='#1f77b4')
        ax.set_ylabel(f"{ch_name}\n({unit_str})", fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.tick_params(labelsize=8)

    axes[-1].set_xlabel("Time (seconds)", fontsize=10)
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n[SUCCESS] Diagnostic plot saved to: {plot_filepath.resolve()}")


if __name__ == "__main__":
    inspect_single_record()
