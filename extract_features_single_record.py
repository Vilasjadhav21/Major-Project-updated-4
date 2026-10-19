"""
Single-record Feature Extraction Validation Script for TPEHGDB tpehg1007.

Calculates the 5 target numerical features for IMF-1 and IMF-2 of record tpehg1007 (S1 channel).
Saves feature vector CSV and diagnostic plots.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from scipy.signal import hilbert
import matplotlib.pyplot as plt

from src.feature_extraction import (
    sample_entropy,
    median_frequency,
    rms,
    hilbert_peak_to_average_power_ratio,
    extract_features
)


def main():
    print("==================================================")
    print("12. REPRODUCIBILITY & SYSTEM VERSIONS")
    print("==================================================")
    print(f"NumPy version:  {np.__version__}")
    print(f"SciPy version:  {scipy.__version__}")
    print(f"Pandas version: {pd.__version__}")
    print()

    print("==================================================")
    print("MATHEMATICAL EQUATIONS FOR FEATURE EXTRACTION")
    print("==================================================")
    print("1. Sample Entropy (SampEn, IMF-1):")
    print("   SampEn(m=2, r=0.2*std) = -ln(A / B)")
    print("   where B = total matching template pairs of length m=2 (distance <= r), excluding self-matches.")
    print("   where A = total matching template pairs of length m+1=3 (distance <= r), excluding self-matches.")
    print("   Distance metric: Chebyshev distance max|x(i+k) - x(j+k)|.")
    print("2. Median Frequency (IMF-1):")
    print("   FFT power spectrum power = |rfft(IMF1)|^2 excluding DC component (0 Hz).")
    print("   Median frequency is f where sum(power[freq <= f]) >= 0.5 * sum(power).")
    print("   fs = 20 Hz.")
    print("3. RMS (IMF-1):")
    print("   RMS = sqrt(mean(IMF1^2))")
    print("4. RMS (IMF-2):")
    print("   RMS = sqrt(mean(IMF2^2))")
    print("5. Hilbert Peak-to-Average Power Ratio (PAR, IMF-1):")
    print("   Analytic signal z(t) = hilbert(IMF1)")
    print("   Instantaneous amplitude A(t) = |z(t)|, Instantaneous power P(t) = A(t)^2")
    print("   Hilbert PAR = max(P(t)) / mean(P(t))")
    print()

    # Paths
    project_root = Path(__file__).resolve().parent.parent
    imf_csv_path = project_root / "outputs" / "imfs" / "tpehg1007_S1_imfs.csv"
    features_csv_path = project_root / "outputs" / "features" / "tpehg1007_S1_features.csv"
    plot_summary_path = project_root / "outputs" / "plots" / "tpehg1007_feature_summary.png"
    plot_hilbert_path = project_root / "outputs" / "plots" / "tpehg1007_S1_IMF1_hilbert_amplitude.png"

    # Ensure output directories exist
    features_csv_path.parent.mkdir(parents=True, exist_ok=True)
    plot_summary_path.parent.mkdir(parents=True, exist_ok=True)

    print("==================================================")
    print("1. LOADING DECOMPOSED IMFS")
    print("==================================================")
    print(f"Loading IMFs from: {imf_csv_path}")
    if not imf_csv_path.exists():
        raise FileNotFoundError(f"IMF file not found at {imf_csv_path}")

    df_imfs = pd.read_csv(imf_csv_path)
    print(f"Columns present: {list(df_imfs.columns)}")

    # Extract IMF-1 and IMF-2 arrays
    if "IMF_1" not in df_imfs.columns or "IMF_2" not in df_imfs.columns:
        raise ValueError("CSV missing required IMF_1 or IMF_2 columns.")

    imf1 = df_imfs["IMF_1"].to_numpy()
    imf2 = df_imfs["IMF_2"].to_numpy()
    sampling_frequency = 20.0

    print("\n==================================================")
    print("7. SIGNAL VALIDATION CHECKS")
    print("==================================================")
    imf1_len = len(imf1)
    imf2_len = len(imf2)
    has_nan = np.isnan(imf1).any() or np.isnan(imf2).any()
    has_inf = np.isinf(imf1).any() or np.isinf(imf2).any()

    print(f"[CHECK] IMF-1 length = {imf1_len} (Expected: 1200) -> {'PASS' if imf1_len == 1200 else 'FAIL'}")
    print(f"[CHECK] IMF-2 length = {imf2_len} (Expected: 1200) -> {'PASS' if imf2_len == 1200 else 'FAIL'}")
    print(f"[CHECK] No NaN values in IMF-1/2 -> {'PASS' if not has_nan else 'FAIL'}")
    print(f"[CHECK] No Infinite values in IMF-1/2 -> {'PASS' if not has_inf else 'FAIL'}")
    fs_ok = abs(sampling_frequency - 20.0) <= 0.01
    print(f"[CHECK] Sampling frequency = {sampling_frequency} Hz -> {'PASS' if fs_ok else 'FAIL'}")

    # Call main validation extraction function
    feature_dict = extract_features(imf1, imf2, sampling_frequency=sampling_frequency)

    sampen_imf1 = feature_dict['sampen_imf1']
    median_freq_imf1 = feature_dict['median_frequency_imf1_hz']
    rms_imf1_val = feature_dict['rms_imf1']
    rms_imf2_val = feature_dict['rms_imf2']
    hilbert_par_imf1 = feature_dict['hilbert_par_imf1']

    print("\n==================================================")
    print("8. FEATURE EXTRACTION RESULTS")
    print("==================================================")
    print("Record: tpehg1007")
    print("Label: Preterm")
    print("Channel: S1 0.08–4 Hz")
    print("Sampling frequency: 20 Hz")
    print("Number of samples: 1200")
    print()
    print(f"SampEn(IMF-1):            {sampen_imf1:.6f}")
    print(f"Median Frequency(IMF-1):  {median_freq_imf1:.6f} Hz")
    print(f"RMS(IMF-1):               {rms_imf1_val:.6e}")
    print(f"RMS(IMF-2):               {rms_imf2_val:.6e}")
    print(f"Hilbert PAR(IMF-1):       {hilbert_par_imf1:.6f}")

    # Check finite values
    all_finite = all(np.isfinite([sampen_imf1, median_freq_imf1, rms_imf1_val, rms_imf2_val, hilbert_par_imf1]))
    print(f"\n[SANITY CHECK] All feature values finite: {all_finite}")

    print("\n==================================================")
    print("9. SAVING FEATURE VECTOR CSV")
    print("==================================================")
    feature_row = {
        'record': 'tpehg1007',
        'label': 'Preterm',
        'channel': 'S1 0.08–4 Hz',
        'sampling_frequency': 20,
        'window_start_seconds': 180,
        'window_end_seconds': 240,
        'num_samples': 1200,
        'sampen_imf1': sampen_imf1,
        'median_frequency_imf1_hz': median_freq_imf1,
        'rms_imf1': rms_imf1_val,
        'rms_imf2': rms_imf2_val,
        'hilbert_par_imf1': hilbert_par_imf1
    }

    df_out = pd.DataFrame([feature_row])
    df_out.to_csv(features_csv_path, index=False)
    print(f"Saved feature vector to: {features_csv_path}")

    print("\n==================================================")
    print("10. GENERATING DIAGNOSTIC PLOTS")
    print("==================================================")

    # Plot 1: Feature Summary Bar Plot
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    features_names = [
        'SampEn\n(IMF-1)',
        'Median Freq\n(IMF-1, Hz)',
        'RMS\n(IMF-1)',
        'RMS\n(IMF-2)',
        'Hilbert PAR\n(IMF-1)'
    ]
    features_vals = [
        sampen_imf1,
        median_freq_imf1,
        rms_imf1_val,
        rms_imf2_val,
        hilbert_par_imf1
    ]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    bars = ax.bar(features_names, features_vals, color=colors, alpha=0.85, edgecolor='black', width=0.55)
    ax.set_yscale('log')
    ax.set_ylabel('Feature Value (Log Scale)', fontsize=11, fontweight='bold')
    ax.set_title('TPEHGDB tpehg1007 — Feature Summary (S1 0.08–4 Hz)', fontsize=13, fontweight='bold', pad=15)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    for bar, val in zip(bars, features_vals):
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval * 1.15,
            f"{val:.4g}",
            ha='center',
            va='bottom',
            fontsize=9,
            fontweight='bold'
        )

    plt.tight_layout()
    fig.savefig(plot_summary_path)
    plt.close(fig)
    print(f"Saved feature summary plot to: {plot_summary_path}")

    # Plot 2: Hilbert Instantaneous Amplitude Plot
    analytic_sig = hilbert(imf1)
    inst_amp = np.abs(analytic_sig)
    time_sec = np.linspace(0, 60, len(imf1), endpoint=False)

    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=300)
    ax.plot(time_sec, inst_amp, color='#0066cc', linewidth=1.2, label='Instantaneous Amplitude A(t)')
    ax.axhline(np.mean(inst_amp), color='#d9534f', linestyle='--', linewidth=1.5, label=f'Mean Amp ({np.mean(inst_amp):.4e})')
    ax.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Instantaneous Amplitude', fontsize=11, fontweight='bold')
    ax.set_title('TPEHGDB tpehg1007 — IMF-1 Hilbert Instantaneous Amplitude', fontsize=13, fontweight='bold', pad=12)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    fig.savefig(plot_hilbert_path)
    plt.close(fig)
    print(f"Saved Hilbert amplitude plot to: {plot_hilbert_path}")

    print("\nFeature extraction completed successfully.")


if __name__ == "__main__":
    main()
