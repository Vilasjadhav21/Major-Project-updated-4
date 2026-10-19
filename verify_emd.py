"""
Numerical validation and artifact generation script for EMD decomposition on single record tpehg1007.

Objective 4: EHG Rapid Triage
Single-record validation only (tpehg1007).
"""

import os
import sys

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PyEMD
from PyEMD import EMD

from src.emd_processing import perform_emd


def main():
    # File paths
    input_csv_path = "outputs/features/tpehg1007_S1_1minute.csv"
    imfs_csv_path = "outputs/imfs/tpehg1007_S1_imfs.csv"
    imf1_csv_path = "outputs/imfs/tpehg1007_S1_IMF1.csv"
    
    plot_emd_path = "outputs/plots/tpehg1007_S1_EMD.png"
    plot_imf1_path = "outputs/plots/tpehg1007_S1_IMF1.png"
    plot_recon_path = "outputs/plots/tpehg1007_S1_EMD_reconstruction.png"

    # Ensure output directories exist
    os.makedirs("outputs/imfs", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)

    # 1. Load input CSV
    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"Input file not found: {input_csv_path}")

    df_in = pd.read_csv(input_csv_path)
    time_seconds = df_in["time_seconds"].values
    ehg_signal = df_in["ehg_amplitude"].values

    # Experiment Metadata
    record_id = "tpehg1007"
    clinical_label = "Preterm"
    channel = "S1 0.08–4 Hz"
    sampling_freq = 20.0
    input_samples = len(ehg_signal)
    input_duration = time_seconds[-1] - time_seconds[0] + (1.0 / sampling_freq)

    print("==================================================")
    print("EMD DECOMPOSITION & VALIDATION (SINGLE-RECORD)")
    print("==================================================")
    print(f"Record: {record_id}")
    print(f"Clinical label: {clinical_label}")
    print(f"Channel: {channel}")
    print(f"Sampling frequency: {sampling_freq} Hz")
    print(f"Input samples: {input_samples}")
    print(f"Input duration: {input_duration:.1f} seconds (Time window: {time_seconds[0]:.1f} - {time_seconds[-1]:.1f} s)")

    # 2. Perform EMD
    imfs = perform_emd(ehg_signal, sampling_frequency=sampling_freq)
    n_components = len(imfs)
    print(f"Number of IMFs/components generated: {n_components}")

    # 3. Save all IMFs to CSV
    # Dynamically build column names: IMF_1, IMF_2, ..., Residue (or IMF_N)
    columns = ["time_seconds"]
    for i in range(n_components):
        if i == n_components - 1 and n_components > 1:
            columns.append(f"IMF_{i+1}_Residue")
        else:
            columns.append(f"IMF_{i+1}")

    imf_dict = {"time_seconds": time_seconds}
    for i in range(n_components):
        imf_dict[columns[i+1]] = imfs[i]

    df_imfs = pd.DataFrame(imf_dict)
    df_imfs.to_csv(imfs_csv_path, index=False)
    print(f"Saved all IMFs to: {imfs_csv_path} (Rows: {len(df_imfs)}, Columns: {list(df_imfs.columns)})")

    # 4. Save IMF-1 to separate CSV
    imf_1 = imfs[0]
    df_imf1 = pd.DataFrame({
        "time_seconds": time_seconds,
        "IMF_1": imf_1
    })
    df_imf1.to_csv(imf1_csv_path, index=False)
    print(f"Saved IMF-1 to: {imf1_csv_path} (Rows: {len(df_imf1)})")

    # 5. Numerical Validation Statistics
    print("\n--------------------------------------------------")
    print("NUMERICAL VALIDATION - ALL COMPONENTS")
    print("--------------------------------------------------")
    print(f"{'Component':<15} {'Minimum':<12} {'Maximum':<12} {'Mean':<12} {'Std Dev':<12} {'Energy':<14}")
    print("-" * 77)

    for i in range(n_components):
        comp_name = columns[i+1]
        comp_data = imfs[i]
        c_min = np.min(comp_data)
        c_max = np.max(comp_data)
        c_mean = np.mean(comp_data)
        c_std = np.std(comp_data)
        c_energy = np.sum(comp_data ** 2)
        print(f"{comp_name:<15} {c_min:<12.6f} {c_max:<12.6f} {c_mean:<12.6f} {c_std:<12.6f} {c_energy:<14.6f}")

    print("\nIMF-1 SEPARATE STATISTICS:")
    print(f"  Mean:               {np.mean(imf_1):.8f}")
    print(f"  Standard Deviation: {np.std(imf_1):.8f}")
    print(f"  Minimum:            {np.min(imf_1):.8f}")
    print(f"  Maximum:            {np.max(imf_1):.8f}")
    print(f"  Energy:             {np.sum(imf_1 ** 2):.8f}")

    # 6. Reconstruction Check
    reconstructed_signal = np.sum(imfs, axis=0)
    recon_error = ehg_signal - reconstructed_signal

    mae = np.mean(np.abs(recon_error))
    rmse = np.sqrt(np.mean(recon_error ** 2))
    max_abs_error = np.max(np.abs(recon_error))

    print("\n--------------------------------------------------")
    print("RECONSTRUCTION CHECK")
    print("--------------------------------------------------")
    print(f"Reconstruction Formula: Sum of all {n_components} components (IMFs + Residue)")
    print(f"MAE:                   {mae:.12e}")
    print(f"RMSE:                  {rmse:.12e}")
    print(f"Maximum Absolute Error:{max_abs_error:.12e}")

    # 7. Create EMD Plots (outputs/plots/tpehg1007_S1_EMD.png)
    n_subplots = n_components + 1  # 1 original + n_components
    fig_height = max(10, 1.8 * n_subplots)
    fig, axes = plt.subplots(n_subplots, 1, figsize=(14, fig_height), sharex=True)

    # Plot Original Signal
    axes[0].plot(time_seconds, ehg_signal, color="#1f77b4", lw=1.2)
    axes[0].set_ylabel("Original", fontsize=10, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].set_title(f"TPEHGDB {record_id} — {channel} — EMD Decomposition", fontsize=14, fontweight="bold", pad=12)

    # Plot each IMF / component
    colors = ["#2ca02c", "#ff7f0e", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    for i in range(n_components):
        ax = axes[i+1]
        c_label = columns[i+1]
        c_color = colors[i % len(colors)]
        ax.plot(time_seconds, imfs[i], color=c_color, lw=1.2)
        ax.set_ylabel(c_label, fontsize=10, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)

    axes[-1].set_xlabel("Time (seconds)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(plot_emd_path, dpi=300)
    plt.close()
    print(f"\nSaved EMD Plot to: {plot_emd_path}")

    # 8. Create IMF-1 Detailed Plot (outputs/plots/tpehg1007_S1_IMF1.png)
    plt.figure(figsize=(12, 5))
    plt.plot(time_seconds, imf_1, color="#2ca02c", lw=1.5)
    plt.title(f"TPEHGDB {record_id} — {channel} — IMF-1", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Time (seconds)", fontsize=11, fontweight="bold")
    plt.ylabel("IMF-1 amplitude", fontsize=11, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(plot_imf1_path, dpi=300)
    plt.close()
    print(f"Saved IMF-1 Plot to: {plot_imf1_path}")

    # 9. Create Reconstruction Plot (outputs/plots/tpehg1007_S1_EMD_reconstruction.png)
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    axes[0].plot(time_seconds, ehg_signal, color="#1f77b4", lw=1.5, label="Original Signal")
    axes[0].set_ylabel("Amplitude", fontsize=10, fontweight="bold")
    axes[0].set_title(f"TPEHGDB {record_id} — {channel} — EMD Reconstruction Check", fontsize=14, fontweight="bold", pad=12)
    axes[0].legend(loc="upper right")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].plot(time_seconds, reconstructed_signal, color="#2ca02c", lw=1.5, linestyle="-", label="Reconstructed Signal")
    axes[1].set_ylabel("Amplitude", fontsize=10, fontweight="bold")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    axes[2].plot(time_seconds, recon_error, color="#d62728", lw=1.2, label=f"Reconstruction Error (Max Abs: {max_abs_error:.2e})")
    axes[2].set_xlabel("Time (seconds)", fontsize=11, fontweight="bold")
    axes[2].set_ylabel("Error", fontsize=10, fontweight="bold")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(plot_recon_path, dpi=300)
    plt.close()
    print(f"Saved Reconstruction Plot to: {plot_recon_path}")

    # 10. Reproducibility & Configuration Summary
    pyemd_version = getattr(PyEMD, "__version__", "1.10.0")
    print("\n--------------------------------------------------")
    print("REPRODUCIBILITY & CONFIGURATION")
    print("--------------------------------------------------")
    print(f"PyEMD Version:         {pyemd_version}")
    print("EMD Configuration:      Default PyEMD standard EMD settings")
    print("Sifting Stopping Rule: Standard PyEMD SD (Standard Deviation) / Cauchy criterion")

    print("\n==================================================")
    print("FINAL SUMMARY REPORT")
    print("==================================================")
    print(f"Record:            {record_id}")
    print(f"Clinical label:    {clinical_label}")
    print(f"Channel:           {channel}")
    print(f"Sampling frequency:{sampling_freq} Hz")
    print(f"Input samples:     {input_samples}")
    print(f"Input duration:    {input_duration:.1f} s")
    print(f"Number of IMFs:    {n_components}")
    print("\nFor IMF-1:")
    print(f"  Mean:               {np.mean(imf_1):.8f}")
    print(f"  Standard deviation: {np.std(imf_1):.8f}")
    print(f"  Minimum:            {np.min(imf_1):.8f}")
    print(f"  Maximum:            {np.max(imf_1):.8f}")
    print(f"  Energy:             {np.sum(imf_1 ** 2):.8f}")
    print("\nReconstruction:")
    print(f"  MAE:                {mae:.12e}")
    print(f"  RMSE:               {rmse:.12e}")
    print(f"  Maximum abs error:  {max_abs_error:.12e}")
    print("\nOutput files created:")
    print(f"  1. {imfs_csv_path}")
    print(f"  2. {imf1_csv_path}")
    print(f"  3. {plot_emd_path}")
    print(f"  4. {plot_imf1_path}")
    print(f"  5. {plot_recon_path}")
    print("==================================================")


if __name__ == "__main__":
    main()
