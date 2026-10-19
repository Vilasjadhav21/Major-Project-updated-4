"""
Validation script for Objective 4: Processing the first 10 TPEHGDB records.

Loads S1 0.08–4 Hz signal using WFDB for the first 10 records in data/tpehgdb/RECORDS,
extracts the 60-second window (180–240s), verifies sampling rate and sample count (1200),
decomposes the window using EMD, computes 5 target features (SampEn IMF-1, Median Freq IMF-1,
RMS IMF-1, RMS IMF-2, Hilbert PAR IMF-1), attaches clinical metadata labels (Term/Preterm),
and outputs results to outputs/features/validation_10_records.csv along with a summary report.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.data_loader import TPEHGDataLoader, load_record
from src.metadata import TPEHGMetadataParser, get_label
from src.window_extraction import EHGWindowExtractor
from src.emd_processing import perform_emd
from src.feature_extraction import (
    sample_entropy,
    median_frequency,
    rms,
    hilbert_peak_to_average_power_ratio
)

# Setup logger for pipeline logging
logger = logging.getLogger("validate_10_records")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def validate_10_records(
    data_dir: Path = PROJECT_ROOT / "data" / "tpehgdb",
    output_csv_path: Path = PROJECT_ROOT / "outputs" / "features" / "validation_10_records.csv",
    num_records_to_process: int = 10
) -> Dict[str, Any]:
    """
    Process the first `num_records_to_process` records from RECORDS list.

    Returns:
        Dictionary containing DataFrame of results, summary stats, success/failure lists.
    """
    data_loader = TPEHGDataLoader(data_dir=data_dir)
    window_extractor = EHGWindowExtractor()

    # Read RECORDS file and take first 10
    all_records = data_loader.read_records_list()
    target_records = all_records[:num_records_to_process]

    logger.info(f"Targeting first {len(target_records)} records from {data_dir / 'RECORDS'}:")
    for r in target_records:
        logger.info(f"  - {r}")

    successful_rows: List[Dict[str, Any]] = []
    successful_records: List[str] = []
    failed_records: List[Dict[str, str]] = []

    target_channel = "S1 0.08-4 Hz"
    window_start = 180.0
    window_end = 240.0
    expected_duration = window_end - window_start  # 60 seconds

    for idx, rec_entry in enumerate(target_records, 1):
        rec_stem = Path(rec_entry).name
        logger.info(f"\n[{idx}/{len(target_records)}] Processing record '{rec_entry}' (stem: '{rec_stem}')...")

        try:
            # 1. Retrieve label metadata
            try:
                meta_info = get_label(rec_entry, smr_filepath=data_dir / "tpehgdb.smr")
                label_val = meta_info.get("label", "Unknown")
            except Exception as e_meta:
                raise ValueError(f"Failed to fetch metadata label for record '{rec_entry}': {e_meta}")

            # 2. Load WFDB record
            try:
                wfdb_rec = load_record(rec_entry, data_dir=data_dir)
            except Exception as e_load:
                raise RuntimeError(f"WFDB load failed for record '{rec_entry}': {e_load}")

            actual_fs = float(wfdb_rec.fs)
            logger.info(f"  Record loaded. Header fs = {actual_fs} Hz")

            # 3. Extract 60-second window (180s - 240s)
            win_dict = window_extractor.extract_one_minute(
                record=wfdb_rec,
                channel_name=target_channel,
                start_time=window_start,
                window_duration=expected_duration
            )

            signal_win = win_dict["signal"]
            extracted_samples = win_dict["extracted_samples"]
            expected_samples = int(round(expected_duration * actual_fs))

            # 4. Verify sample count
            if extracted_samples != expected_samples:
                raise ValueError(
                    f"Sample count mismatch: Expected exactly {expected_samples} samples for fs={actual_fs} Hz, "
                    f"got {extracted_samples} samples."
                )

            if extracted_samples != 1200:
                raise ValueError(f"Extracted window sample count is {extracted_samples}, expected exactly 1200.")

            logger.info(f"  Window extracted successfully: [{window_start}s to {window_end}s], {extracted_samples} samples.")

            # 5. Perform EMD decomposition
            imfs = perform_emd(signal_win, sampling_frequency=actual_fs)
            if len(imfs) < 2:
                raise ValueError(f"EMD produced only {len(imfs)} IMFs, expected at least 2.")

            imf1 = imfs[0]
            imf2 = imfs[1]

            # 6. Calculate exactly the five target features
            sampen_imf1_val = sample_entropy(imf1, m=2, r_ratio=0.2)
            med_freq_imf1_val = median_frequency(imf1, sampling_frequency=actual_fs)
            rms_imf1_val = rms(imf1)
            rms_imf2_val = rms(imf2)
            hilbert_par_imf1_val = hilbert_peak_to_average_power_ratio(imf1)

            feature_values = [
                sampen_imf1_val,
                med_freq_imf1_val,
                rms_imf1_val,
                rms_imf2_val,
                hilbert_par_imf1_val
            ]

            # 7. Validate finite feature values
            nan_check = [np.isnan(v) for v in feature_values]
            inf_check = [np.isinf(v) for v in feature_values]

            if any(nan_check) or any(inf_check):
                raise ValueError(
                    f"Non-finite feature value detected in record '{rec_stem}': "
                    f"SampEn={sampen_imf1_val}, MedFreq={med_freq_imf1_val}, "
                    f"RMS1={rms_imf1_val}, RMS2={rms_imf2_val}, HilbertPAR={hilbert_par_imf1_val}"
                )

            # Store result row
            row = {
                "record": rec_stem,
                "sampling_frequency": actual_fs,
                "window_start_sec": window_start,
                "window_end_sec": window_end,
                "n_samples": extracted_samples,
                "sampen_imf1": sampen_imf1_val,
                "median_freq_imf1_hz": med_freq_imf1_val,
                "rms_imf1": rms_imf1_val,
                "rms_imf2": rms_imf2_val,
                "hilbert_par_imf1": hilbert_par_imf1_val,
                "label": label_val
            }

            successful_rows.append(row)
            successful_records.append(rec_stem)
            logger.info(f"  [SUCCESS] {rec_stem} | Label={label_val} | SampEn={sampen_imf1_val:.4f} | MedFreq={med_freq_imf1_val:.4f} Hz | RMS1={rms_imf1_val:.4e} | RMS2={rms_imf2_val:.4e} | H-PAR={hilbert_par_imf1_val:.4f}")

        except Exception as err:
            logger.error(f"  [FAILED] Record '{rec_entry}': {err}")
            failed_records.append({
                "record": rec_stem,
                "error": str(err)
            })

    # Prepare DataFrame with required columns
    required_columns = [
        "record",
        "sampling_frequency",
        "window_start_sec",
        "window_end_sec",
        "n_samples",
        "sampen_imf1",
        "median_freq_imf1_hz",
        "rms_imf1",
        "rms_imf2",
        "hilbert_par_imf1",
        "label"
    ]

    df_results = pd.DataFrame(successful_rows, columns=required_columns)

    # Save to CSV
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(output_csv_path, index=False)
    logger.info(f"\nSaved validation CSV to: {output_csv_path.resolve()}")

    # Print validation report
    report = print_validation_report(
        df_results=df_results,
        successful_records=successful_records,
        failed_records=failed_records,
        total_attempted=len(target_records)
    )

    return {
        "df_results": df_results,
        "successful_records": successful_records,
        "failed_records": failed_records,
        "report_text": report
    }


def print_validation_report(
    df_results: pd.DataFrame,
    successful_records: List[str],
    failed_records: List[Dict[str, str]],
    total_attempted: int
) -> str:
    """Generate and print concise validation report."""
    lines = []
    lines.append("=" * 80)
    lines.append("        OBJECTIVE 4 — 10-RECORD FEATURE EXTRACTION VALIDATION REPORT        ")
    lines.append("=" * 80)
    lines.append(f"Total Records Attempted:  {total_attempted}")
    lines.append(f"Successful Records Count: {len(successful_records)}")
    lines.append(f"Failed Records Count:     {len(failed_records)}")
    lines.append("-" * 80)

    lines.append("SUCCESSFUL RECORDS:")
    if successful_records:
        lines.append(f"  {', '.join(successful_records)}")
    else:
        lines.append("  None")

    lines.append("\nFAILED RECORDS:")
    if failed_records:
        for item in failed_records:
            lines.append(f"  - {item['record']}: {item['error']}")
    else:
        lines.append("  None (0 failures)")

    lines.append("-" * 80)

    # Label distribution
    lines.append("LABEL DISTRIBUTION:")
    if not df_results.empty:
        label_counts = df_results["label"].value_counts().to_dict()
        for lbl, count in label_counts.items():
            lines.append(f"  {lbl}: {count}")
    else:
        lines.append("  No records processed.")

    lines.append("-" * 80)

    # NaN / Inf Check
    lines.append("NaN / INF STATISTICAL SANITY CHECK:")
    feature_cols = ["sampen_imf1", "median_freq_imf1_hz", "rms_imf1", "rms_imf2", "hilbert_par_imf1"]
    if not df_results.empty:
        total_nan = int(df_results[feature_cols].isna().sum().sum())
        total_inf = int(np.isinf(df_results[feature_cols].values).sum())
        lines.append(f"  Total NaN feature count: {total_nan}")
        lines.append(f"  Total Inf feature count: {total_inf}")
        lines.append(f"  Sanity Check Result:     {'PASS' if (total_nan == 0 and total_inf == 0) else 'FAIL'}")
    else:
        lines.append("  No data available.")

    lines.append("-" * 80)

    # Feature statistics (min, max, mean)
    lines.append("FEATURE SUMMARY STATISTICS (MIN / MAX / MEAN):")
    lines.append(f"{'Feature Name':<25} | {'Min':<14} | {'Max':<14} | {'Mean':<14}")
    lines.append("-" * 80)

    if not df_results.empty:
        for col in feature_cols:
            col_min = df_results[col].min()
            col_max = df_results[col].max()
            col_mean = df_results[col].mean()

            if "rms" in col:
                lines.append(f"{col:<25} | {col_min:<14.6e} | {col_max:<14.6e} | {col_mean:<14.6e}")
            else:
                lines.append(f"{col:<25} | {col_min:<14.6f} | {col_max:<14.6f} | {col_mean:<14.6f}")
    lines.append("=" * 80)

    report_str = "\n".join(lines)
    print("\n" + report_str + "\n")
    return report_str


if __name__ == "__main__":
    validate_10_records()
