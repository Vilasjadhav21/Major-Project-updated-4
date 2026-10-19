"""
Batch Processing script for Objective 4: Rapid 1-Minute EHG Triage (TPEHGDB).

Processes all records from data/tpehgdb/RECORDS in a fault-tolerant manner.
For each record, extracts the 1-minute window (180-240s) from S1 0.08-4 Hz,
decomposes using EMD, extracts 5 target features (SampEn IMF-1, Median Freq IMF-1,
RMS IMF-1, RMS IMF-2, Hilbert PAR IMF-1), attaches clinical target label (Term=0, Preterm=1),
and outputs results to outputs/features/all_records_features.csv and detailed processing
log to outputs/features/all_records_processing_log.csv.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

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
from src.feature_extraction import extract_features

# Configure logging
logger = logging.getLogger("process_all_records")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def process_all_records(
    data_dir: Path = PROJECT_ROOT / "data" / "tpehgdb",
    output_dir: Path = PROJECT_ROOT / "outputs" / "features",
    max_records: Optional[int] = None
) -> Dict[str, Any]:
    """
    Process records listed in `data_dir / RECORDS` in a fault-tolerant batch loop.

    Args:
        data_dir: Directory containing RECORDS file and dataset subfolders.
        output_dir: Output directory for features and log CSV files.
        max_records: Optional cap on the number of records to process (for testing).

    Returns:
        Dictionary with results DataFrame, audit log DataFrame, stats, and sanity check results.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    features_csv_path = output_dir / "all_records_features.csv"
    log_csv_path = output_dir / "all_records_processing_log.csv"

    data_loader = TPEHGDataLoader(data_dir=data_dir)
    window_extractor = EHGWindowExtractor()

    # Read RECORDS file
    all_records = data_loader.read_records_list()
    if max_records is not None and max_records > 0:
        target_records = all_records[:max_records]
    else:
        target_records = all_records

    total_records = len(target_records)
    logger.info(f"Loaded {total_records} records to process from {data_dir / 'RECORDS'}")

    successful_features_rows: List[Dict[str, Any]] = []
    processing_log_rows: List[Dict[str, Any]] = []

    target_channel = "S1 0.08-4 Hz"
    window_start = 180.0
    window_end = 240.0
    expected_duration = window_end - window_start  # 60.0s
    expected_fs = 20.0
    expected_samples = 1200

    for idx, rec_entry in enumerate(target_records, 1):
        rec_stem = Path(rec_entry).name
        print(f"[{idx}/{total_records}] Processing {rec_stem}")

        # Default values for log tracking
        status = "FAILED"
        reason = "Unknown error"
        target_label_val: Optional[int] = None
        actual_fs: Optional[float] = None
        extracted_samples: Optional[int] = None
        imf_count: Optional[int] = None

        try:
            # 1. Obtain clinical label from metadata.py
            meta_info = get_label(rec_entry, smr_filepath=data_dir / "tpehgdb.smr")
            target_label_val = meta_info.get("target")  # 0 for Term, 1 for Preterm
            if target_label_val not in (0, 1):
                raise ValueError(f"Invalid label target '{target_label_val}' in metadata for record '{rec_entry}'. Expected 0 or 1.")

            # 2. Attempt WFDB loading
            wfdb_rec = load_record(rec_entry, data_dir=data_dir)
            actual_fs = float(wfdb_rec.fs)
            fs_tolerance = 0.01  # Documented tolerance in Hz for TPEHGDB v1.0.1 (nominal 20 Hz vs 20.00011 Hz header representation)
            if abs(actual_fs - expected_fs) > fs_tolerance:
                raise ValueError(
                    f"Sampling frequency mismatch: Expected {expected_fs} Hz (+/- {fs_tolerance} Hz), got {actual_fs} Hz."
                )

            # 3. Extract 180-240 seconds window and verify channel S1 0.08-4 Hz exists
            win_dict = window_extractor.extract_one_minute(
                record=wfdb_rec,
                channel_name=target_channel,
                start_time=window_start,
                window_duration=expected_duration
            )

            signal_win = win_dict["signal"]
            extracted_samples = win_dict["extracted_samples"]

            # 4. Verify sample count is exactly 1200
            if extracted_samples != expected_samples:
                raise ValueError(
                    f"Sample count mismatch: Expected exactly {expected_samples} samples for fs={actual_fs} Hz, "
                    f"got {extracted_samples} samples."
                )

            # 5. Run EMD decomposition
            imfs = perform_emd(signal_win, sampling_frequency=actual_fs)
            imf_count = len(imfs)

            # 6. Verify IMF-1 and IMF-2 exist
            if imf_count < 2:
                raise ValueError(f"EMD produced only {imf_count} IMFs, expected at least 2.")

            imf1 = imfs[0]
            imf2 = imfs[1]

            # 7. Calculate five target features
            feats = extract_features(imf1, imf2, sampling_frequency=actual_fs)

            sampen_imf1_val = feats["sampen_imf1"]
            med_freq_imf1_val = feats["median_frequency_imf1_hz"]
            rms_imf1_val = feats["rms_imf1"]
            rms_imf2_val = feats["rms_imf2"]
            hilbert_par_imf1_val = feats["hilbert_par_imf1"]

            feature_vals = [sampen_imf1_val, med_freq_imf1_val, rms_imf1_val, rms_imf2_val, hilbert_par_imf1_val]

            # Verify no NaN or Inf
            if any(np.isnan(v) for v in feature_vals):
                raise ValueError("NaN feature value detected.")
            if any(np.isinf(v) for v in feature_vals):
                raise ValueError("Infinite feature value detected.")

            # Append successful feature row
            features_row = {
                "record": rec_stem,
                "label": target_label_val,
                "sampen_imf1": sampen_imf1_val,
                "median_freq_imf1_hz": med_freq_imf1_val,
                "rms_imf1": rms_imf1_val,
                "rms_imf2": rms_imf2_val,
                "hilbert_par_imf1": hilbert_par_imf1_val
            }
            successful_features_rows.append(features_row)

            # Mark log success
            status = "SUCCESS"
            reason = "OK"

        except Exception as err:
            logger.warning(f"Record '{rec_stem}' failed processing: {err}")
            status = "FAILED"
            reason = str(err)

        # Log entry for every record attempted
        log_row = {
            "record": rec_stem,
            "status": status,
            "reason": reason,
            "label": target_label_val if target_label_val is not None else "N/A",
            "sampling_frequency": actual_fs if actual_fs is not None else "N/A",
            "window_samples": extracted_samples if extracted_samples is not None else "N/A",
            "imf_count": imf_count if imf_count is not None else "N/A"
        }
        processing_log_rows.append(log_row)

    # Convert to DataFrames
    feature_cols = [
        "record",
        "label",
        "sampen_imf1",
        "median_freq_imf1_hz",
        "rms_imf1",
        "rms_imf2",
        "hilbert_par_imf1"
    ]
    df_features = pd.DataFrame(successful_features_rows, columns=feature_cols)

    log_cols = [
        "record",
        "status",
        "reason",
        "label",
        "sampling_frequency",
        "window_samples",
        "imf_count"
    ]
    df_log = pd.DataFrame(processing_log_rows, columns=log_cols)

    # Save to CSV files
    df_features.to_csv(features_csv_path, index=False)
    df_log.to_csv(log_csv_path, index=False)

    logger.info(f"Saved feature matrix to: {features_csv_path}")
    logger.info(f"Saved processing audit log to: {log_csv_path}")

    # Compute summary statistics
    successful_count = len(df_features)
    failed_count = total_records - successful_count

    term_count = int((df_features["label"] == 0).sum()) if not df_features.empty else 0
    preterm_count = int((df_features["label"] == 1).sum()) if not df_features.empty else 0

    numeric_feature_cols = ["sampen_imf1", "median_freq_imf1_hz", "rms_imf1", "rms_imf2", "hilbert_par_imf1"]
    if not df_features.empty:
        nan_records_count = int(df_features[numeric_feature_cols].isna().any(axis=1).sum())
        inf_records_count = int(np.isinf(df_features[numeric_feature_cols].values).any(axis=1).sum())
    else:
        nan_records_count = 0
        inf_records_count = 0

    # Final Sanity Checks
    sanity_pass = True
    sanity_notes = []

    if not df_features.empty:
        # Check 1: Exactly 5 features per record
        if len(df_features.columns) - 2 != 5:  # minus record and label
            sanity_pass = False
            sanity_notes.append(f"Expected 5 feature columns, found {len(df_features.columns) - 2}")

        # Check 2: No NaN
        if nan_records_count > 0:
            sanity_pass = False
            sanity_notes.append(f"Found {nan_records_count} records with NaN features.")

        # Check 3: No Inf
        if inf_records_count > 0:
            sanity_pass = False
            sanity_notes.append(f"Found {inf_records_count} records with Inf features.")

        # Check 4: Labels are strictly 0 or 1
        invalid_labels = set(df_features["label"]) - {0, 1}
        if invalid_labels:
            sanity_pass = False
            sanity_notes.append(f"Invalid label values found: {invalid_labels}")

    report_txt_path = output_dir / "full_dataset_processing_report.txt"

    # Generate full dataset processing report text
    report_lines = [
        "=" * 80,
        "   OBJECTIVE 4: RAPID 1-MINUTE EHG TRIAGE — FULL DATASET PROCESSING REPORT",
        "=" * 80,
        "Dataset:                   PhysioNet TPEHGDB v1.0.1",
        "Target Channel:            S1 0.08–4 Hz (canonical 1_DOCFILT-4-0.08-4)",
        "Window Length:             60.0 seconds (180.0s to 240.0s)",
        "Sampling Frequency:        Read from header (Expected: 20.0 Hz)",
        "Expected Window Samples:   1200 samples",
        "EMD Extracted IMFs:        IMF-1, IMF-2",
        "Target Features (5):       sampen_imf1, median_freq_imf1_hz, rms_imf1, rms_imf2, hilbert_par_imf1",
        "Clinical Labels:           Term = 0, Preterm = 1 (from tpehgdb.smr)",
        "-" * 80,
        "1. PROCESSING SUMMARY STATISTICS",
        "-" * 80,
        f"Total Records Attempted:             {total_records}",
        f"Successfully Processed Records:     {successful_count}",
        f"Failed Records:                     {failed_count}",
        f"Term Records (label = 0):           {term_count}",
        f"Preterm Records (label = 1):        {preterm_count}",
        f"Records with Missing (NaN) Values:  {nan_records_count}",
        f"Records with Infinite Values:       {inf_records_count}",
        f"Sanity Check Status:                {'PASS' if sanity_pass else 'FAIL'}",
        "-" * 80,
        "2. FAILURE REASONS BREAKDOWN",
        "-" * 80,
    ]

    # Aggregate failure reasons
    failed_rows = [r for r in processing_log_rows if r["status"] == "FAILED"]
    if failed_rows:
        reason_counts = {}
        for fr in failed_rows:
            r = fr["reason"]
            reason_counts[r] = reason_counts.get(r, 0) + 1
        for reason_text, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            report_lines.append(f"  [{count} records]: {reason_text}")
    else:
        report_lines.append("  None (0 failed records)")

    report_lines.extend([
        "-" * 80,
        "3. PER-RECORD AUDIT LOG",
        "-" * 80,
        f"{'Record':<15} {'Status':<10} {'Label':<10} {'Fs (Hz)':<10} {'Samples':<10} {'IMFs':<8} {'Reason'}"
    ])

    for row in processing_log_rows:
        rec = str(row["record"])
        stat = str(row["status"])
        lbl = str(row["label"])
        fs_str = str(row["sampling_frequency"])
        samp_str = str(row["window_samples"])
        imf_str = str(row["imf_count"])
        rsn = str(row["reason"])
        report_lines.append(
            f"{rec:<15} {stat:<10} {lbl:<10} {fs_str:<10} {samp_str:<10} {imf_str:<8} {rsn}"
        )

    report_lines.extend([
        "=" * 80,
        "END OF FULL DATASET PROCESSING REPORT",
        "=" * 80,
        ""
    ])

    report_text = "\n".join(report_lines)
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    logger.info(f"Saved full processing report to: {report_txt_path}")

    # Print Summary Report
    print("\n" + "=" * 80)
    print("        OBJECTIVE 4 — TPEHGDB BATCH PROCESSING SUMMARY REPORT        ")
    print("=" * 80)
    print(f"Total records:              {total_records}")
    print(f"Successful records:         {successful_count}")
    print(f"Failed records:             {failed_count}")
    print(f"Term records (0):           {term_count}")
    print(f"Preterm records (1):        {preterm_count}")
    print(f"Records with NaN features:  {nan_records_count}")
    print(f"Records with Inf features:  {inf_records_count}")
    print("-" * 80)
    print(f"FINAL SANITY CHECK:        {'PASS' if sanity_pass else 'FAIL'}")
    if sanity_notes:
        for note in sanity_notes:
            print(f"  - Warning: {note}")
    print(f"Report saved to:           {report_txt_path}")
    print("=" * 80 + "\n")

    return {
        "df_features": df_features,
        "df_log": df_log,
        "report_text": report_text,
        "total_records": total_records,
        "successful_records": successful_count,
        "failed_records": failed_count,
        "term_records": term_count,
        "preterm_records": preterm_count,
        "nan_records_count": nan_records_count,
        "inf_records_count": inf_records_count,
        "sanity_pass": sanity_pass
    }


if __name__ == "__main__":
    process_all_records()
