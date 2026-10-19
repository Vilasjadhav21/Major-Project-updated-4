"""
Multi-Window EHG Feature Extraction Pipeline.

Extracts canonical EHG features across five signal window durations:
1. 1 minute  (60 seconds, start=180.0s)
2. 3 minutes (180 seconds, start=180.0s)
3. 5 minutes (300 seconds, start=180.0s)
4. 10 minutes (600 seconds, start=180.0s)
5. 30 minutes (1800 seconds, start=0.0s)

Harmonized across PhysioNet TPEHGDB v1.0.1 and TPEHGT v1.0.0.
Outputs:
  - outputs/features/window_1min_features.csv
  - outputs/features/window_3min_features.csv
  - outputs/features/window_5min_features.csv
  - outputs/features/window_10min_features.csv
  - outputs/features/window_30min_features.csv
  - outputs/features/window_extraction_audit_log.csv
"""

import concurrent.futures
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import wfdb

from src.audit_combined_datasets import parse_v100_header_comments
from src.config import DATA_TPEHGDB_DIR, DATA_TPEHGT_V1_0_0_DIR, FEATURES_DIR
from src.data_loader import TPEHGDataLoader, load_record, load_tpehgt_record
from src.emd_processing import perform_emd
from src.feature_extraction import extract_canonical_features
from src.metadata import get_label
from src.window_extraction import EHGWindowExtractor

logger = logging.getLogger("multi_window_feature_extraction")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# 5 target window durations in seconds
WINDOWS: Dict[str, float] = {
    "1_min": 60.0,
    "3_min": 180.0,
    "5_min": 300.0,
    "10_min": 600.0,
    "30_min": 1800.0,
}

# Starting time in seconds for each window duration
# 1m to 10m start at 180.0s to bypass initial filter settling artifacts (trim_boundary_seconds = 180.0s)
# 30m window starts at 0.0s because recordings are nominal 30m in length (1800s total)
WINDOW_START_TIMES: Dict[str, float] = {
    "1_min": 180.0,
    "3_min": 180.0,
    "5_min": 180.0,
    "10_min": 180.0,
    "30_min": 0.0,
}


def get_all_dataset_records() -> List[Dict[str, Any]]:
    """
    Collate record descriptors across both TPEHGDB v1.0.1 and TPEHGT v1.0.0.
    """
    records_info = []

    # 1. TPEHGDB v1.0.1
    loader_v101 = TPEHGDataLoader(data_dir=DATA_TPEHGDB_DIR)
    recs_v101 = loader_v101.read_records_list()
    smr_path = DATA_TPEHGDB_DIR / "tpehgdb.smr"

    for r in recs_v101:
        clean_stem = Path(r).name
        meta = get_label(r, smr_filepath=smr_path)
        records_info.append({
            "record_id": clean_stem,
            "composite_id": f"tpehgdb_v1.0.1:{clean_stem}",
            "dataset_version": "v1.0.1",
            "channel_target": "S1 0.08-4 Hz",
            "label": int(meta["target"]),
            "data_dir": str(DATA_TPEHGDB_DIR),
            "loader_type": "v101",
            "is_control": False,
        })

    # 2. TPEHGT v1.0.0
    loader_v100 = TPEHGDataLoader(data_dir=DATA_TPEHGT_V1_0_0_DIR)
    recs_v100 = loader_v100.read_records_list()

    for r in recs_v100:
        clean_stem = Path(r).name
        comm = parse_v100_header_comments(DATA_TPEHGT_V1_0_0_DIR / f"{clean_stem}.hea")
        rec_type = comm.get("RecType", "")

        is_control = "non-pregnant" in rec_type.lower() or clean_stem.startswith("tpehgt_n")
        if is_control:
            label = None
        elif "preterm" in rec_type.lower() or clean_stem.startswith("tpehgt_p"):
            label = 1
        elif "term" in rec_type.lower() or clean_stem.startswith("tpehgt_t"):
            label = 0
        else:
            label = None

        records_info.append({
            "record_id": clean_stem,
            "composite_id": f"tpehgt_v1.0.0:{clean_stem}",
            "dataset_version": "v1.0.0",
            "channel_target": "EHG1_Butter-4-bi-0.08-5",
            "label": label,
            "data_dir": str(DATA_TPEHGT_V1_0_0_DIR),
            "loader_type": "v100",
            "is_control": is_control,
        })

    return records_info


def process_record_for_window(
    record_info: Dict[str, Any],
    window_name: str,
    window_duration: float,
    start_time: float,
) -> Dict[str, Any]:
    """
    Process a single record for a specific window duration.
    Extracts the window, decomposes with EMD, and calculates the 5 canonical features.
    """
    rec_id = record_info["record_id"]
    comp_id = record_info["composite_id"]
    dataset_ver = record_info["dataset_version"]
    label = record_info["label"]

    result = {
        "window": window_name,
        "window_duration_sec": window_duration,
        "start_time_sec": start_time,
        "record_id": rec_id,
        "composite_id": comp_id,
        "patient_id": comp_id,
        "dataset_version": dataset_ver,
        "label": label,
        "status": "FAILED",
        "reason": "Unknown",
        "sampling_frequency": None,
        "required_samples": None,
        "available_samples": None,
        "available_duration_sec": None,
        "features": None,
    }

    # 1. Exclude non-pregnant control records
    if record_info["is_control"] or label is None:
        result["status"] = "EXCLUDED"
        result["reason"] = "Non-pregnant control record (not applicable to delivery outcome)"
        return result

    try:
        # 2. Load record header and data
        data_dir = Path(record_info["data_dir"])
        loader = TPEHGDataLoader(data_dir=data_dir)
        record = loader.load_record(rec_id)

        actual_fs = float(record.fs)
        result["sampling_frequency"] = actual_fs

        total_samples = int(record.sig_len)
        total_duration = total_samples / actual_fs
        result["available_samples"] = total_samples
        result["available_duration_sec"] = total_duration

        # 3. Calculate required sample bounds programmatically from actual_fs
        req_samples = int(round(actual_fs * window_duration))
        start_sample = int(round(actual_fs * start_time))
        end_sample = start_sample + req_samples

        result["required_samples"] = req_samples

        # 4. Check for sufficient recording duration without padding
        if end_sample > total_samples:
            result["status"] = "EXCLUDED"
            result["reason"] = (
                f"Signal duration {total_duration:.2f}s is shorter than requested window "
                f"[{start_time:.1f}s, {start_time + window_duration:.1f}s] "
                f"({end_sample} samples required, {total_samples} available)"
            )
            return result

        # 5. Extract signal window
        extractor = EHGWindowExtractor()
        ch_idx, canonical_ch_name = extractor.resolve_channel_index(record, record_info["channel_target"])
        extracted_signal = record.p_signal[start_sample:end_sample, ch_idx]

        if len(extracted_signal) != req_samples:
            result["status"] = "FAILED"
            result["reason"] = f"Extracted {len(extracted_signal)} samples, expected {req_samples}."
            return result

        if np.isnan(extracted_signal).any():
            result["status"] = "FAILED"
            result["reason"] = "Extracted signal contains NaN values."
            return result
        if np.isinf(extracted_signal).any():
            result["status"] = "FAILED"
            result["reason"] = "Extracted signal contains Infinite values."
            return result

        # 6. Apply EMD decomposition separately on this window
        imfs = perform_emd(extracted_signal, sampling_frequency=actual_fs)
        if len(imfs) < 2:
            result["status"] = "FAILED"
            result["reason"] = f"EMD produced {len(imfs)} IMFs, expected at least 2."
            return result

        imf1 = imfs[0]
        imf2 = imfs[1]

        # 7. Extract 5 canonical features
        feats = extract_canonical_features(imf1, imf2, sampling_frequency=actual_fs, expected_samples=req_samples)

        result["features"] = {
            "sampen_imf1": float(feats["sampen_imf1"]),
            "median_freq_imf1_hz": float(feats["median_freq_imf1_hz"]),
            "rms_imf1": float(feats["rms_imf1"]),
            "rms_imf2": float(feats["rms_imf2"]),
            "hilbert_par_imf1": float(feats["hilbert_par_imf1"]),
        }
        result["status"] = "SUCCESS"
        result["reason"] = "OK"
        return result

    except Exception as err:
        result["status"] = "FAILED"
        result["reason"] = str(err)
        return result


def extract_features_for_all_windows(
    max_workers: Optional[int] = None,
    output_dir: Path = FEATURES_DIR,
) -> Dict[str, pd.DataFrame]:
    """
    Run multi-window feature extraction across all five window durations.
    Saves CSV datasets for each window and an audit log of all exclusions.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    all_records = get_all_dataset_records()

    num_workers = max_workers or min(os.cpu_count() or 4, 12)
    logger.info(f"Loaded {len(all_records)} total records from TPEHGDB and TPEHGT.")
    logger.info(f"Extracting features across {len(WINDOWS)} durations using {num_workers} worker processes...")

    window_dfs: Dict[str, pd.DataFrame] = {}
    audit_logs: List[Dict[str, Any]] = []

    for window_name, window_duration in WINDOWS.items():
        start_time = WINDOW_START_TIMES[window_name]
        logger.info(f"\n>>> Extracting Window: {window_name} (Duration: {window_duration}s, Start: {start_time}s) <<<")
        t0 = time.time()

        tasks = [
            (rec, window_name, window_duration, start_time)
            for rec in all_records
        ]

        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_rec = {
                executor.submit(process_record_for_window, rec, window_name, window_duration, start_time): rec
                for rec in all_records
            }
            for future in concurrent.futures.as_completed(future_to_rec):
                res = future.result()
                results.append(res)

        elapsed = time.time() - t0

        # Sort results deterministically by composite_id
        results.sort(key=lambda x: x["composite_id"])

        success_rows = []
        for r in results:
            audit_logs.append({
                "window": r["window"],
                "window_duration_sec": r["window_duration_sec"],
                "start_time_sec": r["start_time_sec"],
                "composite_id": r["composite_id"],
                "record_id": r["record_id"],
                "dataset_version": r["dataset_version"],
                "label": r["label"],
                "status": r["status"],
                "reason": r["reason"],
                "sampling_frequency": r["sampling_frequency"],
                "required_samples": r["required_samples"],
                "available_samples": r["available_samples"],
                "available_duration_sec": r["available_duration_sec"],
            })

            if r["status"] == "SUCCESS" and r["features"] is not None:
                row = {
                    "record_id": r["composite_id"],
                    "patient_id": r["patient_id"],
                    "dataset_version": r["dataset_version"],
                    "sampen_imf1": r["features"]["sampen_imf1"],
                    "median_freq_imf1_hz": r["features"]["median_freq_imf1_hz"],
                    "rms_imf1": r["features"]["rms_imf1"],
                    "rms_imf2": r["features"]["rms_imf2"],
                    "hilbert_par_imf1": r["features"]["hilbert_par_imf1"],
                    "label": r["label"],
                }
                success_rows.append(row)

        df_win = pd.DataFrame(success_rows)
        window_dfs[window_name] = df_win

        csv_name = f"window_{window_name.replace('_', '')}_features.csv"
        # Standard filenames requested in Step 4:
        # outputs/features/window_1min_features.csv
        # outputs/features/window_3min_features.csv
        # outputs/features/window_5min_features.csv
        # outputs/features/window_10min_features.csv
        # outputs/features/window_30min_features.csv
        filename_map = {
            "1_min": "window_1min_features.csv",
            "3_min": "window_3min_features.csv",
            "5_min": "window_5min_features.csv",
            "10_min": "window_10min_features.csv",
            "30_min": "window_30min_features.csv",
        }
        out_csv_path = output_dir / filename_map[window_name]
        df_win.to_csv(out_csv_path, index=False)

        term_cnt = int((df_win["label"] == 0).sum()) if not df_win.empty else 0
        preterm_cnt = int((df_win["label"] == 1).sum()) if not df_win.empty else 0
        excluded_cnt = len(all_records) - len(df_win)

        logger.info(
            f"Completed {window_name} in {elapsed:.2f}s: "
            f"{len(df_win)} eligible (Term={term_cnt}, Preterm={preterm_cnt}), "
            f"{excluded_cnt} excluded -> Saved to {out_csv_path}"
        )

    # Save complete audit log
    audit_df = pd.DataFrame(audit_logs)
    audit_csv_path = output_dir / "window_extraction_audit_log.csv"
    audit_df.to_csv(audit_csv_path, index=False)
    logger.info(f"Saved complete extraction audit log to {audit_csv_path}")

    return window_dfs


if __name__ == "__main__":
    extract_features_for_all_windows()
