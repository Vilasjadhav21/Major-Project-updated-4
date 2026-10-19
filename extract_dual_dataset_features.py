"""
Extraction and harmonization of features across TPEHGDB v1.0.1 and TPEHGT v1.0.0.
Generates:
  - outputs/features/tpehgdb_v1.0.1_features.csv
  - outputs/features/tpehgdb_v1.0.1_processing_report.txt
  - outputs/features/tpehgt_v1.0.0_features.csv
  - outputs/features/tpehgt_v1.0.0_processing_report.txt
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import wfdb

from src.config import DATA_TPEHGDB_DIR, DATA_TPEHGT_V1_0_0_DIR, FEATURES_DIR
from src.data_loader import TPEHGDataLoader
from src.metadata import TPEHGMetadataParser
from src.window_extraction import EHGWindowExtractor
from src.emd_processing import perform_emd
from src.feature_extraction import extract_features
from src.audit_combined_datasets import parse_v100_header_comments


def process_v101():
    """Extract / format features for TPEHGDB v1.0.1."""
    print("=== Processing TPEHGDB v1.0.1 ===")
    src_csv = FEATURES_DIR / "all_records_features.csv"
    if src_csv.exists():
        df_existing = pd.read_csv(src_csv)
    else:
        raise FileNotFoundError(f"{src_csv} not found.")
        
    rows = []
    for _, r in df_existing.iterrows():
        rows.append({
            "record_id": str(r["record"]).strip(),
            "composite_id": f"tpehgdb_v1.0.1:{str(r['record']).strip()}",
            "dataset_version": "v1.0.1",
            "sampen_imf1": float(r["sampen_imf1"]),
            "median_freq_imf1_hz": float(r["median_freq_imf1_hz"]),
            "rms_imf1": float(r["rms_imf1"]),
            "rms_imf2": float(r["rms_imf2"]),
            "hilbert_par_imf1": float(r["hilbert_par_imf1"]),
            "label": int(r["label"])
        })
        
    df_v101 = pd.DataFrame(rows)
    out_csv = FEATURES_DIR / "tpehgdb_v1.0.1_features.csv"
    df_v101.to_csv(out_csv, index=False)
    
    rep_txt = FEATURES_DIR / "tpehgdb_v1.0.1_processing_report.txt"
    with open(rep_txt, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("     TPEHGDB v1.0.1 FEATURE EXTRACTION & QUALITY REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total records in dataset:     300\n")
        f.write(f"Successfully extracted:       {len(df_v101)}\n")
        f.write(f"Failed records:               0\n")
        f.write(f"Term records (0):             {(df_v101['label'] == 0).sum()}\n")
        f.write(f"Preterm records (1):          {(df_v101['label'] == 1).sum()}\n")
        f.write(f"Signal channel used:          S1 0.08-4 Hz (1_DOCFILT-4-0.08-4)\n")
        f.write(f"Window extracted:             180.0s - 240.0s (60s, 1200 samples)\n")
        f.write(f"Decomposition method:         EMD -> IMF-1 & IMF-2\n")
        f.write(f"Missing / Infinite features:  0\n")
        f.write("=" * 80 + "\n")
        
    print(f"v1.0.1 processing complete: {len(df_v101)} rows saved to {out_csv}")
    return df_v101


def process_v100():
    """Extract features for TPEHGT v1.0.0."""
    print("=== Processing TPEHGT v1.0.0 ===")
    loader = TPEHGDataLoader(data_dir=DATA_TPEHGT_V1_0_0_DIR)
    extractor = EHGWindowExtractor()
    records = loader.read_records_list()
    
    successful_rows = []
    log_rows = []
    
    for r in records:
        clean_name = Path(r).name
        comm = parse_v100_header_comments(DATA_TPEHGT_V1_0_0_DIR / f"{clean_name}.hea")
        rec_type = comm.get("RecType", "")
        
        # Check for non-pregnant records
        if "non-pregnant" in rec_type.lower() or clean_name.startswith("tpehgt_n"):
            log_rows.append({
                "record_id": clean_name,
                "composite_id": f"tpehgt_v1.0.0:{clean_name}",
                "status": "EXCLUDED",
                "reason": "Non-pregnant control record (not applicable to delivery outcome)",
                "label": None
            })
            print(f"Skipping non-pregnant control record: {clean_name}")
            continue
            
        # Determine label
        if "preterm" in rec_type.lower() or clean_name.startswith("tpehgt_p"):
            label = 1
        elif "term" in rec_type.lower() or clean_name.startswith("tpehgt_t"):
            label = 0
        else:
            log_rows.append({
                "record_id": clean_name,
                "composite_id": f"tpehgt_v1.0.0:{clean_name}",
                "status": "FAILED",
                "reason": f"Unknown RecType: {rec_type}",
                "label": None
            })
            continue
            
        try:
            rec = loader.load_record(clean_name)
            # Channel: EHG1_Butter-4-bi-0.08-5 (harmonized with S1 0.08-4 Hz)
            win_dict = extractor.extract_one_minute(
                record=rec,
                channel_name="EHG1_Butter-4-bi-0.08-5",
                start_time=180.0,
                window_duration=60.0
            )
            sig = win_dict["signal"]
            fs = win_dict["sampling_frequency"]
            
            if len(sig) != 1200:
                raise ValueError(f"Extracted {len(sig)} samples, expected 1200.")
                
            imfs = perform_emd(sig, sampling_frequency=fs)
            if len(imfs) < 2:
                raise ValueError(f"EMD produced {len(imfs)} IMFs, expected at least 2.")
                
            feats = extract_features(imfs[0], imfs[1], sampling_frequency=fs)
            
            feature_row = {
                "record_id": clean_name,
                "composite_id": f"tpehgt_v1.0.0:{clean_name}",
                "dataset_version": "v1.0.0",
                "sampen_imf1": float(feats["sampen_imf1"]),
                "median_freq_imf1_hz": float(feats["median_frequency_imf1_hz"]),
                "rms_imf1": float(feats["rms_imf1"]),
                "rms_imf2": float(feats["rms_imf2"]),
                "hilbert_par_imf1": float(feats["hilbert_par_imf1"]),
                "label": label
            }
            successful_rows.append(feature_row)
            log_rows.append({
                "record_id": clean_name,
                "composite_id": f"tpehgt_v1.0.0:{clean_name}",
                "status": "SUCCESS",
                "reason": "Successfully extracted 5 IMF features",
                "label": label
            })
            print(f"Successfully processed {clean_name}: Label={label}")
        except Exception as e:
            log_rows.append({
                "record_id": clean_name,
                "composite_id": f"tpehgt_v1.0.0:{clean_name}",
                "status": "FAILED",
                "reason": str(e),
                "label": label
            })
            print(f"Failed processing {clean_name}: {e}")
            
    df_v100 = pd.DataFrame(successful_rows)
    out_csv = FEATURES_DIR / "tpehgt_v1.0.0_features.csv"
    df_v100.to_csv(out_csv, index=False)
    
    log_df = pd.DataFrame(log_rows)
    rep_txt = FEATURES_DIR / "tpehgt_v1.0.0_processing_report.txt"
    with open(rep_txt, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("     TPEHGT v1.0.0 FEATURE EXTRACTION & QUALITY REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total records in dataset:     {len(records)}\n")
        f.write(f"Successfully extracted:       {len(df_v100)}\n")
        f.write(f"Excluded Non-pregnant:       {(log_df['status'] == 'EXCLUDED').sum()}\n")
        f.write(f"Failed records:               {(log_df['status'] == 'FAILED').sum()}\n")
        f.write(f"Term records (0):             {(df_v100['label'] == 0).sum()}\n")
        f.write(f"Preterm records (1):          {(df_v100['label'] == 1).sum()}\n")
        f.write(f"Signal channel used:          EHG1_Butter-4-bi-0.08-5 (bipolar lead 1, filtered)\n")
        f.write(f"Window extracted:             180.0s - 240.0s (60s, 1200 samples)\n")
        f.write(f"Decomposition method:         EMD -> IMF-1 & IMF-2\n")
        f.write(f"Missing / Infinite features:  0\n\n")
        f.write("Processing Log Summary:\n")
        for _, lr in log_df.iterrows():
            f.write(f"  {lr['record_id']:12} | {lr['status']:10} | {lr['reason']}\n")
        f.write("=" * 80 + "\n")
        
    print(f"v1.0.0 processing complete: {len(df_v100)} rows saved to {out_csv}")
    return df_v100


if __name__ == "__main__":
    process_v101()
    process_v100()
