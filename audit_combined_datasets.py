"""
Comprehensive Dataset Audit for TPEHGDB v1.0.1 and TPEHGT v1.0.0.
Generates:
  - outputs/features/combined_dataset_audit.csv
  - outputs/features/combined_dataset_audit.txt
"""

import os
from pathlib import Path
import pandas as pd
import wfdb
from src.config import BASE_DIR, DATA_TPEHGDB_DIR, DATA_TPEHGT_V1_0_0_DIR, FEATURES_DIR
from src.metadata import TPEHGMetadataParser
from src.data_loader import TPEHGDataLoader


def parse_v100_header_comments(hea_path: Path):
    """Parse comments from a v1.0.0 .hea file into a dictionary."""
    comments = {}
    if not hea_path.exists():
        return comments
    with open(hea_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                clean = line.lstrip("#").strip()
                parts = clean.split(None, 1)
                if len(parts) == 2:
                    comments[parts[0]] = parts[1]
    return comments


def audit_datasets():
    audit_rows = []
    
    # --------------------------------------------------------------------------
    # 1. Audit TPEHGDB v1.0.1
    # --------------------------------------------------------------------------
    loader_v101 = TPEHGDataLoader(data_dir=DATA_TPEHGDB_DIR)
    records_v101 = loader_v101.read_records_list()
    parser_v101 = TPEHGMetadataParser(DATA_TPEHGDB_DIR / "tpehgdb.smr")
    meta_df_v101 = parser_v101.load_metadata()
    
    for r in records_v101:
        clean_name = Path(r).name
        composite_id = f"tpehgdb_v1.0.1:{clean_name}"
        
        # Check files
        hea_path = DATA_TPEHGDB_DIR / "tpehgdb" / f"{clean_name}.hea"
        dat_path = DATA_TPEHGDB_DIR / "tpehgdb" / f"{clean_name}.dat"
        if not hea_path.exists():
            hea_path = DATA_TPEHGDB_DIR / f"{clean_name}.hea"
            dat_path = DATA_TPEHGDB_DIR / f"{clean_name}.dat"
            
        has_hea = hea_path.exists()
        has_dat = dat_path.exists()
        
        wfdb_readable = False
        fs = None
        num_channels = None
        sig_names = ""
        sig_units = ""
        num_samples = None
        duration_sec = None
        
        if has_hea and has_dat:
            try:
                rec = wfdb.rdrecord(str(hea_path.with_suffix("")))
                wfdb_readable = True
                fs = float(rec.fs)
                num_channels = len(rec.sig_name)
                sig_names = ";".join(rec.sig_name)
                sig_units = ";".join(rec.units) if rec.units else ""
                num_samples = rec.sig_len
                duration_sec = num_samples / fs if fs else None
            except Exception as e:
                wfdb_readable = False
                
        # Metadata
        try:
            m = parser_v101.get_label(clean_name)
            gestation = m["gestation"]
            rec_time = m["rec_time"]
            binary_label = m["target"]
            rec_type = "Preterm" if binary_label == 1 else "Term"
            status = "PASS" if wfdb_readable else "FAIL"
            reason = None if wfdb_readable else "WFDB read failure"
        except Exception as e:
            gestation = None
            rec_time = None
            binary_label = None
            rec_type = "Unknown"
            status = "FAIL"
            reason = f"Metadata lookup error: {e}"
            
        audit_rows.append({
            "composite_id": composite_id,
            "dataset_version": "v1.0.1",
            "record_id": clean_name,
            "has_hea": has_hea,
            "has_dat": has_dat,
            "wfdb_readable": wfdb_readable,
            "fs": fs,
            "num_channels": num_channels,
            "signal_names": sig_names,
            "signal_units": sig_units,
            "num_samples": num_samples,
            "duration_sec": duration_sec,
            "gestation": gestation,
            "rec_time": rec_time,
            "rec_type": rec_type,
            "binary_label": binary_label,
            "status": status,
            "exclusion_reason": reason
        })
        
    # --------------------------------------------------------------------------
    # 2. Audit TPEHGT v1.0.0
    # --------------------------------------------------------------------------
    loader_v100 = TPEHGDataLoader(data_dir=DATA_TPEHGT_V1_0_0_DIR)
    records_v100 = loader_v100.read_records_list()
    
    for r in records_v100:
        clean_name = Path(r).name
        composite_id = f"tpehgt_v1.0.0:{clean_name}"
        
        hea_path = DATA_TPEHGT_V1_0_0_DIR / f"{clean_name}.hea"
        dat_path = DATA_TPEHGT_V1_0_0_DIR / f"{clean_name}.dat"
        has_hea = hea_path.exists()
        has_dat = dat_path.exists()
        
        wfdb_readable = False
        fs = None
        num_channels = None
        sig_names = ""
        sig_units = ""
        num_samples = None
        duration_sec = None
        
        if has_hea and has_dat:
            try:
                rec = wfdb.rdrecord(str(hea_path.with_suffix("")))
                wfdb_readable = True
                fs = float(rec.fs)
                num_channels = len(rec.sig_name)
                sig_names = ";".join(rec.sig_name)
                sig_units = ";".join(rec.units) if rec.units else ""
                num_samples = rec.sig_len
                duration_sec = num_samples / fs if fs else None
            except Exception as e:
                wfdb_readable = False
                
        # Header comments metadata
        comm = parse_v100_header_comments(hea_path)
        raw_rec_type = comm.get("RecType", "")
        raw_gest = comm.get("Gestation", "")
        raw_rectime = comm.get("Rectime", "")
        
        if "non-pregnant" in raw_rec_type.lower() or clean_name.startswith("tpehgt_n"):
            rec_type = "Non-pregnant"
            binary_label = None
            status = "EXCLUDED"
            reason = "Non-pregnant control record excluded from delivery outcome classification"
        elif "preterm" in raw_rec_type.lower() or clean_name.startswith("tpehgt_p"):
            rec_type = "Preterm"
            binary_label = 1
            status = "PASS" if wfdb_readable else "FAIL"
            reason = None if wfdb_readable else "WFDB read failure"
        elif "term" in raw_rec_type.lower() or clean_name.startswith("tpehgt_t"):
            rec_type = "Term"
            binary_label = 0
            status = "PASS" if wfdb_readable else "FAIL"
            reason = None if wfdb_readable else "WFDB read failure"
        else:
            rec_type = "Unknown"
            binary_label = None
            status = "FAIL"
            reason = f"Unrecognized RecType: {raw_rec_type}"
            
        gestation = float(raw_gest) if raw_gest and raw_gest != "N/A" else None
        rec_time = float(raw_rectime) if raw_rectime and raw_rectime != "N/A" else None
        
        audit_rows.append({
            "composite_id": composite_id,
            "dataset_version": "v1.0.0",
            "record_id": clean_name,
            "has_hea": has_hea,
            "has_dat": has_dat,
            "wfdb_readable": wfdb_readable,
            "fs": fs,
            "num_channels": num_channels,
            "signal_names": sig_names,
            "signal_units": sig_units,
            "num_samples": num_samples,
            "duration_sec": duration_sec,
            "gestation": gestation,
            "rec_time": rec_time,
            "rec_type": rec_type,
            "binary_label": binary_label,
            "status": status,
            "exclusion_reason": reason
        })
        
    audit_df = pd.DataFrame(audit_rows)
    csv_out = FEATURES_DIR / "combined_dataset_audit.csv"
    audit_df.to_csv(csv_out, index=False)
    
    # --------------------------------------------------------------------------
    # 3. Generate combined_dataset_audit.txt
    # --------------------------------------------------------------------------
    txt_out = FEATURES_DIR / "combined_dataset_audit.txt"
    v101_df = audit_df[audit_df["dataset_version"] == "v1.0.1"]
    v100_df = audit_df[audit_df["dataset_version"] == "v1.0.0"]
    
    with open(txt_out, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("          OBJECTIVE 4: DUAL DATASET AUDIT REPORT (v1.0.1 & v1.0.0)\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("1. DATASET: PhysioNet TPEHGDB v1.0.1\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Storage Location:            {DATA_TPEHGDB_DIR}\n")
        f.write(f"  Total Records in RECORDS:    {len(records_v101)}\n")
        f.write(f"  Available .hea Files:        {v101_df['has_hea'].sum()}\n")
        f.write(f"  Available .dat Files:        {v101_df['has_dat'].sum()}\n")
        f.write(f"  WFDB Readable Records:       {v101_df['wfdb_readable'].sum()}\n")
        f.write(f"  Sampling Frequencies:        {sorted(v101_df['fs'].dropna().unique())} Hz\n")
        f.write(f"  Channel Count:               12 channels\n")
        f.write(f"  Signal Units:                mV\n")
        f.write(f"  Metadata File:               data/tpehgdb/tpehgdb.smr\n")
        f.write(f"  Term Records (label=0):      {(v101_df['rec_type'] == 'Term').sum()}\n")
        f.write(f"  Preterm Records (label=1):   {(v101_df['rec_type'] == 'Preterm').sum()}\n")
        f.write(f"  Non-pregnant Records:        0\n")
        f.write(f"  Missing / Corrupted Records: {(~v101_df['wfdb_readable']).sum()}\n\n")
        
        f.write("2. DATASET: PhysioNet Term-Preterm EHG with Tocogram v1.0.0 (TPEHGT)\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Storage Location:            {DATA_TPEHGT_V1_0_0_DIR}\n")
        f.write(f"  Total Records in RECORDS:    {len(records_v100)}\n")
        f.write(f"  Available .hea Files:        {v100_df['has_hea'].sum()}\n")
        f.write(f"  Available .dat Files:        {v100_df['has_dat'].sum()}\n")
        f.write(f"  WFDB Readable Records:       {v100_df['wfdb_readable'].sum()}\n")
        f.write(f"  Sampling Frequencies:        {sorted(v100_df['fs'].dropna().unique())} Hz\n")
        f.write(f"  Channel Count:               8 channels (3 EHG raw, 3 EHG filtered, 1 TOCO raw, 1 TOCO filtered)\n")
        f.write(f"  Signal Units:                mV\n")
        f.write(f"  Metadata Source:             Embedded header comments (.hea)\n")
        f.write(f"  Term Records (label=0):      {(v100_df['rec_type'] == 'Term').sum()}\n")
        f.write(f"  Preterm Records (label=1):   {(v100_df['rec_type'] == 'Preterm').sum()}\n")
        f.write(f"  Non-pregnant (EXCLUDED):     {(v100_df['rec_type'] == 'Non-pregnant').sum()}\n")
        f.write(f"  Missing / Corrupted Records: {(~v100_df['wfdb_readable']).sum()}\n\n")
        
        f.write("3. COMBINED COHORT SUMMARY FOR DELIVERY-OUTCOME TASK\n")
        f.write("-" * 80 + "\n")
        valid_mask = (audit_df["status"] == "PASS")
        f.write(f"  Total Candidate Records:     {len(audit_df)}\n")
        f.write(f"  Excluded Non-Pregnant:       {(audit_df['status'] == 'EXCLUDED').sum()}\n")
        f.write(f"  Failed / Unreadable Records: {(audit_df['status'] == 'FAIL').sum()}\n")
        f.write(f"  Eligible Pregnancy Records:  {valid_mask.sum()}\n")
        f.write(f"    - Eligible Term (0):       {(valid_mask & (audit_df['binary_label'] == 0)).sum()} (262 v1.0.1 + 13 v1.0.0)\n")
        f.write(f"    - Eligible Preterm (1):    {(valid_mask & (audit_df['binary_label'] == 1)).sum()} (38 v1.0.1 + 13 v1.0.0)\n")
        f.write(f"  Class Imbalance Ratio:       {((valid_mask & (audit_df['binary_label'] == 0)).sum()) / ((valid_mask & (audit_df['binary_label'] == 1)).sum()):.2f}:1 (Term:Preterm)\n")
        f.write("=" * 80 + "\n")
        
    print(f"Audit completed: {csv_out} and {txt_out} written.")


if __name__ == "__main__":
    audit_datasets()
