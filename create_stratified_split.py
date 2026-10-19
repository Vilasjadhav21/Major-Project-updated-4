"""
Reproducible Leakage-Safe Stratified Train/Test Split on Combined EHG Features.
Preserves existing train/test files as backups.
Generates:
  - outputs/features/train_dataset.csv
  - outputs/features/test_dataset.csv
  - outputs/features/train_test_split_report.txt
"""

import shutil
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import FEATURES_DIR


def make_split():
    combined_csv = FEATURES_DIR / "combined_features.csv"
    df = pd.read_csv(combined_csv)
    
    # 1. Back up existing train/test files
    train_file = FEATURES_DIR / "train_dataset.csv"
    test_file = FEATURES_DIR / "test_dataset.csv"
    
    backup_train = FEATURES_DIR / "train_dataset_v101_backup.csv"
    backup_test = FEATURES_DIR / "test_dataset_v101_backup.csv"
    
    if train_file.exists():
        shutil.copyfile(train_file, backup_train)
        print(f"Backed up {train_file} -> {backup_train}")
    if test_file.exists():
        shutil.copyfile(test_file, backup_test)
        print(f"Backed up {test_file} -> {backup_test}")
        
    # 2. Stratification strata combining dataset_version and label
    df["strata"] = df["dataset_version"] + "_" + df["label"].astype(str)
    
    # 80/20 stratified split
    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df["strata"]
    )
    
    # Drop helper strata column
    train_df = train_df.drop(columns=["strata"]).copy()
    test_df = test_df.drop(columns=["strata"]).copy()
    
    # Verify zero leakage / overlap
    train_ids = set(train_df["record_id"])
    test_ids = set(test_df["record_id"])
    overlap = train_ids.intersection(test_ids)
    assert len(overlap) == 0, f"DATA LEAKAGE DETECTED! Overlapping record IDs: {overlap}"
    
    # Save splits
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)
    print(f"Saved train set ({len(train_df)} rows) to {train_file}")
    print(f"Saved test set ({len(test_df)} rows) to {test_file}")
    
    # Generate split report
    rep_txt = FEATURES_DIR / "train_test_split_report.txt"
    with open(rep_txt, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("      OBJECTIVE 4: STRATIFIED TRAIN/TEST SPLIT INTEGRITY REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("1. SPLIT SUMMARY:\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Total Cohort Records:        {len(df)}\n")
        f.write(f"  Training Split (80.0%):      {len(train_df)}\n")
        f.write(f"  Holdout Test Split (20.0%):  {len(test_df)}\n")
        f.write(f"  Random State Seed:           42\n")
        f.write(f"  Record Overlap:              0 (PASS - Strictly Disjoint)\n\n")
        
        f.write("2. TRAINING SPLIT COMPOSITION (N=260):\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Term Records (0):            {(train_df['label'] == 0).sum()}\n")
        f.write(f"  Preterm Records (1):         {(train_df['label'] == 1).sum()}\n")
        f.write(f"  v1.0.1 Records:              {(train_df['dataset_version'] == 'v1.0.1').sum()}\n")
        f.write(f"  v1.0.0 Records:              {(train_df['dataset_version'] == 'v1.0.0').sum()}\n")
        f.write(f"  Imbalance Ratio:             {((train_df['label'] == 0).sum()) / ((train_df['label'] == 1).sum()):.2f}:1\n\n")
        
        f.write("3. INDEPENDENT TEST SPLIT COMPOSITION (N=66):\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Term Records (0):            {(test_df['label'] == 0).sum()}\n")
        f.write(f"  Preterm Records (1):         {(test_df['label'] == 1).sum()}\n")
        f.write(f"  v1.0.1 Records:              {(test_df['dataset_version'] == 'v1.0.1').sum()}\n")
        f.write(f"  v1.0.0 Records:              {(test_df['dataset_version'] == 'v1.0.0').sum()}\n")
        f.write(f"  Imbalance Ratio:             {((test_df['label'] == 0).sum()) / ((test_df['label'] == 1).sum()):.2f}:1\n\n")
        
        f.write("4. SUBGROUP MATRIX:\n")
        f.write("-" * 80 + "\n")
        sub_train = train_df.groupby(["dataset_version", "label"]).size().unstack(fill_value=0)
        sub_test = test_df.groupby(["dataset_version", "label"]).size().unstack(fill_value=0)
        f.write("Training Split Subgroups (Rows=Dataset, Cols=Label):\n")
        f.write(sub_train.to_string() + "\n\n")
        f.write("Test Split Subgroups (Rows=Dataset, Cols=Label):\n")
        f.write(sub_test.to_string() + "\n")
        f.write("=" * 80 + "\n")
        
    print(f"Split report written to {rep_txt}")


if __name__ == "__main__":
    make_split()
