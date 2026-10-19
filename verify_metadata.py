"""
Verification script for PhysioNet TPEHGDB v1.0.1 metadata parsing and label extraction.

Loads tpehgdb.smr, validates data integrity against records list, prints verification report,
tests get_label(), and exports derived metadata CSV to outputs/features/tpehgdb_metadata.csv.
"""

import sys
from pathlib import Path

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.data_loader import TPEHGDataLoader
from src.metadata import TPEHGMetadataParser, load_metadata, get_label


def verify_metadata():
    """Execute complete metadata verification and output report."""
    data_dir = PROJECT_ROOT / "data" / "tpehgdb"
    smr_path = data_dir / "tpehgdb.smr"
    output_dir = PROJECT_ROOT / "outputs" / "features"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_output_path = output_dir / "tpehgdb_metadata.csv"

    # 1, 2, 3, 4, 5. Parse metadata using load_metadata()
    parser = TPEHGMetadataParser(smr_filepath=smr_path)
    df_meta = parser.load_metadata()

    # 6. Read records from data/tpehgdb/RECORDS
    data_loader = TPEHGDataLoader(data_dir=data_dir)
    records_list = data_loader.read_records_list()
    record_stems = [Path(r).name for r in records_list]

    # Data integrity & validation checks
    n_meta_rows = len(df_meta)
    n_unique_records = df_meta["Record"].nunique()

    # Target label counts (Premature = 't' -> target = 1 -> Preterm, 'f' -> 0 -> Term)
    n_term = int((df_meta["target"] == 0).sum())
    n_preterm = int((df_meta["target"] == 1).sum())
    n_missing_labels = int(df_meta["target"].isna().sum())

    # Duplicate records check
    n_duplicates = n_meta_rows - n_unique_records

    # Matching checks between RECORDS file and metadata
    set_records_stems = set(record_stems)
    set_meta_stems = set(df_meta["Record"])

    unmatched_in_meta = set_records_stems - set_meta_stems
    unmatched_in_records = set_meta_stems - set_records_stems
    n_unmatched_records = len(unmatched_in_meta) + len(unmatched_in_records)

    # 8. Print complete verification report with exact required field names
    print("=" * 75)
    print("               TPEHGDB METADATA VERIFICATION REPORT               ")
    print("=" * 75)
    print(f"Number of metadata rows:       {n_meta_rows}")
    print(f"Number of unique records:     {n_unique_records}")
    print(f"Number of Term records:       {n_term}")
    print(f"Number of Preterm records:    {n_preterm}")
    print(f"Number of missing labels:     {n_missing_labels}")
    print(f"Number of duplicate records:  {n_duplicates}")
    print(f"Number of unmatched records:  {n_unmatched_records}")
    print("=" * 75)

    # 9. Print the first 10 rows of parsed metadata
    print("\nFirst 10 Rows of Parsed Metadata:")
    print("-" * 75)
    display_cols = ["Record", "Gestation", "Rec. time", "Group", "Premature", "Early", "target", "label"]
    print(df_meta[display_cols].head(10).to_string(index=False))
    print("-" * 75)

    # 10. Print metadata entry corresponding to the FIRST record loaded in previous step
    first_record_entry = data_loader.get_first_record_name()
    print(f"\nMetadata Entry for FIRST Record ('{first_record_entry}'):")
    print("-" * 75)
    first_record_dict = parser.get_label(first_record_entry)
    for key, val in first_record_dict.items():
        print(f"  {key:<12}: {val}")
    print("-" * 75)

    # 11, 12. Test get_label() validation for unknown record
    print("\nTesting get_label() Validation:")
    try:
        get_label("non_existent_record_9999", smr_filepath=smr_path)
    except KeyError as e:
        print(f"  [PASS] Successfully caught invalid record error: {e}")

    # 15. Save final parsed metadata table to outputs/features/tpehgdb_metadata.csv
    df_meta.to_csv(csv_output_path, index=False)
    print(f"\n[SUCCESS] Derived metadata CSV saved to: {csv_output_path.resolve()}")


if __name__ == "__main__":
    verify_metadata()
