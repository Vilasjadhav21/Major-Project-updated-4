"""
Integration tests for Term-Preterm EHG Dataset with Tocogram v1.0.0 (TPEHGT)
and regression verification for existing TPEHGDB v1.0.1 dataset.
"""

from pathlib import Path
import pytest
import wfdb
from src.config import (
    BASE_DIR,
    DATA_TPEHGDB_DIR,
    DATA_TPEHGT_V1_0_0_DIR,
    ProjectConfig,
    TPEHGTDatasetConfig,
    DatasetConfig
)
from src.data_loader import TPEHGDataLoader, load_record, load_tpehgt_record


def test_v1_0_1_intact_and_loader_works():
    """Verify v1.0.1 dataset directory, RECORDS file, and first record loading."""
    assert DATA_TPEHGDB_DIR.exists(), f"v1.0.1 directory not found: {DATA_TPEHGDB_DIR}"
    
    loader_v101 = TPEHGDataLoader(data_dir=DATA_TPEHGDB_DIR)
    records = loader_v101.read_records_list()
    assert len(records) == 300, f"Expected 300 records in v1.0.1, found {len(records)}"
    
    first_record = records[0]
    rec = loader_v101.load_record(first_record)
    assert rec.fs in [20, 20.0, 20.00011] or abs(rec.fs - 20.0) < 0.01
    assert len(rec.sig_name) == 12
    # In v1.0.1 header files, channels are named '1', '2', '3' (referred to as S1, S2, S3 in project docs)
    assert "1" in rec.sig_name or "S1 raw" in rec.sig_name


def test_v1_0_0_directory_and_records():
    """Verify v1.0.0 dataset directory and RECORDS file content."""
    assert DATA_TPEHGT_V1_0_0_DIR.exists(), f"v1.0.0 directory not found: {DATA_TPEHGT_V1_0_0_DIR}"
    
    loader_v100 = TPEHGDataLoader(data_dir=DATA_TPEHGT_V1_0_0_DIR)
    records = loader_v100.read_records_list()
    assert len(records) == 31, f"Expected 31 records in v1.0.0, found {len(records)}"
    
    term_recs = [r for r in records if r.startswith("tpehgt_t")]
    preterm_recs = [r for r in records if r.startswith("tpehgt_p")]
    non_preg_recs = [r for r in records if r.startswith("tpehgt_n")]
    
    assert len(term_recs) == 13, f"Expected 13 term records, found {len(term_recs)}"
    assert len(preterm_recs) == 13, f"Expected 13 preterm records, found {len(preterm_recs)}"
    assert len(non_preg_recs) == 5, f"Expected 5 non-pregnant records, found {len(non_preg_recs)}"


def test_v1_0_0_representative_records_loading():
    """Verify loading representative records from Term, Preterm, and Non-pregnant classes."""
    for record_name in ["tpehgt_t001", "tpehgt_p001", "tpehgt_n001"]:
        rec = load_tpehgt_record(record_name)
        assert rec is not None
        assert rec.p_signal is not None
        assert rec.p_signal.shape[1] == 8
        assert rec.fs == 20.0
        assert rec.sig_name == [
            "EHG1", "EHG1_Butter-4-bi-0.08-5",
            "EHG2", "EHG2_Butter-4-bi-0.08-5",
            "EHG3", "EHG3_Butter-4-bi-0.08-5",
            "TOCO", "TOCO_Butter-4-bi-0.08-5"
        ]


def test_v1_0_0_annotation_loading():
    """Verify loading annotation (.atr) files for v1.0.0 records."""
    loader_v100 = TPEHGDataLoader(data_dir=DATA_TPEHGT_V1_0_0_DIR)
    ann = loader_v100.load_annotation("tpehgt_t001")
    assert ann is not None
    assert len(ann.sample) > 0


def test_dataset_isolation():
    """Verify datasets are stored in separate directories and no v1.0.0 files exist in v1.0.1 folder."""
    v101_files = set(p.name for p in DATA_TPEHGDB_DIR.glob("**/*"))
    v100_files = set(p.name for p in DATA_TPEHGT_V1_0_0_DIR.glob("**/*"))
    
    # Assert no tpehgt files leaked into tpehgdb
    tpehgt_in_v101 = [f for f in v101_files if f.startswith("tpehgt_")]
    assert len(tpehgt_in_v101) == 0, f"Found v1.0.0 files inside v1.0.1 dir: {tpehgt_in_v101}"


def test_combined_features_integrity():
    """Verify combined features dataset has 326 rows, no nulls, and no duplicate IDs."""
    import pandas as pd
    from src.config import FEATURES_DIR
    
    csv_p = FEATURES_DIR / "combined_features.csv"
    assert csv_p.exists(), f"Missing {csv_p}"
    df = pd.read_csv(csv_p)
    assert len(df) == 326
    assert df["record_id"].nunique() == 326
    assert df[["sampen_imf1", "median_freq_imf1_hz", "rms_imf1", "rms_imf2", "hilbert_par_imf1"]].isna().sum().sum() == 0
    assert set(df["label"].unique()).issubset({0, 1})


def test_train_test_split_disjoint():
    """Verify train and test splits have 0 record overlap."""
    import pandas as pd
    from src.config import FEATURES_DIR
    
    train_df = pd.read_csv(FEATURES_DIR / "train_dataset.csv")
    test_df = pd.read_csv(FEATURES_DIR / "test_dataset.csv")
    
    assert len(train_df) == 260
    assert len(test_df) == 66
    overlap = set(train_df["record_id"]).intersection(set(test_df["record_id"]))
    assert len(overlap) == 0, f"Overlapping IDs found: {overlap}"


def test_model_loading_and_inference():
    """Verify final saved model pipeline loads and makes predictions."""
    import joblib
    import numpy as np
    import pandas as pd
    from src.config import MODELS_DIR, FEATURES_DIR
    
    model = joblib.load(MODELS_DIR / "ehg_classifier.joblib")
    assert model is not None
    test_df = pd.read_csv(FEATURES_DIR / "test_dataset.csv")
    X = test_df[["sampen_imf1", "median_freq_imf1_hz", "rms_imf1", "rms_imf2", "hilbert_par_imf1"]].values[:5]
    preds = model.predict(X)
    probs = model.predict_proba(X)
    assert len(preds) == 5
    assert probs.shape == (5, 2)

