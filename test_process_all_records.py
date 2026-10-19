"""
Unit tests for Objective 4: Batch Processing Script (src/process_all_records.py).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.process_all_records import process_all_records


def test_process_all_records_capped(tmp_path):
    """Test process_all_records on first 5 records in fault-tolerant mode."""
    data_dir = Path("data/tpehgdb")
    res = process_all_records(
        data_dir=data_dir,
        output_dir=tmp_path,
        max_records=5
    )

    # Check return dictionary keys
    assert "df_features" in res
    assert "df_log" in res
    assert "total_records" in res
    assert "successful_records" in res
    assert "failed_records" in res
    assert "sanity_pass" in res

    df_feats = res["df_features"]
    df_log = res["df_log"]

    # Verify total attempted matches capped limit
    assert res["total_records"] == 5
    assert len(df_log) == 5

    # Check output CSV existence
    feats_csv = tmp_path / "all_records_features.csv"
    log_csv = tmp_path / "all_records_processing_log.csv"

    assert feats_csv.exists()
    assert log_csv.exists()

    # Verify feature columns
    expected_feat_cols = [
        "record",
        "label",
        "sampen_imf1",
        "median_freq_imf1_hz",
        "rms_imf1",
        "rms_imf2",
        "hilbert_par_imf1"
    ]
    assert list(df_feats.columns) == expected_feat_cols

    # Verify log columns
    expected_log_cols = [
        "record",
        "status",
        "reason",
        "label",
        "sampling_frequency",
        "window_samples",
        "imf_count"
    ]
    assert list(df_log.columns) == expected_log_cols

    # Sanity checks on successful rows
    if not df_feats.empty:
        assert set(df_feats["label"]).issubset({0, 1})
        num_cols = ["sampen_imf1", "median_freq_imf1_hz", "rms_imf1", "rms_imf2", "hilbert_par_imf1"]
        assert not df_feats[num_cols].isna().any().any()
        assert not np.isinf(df_feats[num_cols].values).any()

    # Verify sanity pass
    assert res["sanity_pass"] is True
