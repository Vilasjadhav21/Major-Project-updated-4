"""
Unit tests for Objective 4: 10-Record Feature Extraction Validation Pipeline.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.validate_10_records import validate_10_records, print_validation_report


def test_validation_10_records_execution(tmp_path):
    """Test full execution of validate_10_records pipeline on 10 records."""
    csv_out = tmp_path / "test_validation_10_records.csv"
    res = validate_10_records(output_csv_path=csv_out, num_records_to_process=10)

    # Check return keys
    assert "df_results" in res
    assert "successful_records" in res
    assert "failed_records" in res
    assert "report_text" in res

    df = res["df_results"]
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert len(df) == len(res["successful_records"])
    assert len(res["successful_records"]) + len(res["failed_records"]) == 10

    # Verify CSV file existence and exact column names
    assert csv_out.exists()
    df_csv = pd.read_csv(csv_out)
    expected_cols = [
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
    assert list(df_csv.columns) == expected_cols

    # Verify rows and window properties
    assert (df_csv["sampling_frequency"] == 20.0).all()
    assert (df_csv["window_start_sec"] == 180.0).all()
    assert (df_csv["window_end_sec"] == 240.0).all()
    assert (df_csv["n_samples"] == 1200).all()

    # Verify feature values are all finite numbers
    feature_cols = ["sampen_imf1", "median_freq_imf1_hz", "rms_imf1", "rms_imf2", "hilbert_par_imf1"]
    for col in feature_cols:
        assert not df_csv[col].isna().any(), f"NaN values found in {col}"
        assert not np.isinf(df_csv[col]).any(), f"Infinite values found in {col}"

    # Verify labels are Term or Preterm
    valid_labels = {"Term", "Preterm"}
    assert set(df_csv["label"]).issubset(valid_labels)


def test_validation_report_formatting():
    """Verify formatting of print_validation_report."""
    df_dummy = pd.DataFrame([
        {
            "record": "tpehg1007",
            "sampling_frequency": 20.0,
            "window_start_sec": 180.0,
            "window_end_sec": 240.0,
            "n_samples": 1200,
            "sampen_imf1": 1.2345,
            "median_freq_imf1_hz": 0.456,
            "rms_imf1": 0.0123,
            "rms_imf2": 0.0098,
            "hilbert_par_imf1": 3.456,
            "label": "Preterm"
        }
    ])

    report_str = print_validation_report(
        df_results=df_dummy,
        successful_records=["tpehg1007"],
        failed_records=[],
        total_attempted=1
    )

    assert "Total Records Attempted:  1" in report_str
    assert "Successful Records Count: 1" in report_str
    assert "Failed Records Count:     0" in report_str
    assert "Preterm: 1" in report_str
    assert "PASS" in report_str
