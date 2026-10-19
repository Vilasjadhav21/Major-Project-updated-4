"""
Metadata module for parsing PhysioNet TPEHGDB v1.0.1 clinical attributes
and generating standardized Term/Preterm binary labels.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd


class TPEHGMetadataParser:
    """Parser and query manager for TPEHGDB clinical metadata (tpehgdb.smr)."""

    def __init__(self, smr_filepath: Union[str, Path] = "data/tpehgdb/tpehgdb.smr"):
        self.smr_filepath = Path(smr_filepath)
        self._metadata_df: Optional[pd.DataFrame] = None

    @staticmethod
    def parse_header_comments(comments: List[str]) -> Dict[str, Any]:
        """Parse WFDB header comment lines for clinical attributes."""
        res: Dict[str, Any] = {}
        for c in comments:
            line = c.strip().lstrip("#").strip()
            if ":" in line:
                key, val = [x.strip() for x in line.split(":", 1)]
            elif " " in line:
                parts = line.split(None, 1)
                key, val = parts[0], parts[1]
            else:
                continue
            key_lower = key.lower()
            if "gestation" in key_lower and "rec" in key_lower:
                res["gestation_rec_weeks"] = float(val.replace("weeks", "").strip())
            elif "gestation" in key_lower and "del" not in key_lower:
                res["gestation_rec_weeks"] = float(val.replace("weeks", "").strip())
            elif "delivery" in key_lower or ("gestation" in key_lower and "del" in key_lower):
                res["gestation_del_weeks"] = float(val.replace("weeks", "").strip())
            elif "age" in key_lower:
                try:
                    res["maternal_age"] = int(val)
                except ValueError:
                    res["maternal_age"] = float(val)
            elif "parity" in key_lower:
                res["parity"] = int(val)
        if "gestation_del_weeks" in res:
            res["is_preterm"] = bool(res["gestation_del_weeks"] < 37.0)
        return res

    def load_metadata(self) -> pd.DataFrame:
        """
        Parse tpehgdb.smr into a pandas DataFrame.

        Reads the pipe-delimited format (|), strips whitespace from column headers and values,
        preserves original fields, and creates a binary target column (1=Preterm, 0=Term).
        """
        if not self.smr_filepath.exists():
            fallback = Path("data/tpehgdb/tpehgdb.smr")
            if fallback.exists():
                self.smr_filepath = fallback
            else:
                raise FileNotFoundError(f"Metadata file not found at {self.smr_filepath}")

        # Pipe-delimited SMR parsing, skipping separator line at index 1
        df = pd.read_csv(self.smr_filepath, sep="|", skiprows=[1], engine="python")

        # Strip whitespace from column headers
        df.columns = [col.strip() for col in df.columns]

        # Strip whitespace from string values
        for col in df.columns:
            df[col] = df[col].apply(lambda x: str(x).strip() if pd.notna(x) else x)

        # Convert numeric columns
        df["Gestation"] = pd.to_numeric(df["Gestation"], errors="coerce")
        df["Rec. time"] = pd.to_numeric(df["Rec. time"], errors="coerce")

        # Standardized binary target column: Premature = 't' -> target = 1, 'f' -> target = 0
        def map_target(val: str) -> Optional[int]:
            val_clean = str(val).strip().lower()
            if val_clean == 't':
                return 1
            elif val_clean == 'f':
                return 0
            return None

        df["target"] = df["Premature"].apply(map_target)
        df["label"] = df["target"].map({1: "Preterm", 0: "Term"})

        # Standardized column aliases for convenience
        df["gestation"] = df["Gestation"]
        df["rec_time"] = df["Rec. time"]
        df["group"] = df["Group"]
        df["early"] = df["Early"]
        df["record"] = df["Record"]

        self._metadata_df = df
        return df

    def get_label(self, record_name: str) -> Dict[str, Union[str, int, float]]:
        """
        Get metadata entry dictionary for a specific record.

        Args:
            record_name: Name of record (e.g., 'tpehg1007' or 'tpehgdb/tpehg1007').

        Returns:
            Dictionary containing record, target, label, gestation, rec_time, group, early.

        Raises:
            KeyError: If record_name is not found in metadata.
        """
        if self._metadata_df is None:
            self.load_metadata()

        df = self._metadata_df

        # Clean record name to stem (e.g., 'tpehgdb/tpehg1007' -> 'tpehg1007')
        clean_record = record_name.strip()
        if "/" in clean_record or "\\" in clean_record:
            clean_record = Path(clean_record).name
        clean_record = clean_record.replace(".hea", "").replace(".dat", "").strip()

        match = df[df["Record"] == clean_record]
        if match.empty:
            raise KeyError(f"Record '{record_name}' (normalized: '{clean_record}') not found in metadata.")

        row = match.iloc[0]
        return {
            "record": str(row["Record"]),
            "target": int(row["target"]) if pd.notna(row["target"]) else None,
            "label": str(row["label"]),
            "gestation": float(row["Gestation"]) if pd.notna(row["Gestation"]) else None,
            "rec_time": float(row["Rec. time"]) if pd.notna(row["Rec. time"]) else None,
            "group": str(row["Group"]),
            "early": str(row["Early"])
        }


def load_metadata(smr_filepath: Union[str, Path] = "data/tpehgdb/tpehgdb.smr") -> pd.DataFrame:
    """Convenience function to parse and load metadata DataFrame."""
    parser = TPEHGMetadataParser(smr_filepath=smr_filepath)
    return parser.load_metadata()


def get_label(record_name: str, smr_filepath: Union[str, Path] = "data/tpehgdb/tpehgdb.smr") -> Dict[str, Union[str, int, float]]:
    """Convenience function to get metadata label dictionary for record_name."""
    parser = TPEHGMetadataParser(smr_filepath=smr_filepath)
    return parser.get_label(record_name)
