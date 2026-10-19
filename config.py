"""
Configuration settings for Objective 4: Rapid Triage (1-Minute Algorithm).

Centralized configuration for dataset paths, PhysioNet TPEHGDB v1.0.1 facts,
rapid triage windowing, EMD decomposition targets, and model definitions.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

# Project Base Directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_TPEHGDB_DIR = DATA_DIR / "tpehgdb"
DATA_TPEHGT_V1_0_0_DIR = DATA_DIR / "tpehgt_v1.0.0"
OUTPUT_DIR = BASE_DIR / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"
IMFS_DIR = OUTPUT_DIR / "imfs"
FEATURES_DIR = OUTPUT_DIR / "features"
MODELS_DIR = OUTPUT_DIR / "models"


@dataclass
class TPEHGTDatasetConfig:
    """PhysioNet Term-Preterm EHG Dataset with Tocogram v1.0.0 Specifications."""
    dataset_name: str = "PhysioNet Term-Preterm EHG Dataset with Tocogram v1.0.0"
    num_records: int = 31
    num_term_records: int = 13
    num_preterm_records: int = 13
    num_non_pregnant_records: int = 5
    sampling_frequency: float = 20.0  # Hz
    sampling_rate: float = 20.0       # Alias for pipeline compatibility
    sampling_frequency_tolerance: float = 0.01  # Hz tolerance
    record_duration_minutes: float = 30.0  # minutes per record
    record_duration_seconds: float = 1800.0  # seconds per record
    num_channels: int = 8  # 3 EHG raw, 3 EHG filtered, 1 TOCO raw, 1 TOCO filtered

    # 8 signal channels per record in v1.0.0
    signal_channels: List[str] = field(default_factory=lambda: [
        "EHG1", "EHG1_Butter-4-bi-0.08-5",
        "EHG2", "EHG2_Butter-4-bi-0.08-5",
        "EHG3", "EHG3_Butter-4-bi-0.08-5",
        "TOCO", "TOCO_Butter-4-bi-0.08-5"
    ])

    annotation_extension: str = "atr"


@dataclass
class DatasetConfig:
    """PhysioNet TPEHGDB v1.0.1 Dataset Specifications."""
    dataset_name: str = "PhysioNet TPEHGDB v1.0.1"
    num_records: int = 300
    num_pregnancies: int = 300
    sampling_frequency: float = 20.0  # Hz
    sampling_rate: float = 20.0       # Alias for pipeline compatibility
    sampling_frequency_tolerance: float = 0.01  # Hz tolerance for hardware/header representations (e.g. 20.00011 Hz)
    record_duration_minutes: float = 30.0  # minutes per record
    record_duration_seconds: float = 1800.0  # seconds per record
    num_ehg_channels: int = 3  # S1, S2, S3

    # Signal filtering parameters
    bandpass_low: float = 0.1
    bandpass_high: float = 4.0
    filter_order: int = 4
    window_length_sec: float = 60.0
    window_overlap_sec: float = 0.0

    # 12 signal channels per record
    signal_channels: List[str] = field(default_factory=lambda: [
        "S1 raw", "S1 0.08-4 Hz", "S1 0.3-3 Hz", "S1 0.3-4 Hz",
        "S2 raw", "S2 0.08-4 Hz", "S2 0.3-3 Hz", "S2 0.3-4 Hz",
        "S3 raw", "S3 0.08-4 Hz", "S3 0.3-3 Hz", "S3 0.3-4 Hz"
    ])

    # Filtered signals boundary trimming rule
    trim_boundary_seconds: float = 180.0  # Ignore first and last 180 seconds for filtered signals

    # Metadata file
    metadata_filename: str = "tpehgdb.smr"


@dataclass
class TriagePipelineConfig:
    """Target Pipeline Parameters for Rapid Triage (1-Minute Algorithm)."""
    triage_window_seconds: float = 60.0  # 1-minute EHG recording
    target_imf: int = 1                  # IMF-1 component
    num_top_features: int = 5            # Top 5 EHG features
    classifier_model: str = "Fine Gaussian Kernel SVM"
    model_type: str = "svm"
    random_state: int = 42
    n_splits: int = 5
    svm_c: float = 1.0
    rf_n_estimators: int = 100
    rf_max_depth: Optional[int] = None


@dataclass
class EMDConfig:
    """EMD Decomposition Parameters."""
    sampling_frequency: float = 20.0
    max_imfs: int = 10


@dataclass
class ProjectConfig:
    """Master configuration bundling dataset and pipeline settings."""
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    triage: TriagePipelineConfig = field(default_factory=TriagePipelineConfig)
    emd: EMDConfig = field(default_factory=EMDConfig)

    data_dir: Path = DATA_DIR
    tpehgdb_dir: Path = DATA_TPEHGDB_DIR
    tpehgt_v1_0_0_dir: Path = DATA_TPEHGT_V1_0_0_DIR
    tpehgt_dataset: TPEHGTDatasetConfig = field(default_factory=TPEHGTDatasetConfig)
    plots_dir: Path = PLOTS_DIR
    imfs_dir: Path = IMFS_DIR
    features_dir: Path = FEATURES_DIR
    models_dir: Path = MODELS_DIR

    @property
    def signal(self) -> DatasetConfig:
        return self.dataset


# Legacy & pipeline compatibility aliases
PipelineConfig = ProjectConfig
SignalConfig = DatasetConfig
FeatureConfig = TriagePipelineConfig
ModelConfig = TriagePipelineConfig


def ensure_directories():
    """Create all required research project directories if they do not exist."""
    directories = [
        DATA_DIR,
        DATA_TPEHGDB_DIR,
        DATA_TPEHGT_V1_0_0_DIR,
        OUTPUT_DIR,
        PLOTS_DIR,
        IMFS_DIR,
        FEATURES_DIR,
        MODELS_DIR,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_directories()
    config = ProjectConfig()
    print(f"Loaded config for {config.dataset.dataset_name}")
    print(f"Sampling Frequency: {config.dataset.sampling_frequency} Hz")
    print(f"Signal Channels ({len(config.dataset.signal_channels)}): {config.dataset.signal_channels}")
    print(f"Trim Rule: Ignore first and last {config.dataset.trim_boundary_seconds}s for filtered signals")
    print(f"Target Classifier: {config.triage.classifier_model}")
