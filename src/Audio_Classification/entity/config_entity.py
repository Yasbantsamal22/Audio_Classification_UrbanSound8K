from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    csv_path: str
    preprocess_dir: Path
    audio_dataset_path: Path
    split_dir: Path
    preprocess_data_dir: Path
    train_data_dir: Path
    test_data_dir: Path


@dataclass(frozen=True)
class BasemodelConfig:
    root_dir: Path
    base_model_path: Path


@dataclass(frozen=True)
class ModelTrainingConfig:
    root_dir: Path
    trained_model_path: Path
    checkpoint_model_filepath: Path
    train_data_dir: Path
    test_data_dir: Path
    base_model_path: Path
    params_epochs: int
    params_batch_size: int


@dataclass(frozen=True)
class EvaluationConfig:
    path_of_model: Path
    test_data: Path
    train_data: Path