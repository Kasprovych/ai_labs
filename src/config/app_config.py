"""Configuration module for Laboratory Work #1 (TensorFlow Deep Learning).

Contains immutable configuration settings for data, training, paths, and experiments.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


@dataclass(frozen=True)
class DataConfig:
    """Configuration for data loading, scaling, and splitting."""
    dataset_name: str = "creditcard"
    test_size: float = 0.15
    val_size: float = 0.15
    random_seed: int = 42
    target_column: str = "Class"
    # Columns to scale using RobustScaler
    features_to_scale: Tuple[str, ...] = ("Time", "Amount")
    # Subsampling ratio or SMOTE sampling strategy
    smote_sampling_strategy: float = 0.2  # Minority class oversampled to 20% of majority


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters and settings for model training."""
    epochs: int = 15
    batch_size: int = 512
    learning_rate: float = 0.001
    optimizer_name: str = "adam"
    loss_function: str = "binary_crossentropy"
    metrics: Tuple[str, ...] = ("accuracy", "precision", "recall", "auc")


@dataclass(frozen=True)
class PathConfig:
    """Centralized path management for artifacts, logs, datasets, and outputs."""
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent)

    @property
    def artifacts_dir(self) -> Path:
        path = self.project_root / "artifacts"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def plots_dir(self) -> Path:
        path = self.artifacts_dir / "plots"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def data_dir(self) -> Path:
        path = self.project_root / "data"
        path.mkdir(parents=True, exist_ok=True)
        return path


@dataclass(frozen=True)
class ExperimentConfig:
    """Root configuration aggregator (Composition over inheritance)."""
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: PathConfig = field(default_factory=PathConfig)
