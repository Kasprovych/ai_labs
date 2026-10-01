"""Data preprocessing and splitting module implementing Strategy Pattern (GoF)
and Single Responsibility Principle (SOLID).

Scalers and balancing strategies are fitted strictly on the Training set to prevent data leakage.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from typing import Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from imblearn.over_sampling import SMOTE

from src.config.app_config import DataConfig

if TYPE_CHECKING:
    from src.data.loader import DatasetContainer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreprocessedData:
    """Immutable DTO containing train, validation, and test datasets."""
    x_train: np.ndarray
    x_val: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: Tuple[str, ...]

    @property
    def input_dim(self) -> int:
        return self.x_train.shape[1]


class IBalancingStrategy(ABC):
    """Strategy interface for handling class imbalance."""

    @abstractmethod
    def balance(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Balances the dataset features and labels."""
        pass


class NoBalancingStrategy(IBalancingStrategy):
    """Null Object Pattern: does not alter class distribution."""

    def balance(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return x, y


class SmoteBalancingStrategy(IBalancingStrategy):
    """SMOTE (Synthetic Minority Over-sampling Technique) implementation."""

    def __init__(self, sampling_strategy: float = 0.2, random_state: int = 42) -> None:
        self._smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)

    def balance(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        logger.info("Applying SMOTE oversampling to training set...")
        x_res, y_res = self._smote.fit_resample(x, y)
        logger.info(
            "SMOTE applied: original %d samples -> resampled %d samples (positives: %d)",
            len(x), len(x_res), int((y_res == 1).sum())
        )
        return x_res, y_res


class DataPreprocessor:
    """Coordinates train/val/test splitting, feature normalization, and resampling.

    Designed with Dependency Inversion: receives `IBalancingStrategy` via constructor.
    """

    def __init__(
        self,
        config: DataConfig,
        balancing_strategy: IBalancingStrategy | None = None
    ) -> None:
        self._config = config
        self._balancing_strategy = balancing_strategy or SmoteBalancingStrategy(
            sampling_strategy=config.smote_sampling_strategy,
            random_state=config.random_seed
        )
        self._time_scaler = RobustScaler()
        self._amount_scaler = RobustScaler()

    def process(self, df: pd.DataFrame) -> PreprocessedData:
        """Executes full preprocessing pipeline:
        1. Train / Val / Test Stratified Split
        2. Feature Scaling (Fitting ONLY on Train to prevent Data Leakage)
        3. Class Balancing via Strategy (Applied ONLY to Train)
        """
        logger.info("Starting preprocessing pipeline on %d rows...", len(df))

        # Separate features and target
        x_raw = df.drop(columns=[self._config.target_column]).copy()
        y_raw = df[self._config.target_column].to_numpy(dtype=np.int32)
        feature_names = tuple(x_raw.columns)

        # 1. Stratified Split into Train (70%) and Temp (30%)
        # test_size + val_size = 0.15 + 0.15 = 0.30
        temp_ratio = self._config.test_size + self._config.val_size
        x_train_df, x_temp_df, y_train_arr, y_temp_arr = train_test_split(
            x_raw,
            y_raw,
            test_size=temp_ratio,
            random_state=self._config.random_seed,
            stratify=y_raw
        )

        # Split Temp equally into Val (15%) and Test (15%)
        val_test_ratio = self._config.val_size / temp_ratio  # 0.15 / 0.30 = 0.50
        x_val_df, x_test_df, y_val_arr, y_test_arr = train_test_split(
            x_temp_df,
            y_temp_arr,
            test_size=1.0 - val_test_ratio,
            random_state=self._config.random_seed,
            stratify=y_temp_arr
        )

        logger.info(
            "Dataset split completed: Train=%d, Val=%d, Test=%d (Stratified)",
            len(x_train_df), len(x_val_df), len(x_test_df)
        )

        # 2. Scale features (Time and Amount). Fit on Train ONLY!
        for col, scaler in [("Time", self._time_scaler), ("Amount", self._amount_scaler)]:
            if col in x_train_df.columns:
                x_train_df[col] = scaler.fit_transform(x_train_df[[col]])
                x_val_df[col] = scaler.transform(x_val_df[[col]])
                x_test_df[col] = scaler.transform(x_test_df[[col]])

        x_train = x_train_df.to_numpy(dtype=np.float32)
        x_val = x_val_df.to_numpy(dtype=np.float32)
        x_test = x_test_df.to_numpy(dtype=np.float32)

        # 3. Apply Balancing Strategy ONLY on Training data
        x_train_balanced, y_train_balanced = self._balancing_strategy.balance(x_train, y_train_arr)

        return PreprocessedData(
            x_train=x_train_balanced,
            x_val=x_val,
            x_test=x_test,
            y_train=y_train_balanced,
            y_val=y_val_arr,
            y_test=y_test_arr,
            feature_names=feature_names
        )
