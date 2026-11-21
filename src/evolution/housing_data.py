"""California Housing dataset: loading and train / validation / test split."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from src.config.lab3_config import HousingDataConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HousingSplits:
    """Feature matrices and targets (median house value, in $100,000) of the three splits."""
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    feature_names: Tuple[str, ...]

    @property
    def x_train_val(self) -> np.ndarray:
        return np.vstack([self.x_train, self.x_val])

    @property
    def y_train_val(self) -> np.ndarray:
        return np.concatenate([self.y_train, self.y_val])


class HousingDataLoader:
    """Downloads (and caches) California Housing and splits it 60/20/20.

    No scaling is applied: tree-based models are invariant to monotonic feature transformations,
    so there is nothing to fit on the data before the split and no leakage risk.
    """

    def __init__(self, config: HousingDataConfig, cache_dir: Path) -> None:
        self._config = config
        self._cache_dir = cache_dir

    def load(self) -> HousingSplits:
        dataset = fetch_california_housing(data_home=str(self._cache_dir))
        features, target = dataset.data, dataset.target
        x_rest, x_test, y_rest, y_test = train_test_split(
            features, target, test_size=self._config.test_size, random_state=self._config.random_seed)
        x_train, x_val, y_train, y_val = train_test_split(
            x_rest, y_rest, test_size=self._config.val_size_of_rest, random_state=self._config.random_seed)
        logger.info("California Housing: %d rows, %d features -> train=%d, val=%d, test=%d",
                    len(features), features.shape[1], len(x_train), len(x_val), len(x_test))
        return HousingSplits(x_train, y_train, x_val, y_val, x_test, y_test, tuple(dataset.feature_names))
