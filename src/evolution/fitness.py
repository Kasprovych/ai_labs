"""Fitness function: how good is a set of hyperparameters?

A model is trained on the train split and scored on the validation split. The test split is
never seen here, so the hyperparameter search cannot leak test information.
"""

import time
from abc import ABC, abstractmethod
from typing import Dict, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error

from src.evolution.housing_data import HousingSplits
from src.evolution.search_space import HyperparameterValue, SearchSpace


class IFitnessFunction(ABC):
    @abstractmethod
    def evaluate(self, genome: np.ndarray) -> float:
        """Returns fitness of the individual; higher is better."""


class ValidationRmseFitness(IFitnessFunction):
    """Fitness = -RMSE of HistGradientBoostingRegressor on the validation split.

    The sign is flipped because the genetic algorithm maximizes fitness while RMSE is an error.
    Results are cached by decoded hyperparameters: different genomes can decode to the same
    integer values, and elites are re-used, so repeated model fits would be wasted time.
    """

    def __init__(self, search_space: SearchSpace, data: HousingSplits, model_seed: int) -> None:
        self._search_space = search_space
        self._data = data
        self._model_seed = model_seed
        self._cache: Dict[Tuple[HyperparameterValue, ...], float] = {}
        self._model_fits = 0
        self._fit_seconds = 0.0

    @property
    def model_fits(self) -> int:
        """Number of models actually trained (cache hits excluded)."""
        return self._model_fits

    @property
    def fit_seconds(self) -> float:
        return self._fit_seconds

    def build_model(self, hyperparameters: Dict[str, HyperparameterValue]) -> HistGradientBoostingRegressor:
        # early_stopping=False: max_iter is a tuned hyperparameter, so it must be used exactly.
        # **hyperparameters unpacks the dict into keyword arguments (max_iter=..., learning_rate=...).
        return HistGradientBoostingRegressor(random_state=self._model_seed, early_stopping=False, **hyperparameters)

    def validation_rmse(self, genome: np.ndarray) -> float:
        hyperparameters = self._search_space.decode(genome)
        key = tuple(round(value, 10) if isinstance(value, float) else value for value in hyperparameters.values())
        if key not in self._cache:
            started = time.perf_counter()
            model = self.build_model(hyperparameters).fit(self._data.x_train, self._data.y_train)
            self._cache[key] = float(root_mean_squared_error(self._data.y_val, model.predict(self._data.x_val)))
            self._model_fits += 1
            self._fit_seconds += time.perf_counter() - started
        return self._cache[key]

    def evaluate(self, genome: np.ndarray) -> float:
        return -self.validation_rmse(genome)
