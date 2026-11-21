"""Final, one-time evaluation of selected hyperparameters on the held-out test split."""

from dataclasses import dataclass
from typing import Any, Dict

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from src.evolution.housing_data import HousingSplits
from src.evolution.search_space import HyperparameterValue


@dataclass(frozen=True)
class FinalEvaluation:
    """Quality of one hyperparameter set: validation (used for the search) and test (never used before)."""
    label: str
    hyperparameters: Dict[str, HyperparameterValue]
    val_rmse: float
    test_rmse: float
    test_mae: float
    test_r2: float

    def to_row(self) -> Dict[str, Any]:
        return {"Configuration": self.label, "Val RMSE": round(self.val_rmse, 4),
                "Test RMSE": round(self.test_rmse, 4), "Test MAE": round(self.test_mae, 4),
                "Test R2": round(self.test_r2, 4)}


class FinalEvaluator:
    """Trains on train (-> validation score) and on train+validation (-> test score).

    Refitting on train+validation before the test is standard practice: once the hyperparameters
    are fixed, the validation rows are no longer needed for selection and can improve the model.
    """

    def __init__(self, data: HousingSplits, model_seed: int) -> None:
        self._data = data
        self._model_seed = model_seed

    def evaluate(self, label: str, hyperparameters: Dict[str, HyperparameterValue]) -> FinalEvaluation:
        data = self._data
        val_model = self._build(hyperparameters).fit(data.x_train, data.y_train)
        val_rmse = root_mean_squared_error(data.y_val, val_model.predict(data.x_val))
        final_model = self._build(hyperparameters).fit(data.x_train_val, data.y_train_val)
        predictions = final_model.predict(data.x_test)
        return FinalEvaluation(
            label=label, hyperparameters=dict(hyperparameters), val_rmse=float(val_rmse),
            test_rmse=float(root_mean_squared_error(data.y_test, predictions)),
            test_mae=float(mean_absolute_error(data.y_test, predictions)),
            test_r2=float(r2_score(data.y_test, predictions)))

    def _build(self, hyperparameters: Dict[str, HyperparameterValue]) -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(random_state=self._model_seed, early_stopping=False, **hyperparameters)
