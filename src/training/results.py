"""Framework-agnostic experiment result DTO shared by the TensorFlow and PyTorch trainers.

Kept free of any deep learning framework import so that evaluation and visualization code
can be reused by both laboratory works.
"""

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from src.training.evaluator import EvaluationResult


@dataclass(frozen=True)
class ExperimentRunResult:
    """Immutable result bundle from a single model experiment."""
    model_name: str
    activation: str
    history: Dict[str, List[float]]
    test_metrics: EvaluationResult
    test_predictions: np.ndarray
    # Test metrics at the F1-optimal threshold tuned on the validation set
    test_metrics_tuned: EvaluationResult
    tuned_threshold: float
    # Wall-clock time of model.fit / the training loop (NaN when not measured)
    training_seconds: float = float("nan")

    @property
    def best_val_loss_epoch(self) -> int:
        """1-based epoch with the lowest validation loss (later epochs = overfitting)."""
        return int(np.argmin(self.history["val_loss"])) + 1

    @property
    def overfitting_summary(self) -> Dict[str, Any]:
        """Train/val loss diagnostics used to judge over- and underfitting."""
        val_loss = self.history["val_loss"]
        best = self.best_val_loss_epoch
        return {
            "Model": self.model_name,
            "Final train loss": f"{self.history['loss'][-1]:.4f}",
            "Final val loss": f"{val_loss[-1]:.4f}",
            "Best val loss": f"{min(val_loss):.4f}",
            "Best epoch": best,
            "Val loss rise after best": f"{(val_loss[-1] - min(val_loss)) / min(val_loss) * 100:+.1f}%",
            "Final val PR-AUC": f"{self.history['val_pr_auc'][-1]:.4f}",
        }
