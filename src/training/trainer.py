"""Training orchestration module adhering to Dependency Inversion (DIP)
and Single Responsibility Principle (SRP).
"""

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD, RMSprop

from src.config.app_config import TrainingConfig
from src.data.preprocessor import PreprocessedData
from src.models.builder import IModelBuilder
from src.training.evaluator import EvaluationResult, MetricsEvaluator

logger = logging.getLogger(__name__)


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


class ModelTrainer:
    """Orchestrates model compilation, fitting, prediction, and metric collection."""

    def __init__(self, config: TrainingConfig) -> None:
        self._config = config

    def _create_optimizer(self) -> tf.keras.optimizers.Optimizer:
        """Factory method for optimizer instantiation."""
        name = self._config.optimizer_name.lower()
        lr = self._config.learning_rate
        if name == "adam":
            return Adam(learning_rate=lr)
        elif name == "sgd":
            return SGD(learning_rate=lr)
        elif name == "rmsprop":
            return RMSprop(learning_rate=lr)
        else:
            return Adam(learning_rate=lr)

    def train_and_evaluate(
        self,
        builder: IModelBuilder,
        data: PreprocessedData,
        model_save_dir: Path | None = None
    ) -> ExperimentRunResult:
        """Compiles, trains for fixed epochs, and evaluates the given model builder."""
        logger.info("==================================================")
        logger.info("Starting experiment: %s (Activation: %s)", builder.name, builder.activation)
        logger.info("==================================================")

        # 0. Re-seed before every experiment so each model starts from the same RNG state
        #    (identical weight init seeds / dropout masks / batch order) -> reproducible comparison
        tf.keras.utils.set_random_seed(self._config.random_seed)

        # 1. Instantiate network architecture
        model = builder.build(input_dim=data.input_dim)

        # 2. Compile model with fixed optimizer, loss and monitoring metrics
        optimizer = self._create_optimizer()
        model.compile(
            optimizer=optimizer,
            loss=self._config.loss_function,
            metrics=[
                "accuracy",
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
                tf.keras.metrics.AUC(name="auc", curve="ROC"),
                tf.keras.metrics.AUC(name="pr_auc", curve="PR")
            ]
        )

        model.summary(print_fn=lambda x: logger.debug(x))

        # 3. Train model with fixed hyperparameters
        logger.info(
            "Fitting %s for %d epochs (batch_size=%d, lr=%.4f)...",
            builder.name, self._config.epochs, self._config.batch_size, self._config.learning_rate
        )

        history_obj = model.fit(
            data.x_train,
            data.y_train,
            validation_data=(data.x_val, data.y_val),
            epochs=self._config.epochs,
            batch_size=self._config.batch_size,
            shuffle=True,
            verbose=2
        )

        # Extract history as serializable python floats
        history: Dict[str, List[float]] = {
            metric: [float(v) for v in values]
            for metric, values in history_obj.history.items()
        }

        # 4. Tune the decision threshold on the validation set (never on test)
        val_pred_proba = model.predict(data.x_val, batch_size=self._config.batch_size, verbose=0)
        tuned_threshold = MetricsEvaluator.find_best_f1_threshold(data.y_val, val_pred_proba)
        logger.info("F1-optimal threshold on validation set: %.3f", tuned_threshold)

        # 5. Predict on unseen Test set at default and tuned thresholds
        logger.info("Evaluating %s on independent Test set...", builder.name)
        test_pred_proba = model.predict(data.x_test, batch_size=self._config.batch_size, verbose=0)
        test_metrics = MetricsEvaluator.evaluate(
            model_name=builder.name,
            y_true=data.y_test,
            y_pred_proba=test_pred_proba,
            threshold=self._config.default_threshold
        )
        test_metrics_tuned = MetricsEvaluator.evaluate(
            model_name=builder.name,
            y_true=data.y_test,
            y_pred_proba=test_pred_proba,
            threshold=tuned_threshold
        )

        # 6. Optionally persist trained weights
        if model_save_dir:
            model_save_dir.mkdir(parents=True, exist_ok=True)
            save_path = model_save_dir / f"{builder.name}.keras"
            model.save(save_path)
            logger.info("Model saved to %s", save_path)

        return ExperimentRunResult(
            model_name=builder.name,
            activation=builder.activation,
            history=history,
            test_metrics=test_metrics,
            test_predictions=test_pred_proba.ravel(),
            test_metrics_tuned=test_metrics_tuned,
            tuned_threshold=tuned_threshold
        )
