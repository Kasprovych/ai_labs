"""PyTorch training orchestration for Laboratory Work #2.

Counterpart of `ModelTrainer` (TensorFlow): same fixed hyperparameters, same validation-tuned
threshold and the same `ExperimentRunResult`, so evaluation and plots are shared between labs.
Unlike Keras' `model.fit`, the training loop is written explicitly (train/eval modes, zero_grad,
backward, step), which is the main practical difference between the two frameworks.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config.app_config import TrainingConfig
from src.data.preprocessor import PreprocessedData
from src.models.torch_builder import FeedForwardNet, ITorchModelBuilder
from src.training.evaluator import MetricsEvaluator
from src.training.results import ExperimentRunResult

logger = logging.getLogger(__name__)


def set_global_seed(seed: int) -> None:
    """Seeds Python, NumPy and PyTorch RNGs and forces deterministic kernels."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


class TorchModelTrainer:
    """Builds, trains for fixed epochs, tunes the threshold on validation and evaluates on test."""

    def __init__(self, config: TrainingConfig) -> None:
        self._config = config
        self._device = torch.device(config.device)

    def _create_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Factory method for optimizer instantiation (same choices as the Keras trainer)."""
        name = self._config.optimizer_name.lower()
        lr = self._config.learning_rate
        if name == "sgd":
            return torch.optim.SGD(model.parameters(), lr=lr)
        if name == "rmsprop":
            return torch.optim.RMSprop(model.parameters(), lr=lr)
        # Keras Adam uses epsilon=1e-7, PyTorch 1e-8: align with Lab 1
        return torch.optim.Adam(model.parameters(), lr=lr, eps=1e-7)

    @staticmethod
    def count_parameters(model: nn.Module) -> Tuple[int, int]:
        """Returns (trainable params, non-trainable state such as BN running mean/var)."""
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        buffers = sum(b.numel() for name, b in model.named_buffers() if "num_batches" not in name)
        return trainable, buffers

    @torch.no_grad()
    def _predict_logits(self, model: FeedForwardNet, x: np.ndarray) -> torch.Tensor:
        model.eval()
        x_tensor = torch.from_numpy(x).to(self._device)
        return torch.cat([model(batch) for batch in torch.split(x_tensor, self._config.batch_size)])

    def predict_proba(self, model: FeedForwardNet, x: np.ndarray) -> np.ndarray:
        return torch.sigmoid(self._predict_logits(model, x)).cpu().numpy()

    def _train_one_epoch(
        self,
        model: FeedForwardNet,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """One pass over the training set; returns mean loss and the batch predictions seen."""
        model.train()
        batch_losses: List[float] = []
        probs: List[np.ndarray] = []
        labels: List[np.ndarray] = []
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(self._device), y_batch.to(self._device)
            optimizer.zero_grad()
            logits = model(x_batch)
            # Like Keras, the reported training loss includes the L2 penalty
            loss = criterion(logits, y_batch) + model.regularization_loss()
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
            probs.append(torch.sigmoid(logits).detach().cpu().numpy())
            labels.append(y_batch.cpu().numpy())
        return float(np.mean(batch_losses)), np.concatenate(labels), np.concatenate(probs)

    def train_and_evaluate(
        self,
        builder: ITorchModelBuilder,
        data: PreprocessedData,
        model_save_dir: Path | None = None
    ) -> ExperimentRunResult:
        """Trains for fixed epochs with fixed optimizer / lr and evaluates on the test set."""
        logger.info("==================================================")
        logger.info("Starting experiment: %s (Activation: %s)", builder.name, builder.activation)
        logger.info("==================================================")

        # Re-seed before every experiment: identical init, dropout masks and batch order
        set_global_seed(self._config.random_seed)

        model = builder.build(input_dim=data.input_dim).to(self._device)
        trainable, buffers = self.count_parameters(model)
        logger.info("Parameters: %d trainable, %d non-trainable (BN running stats)", trainable, buffers)
        logger.debug("%s", model)

        optimizer = self._create_optimizer(model)
        criterion = nn.BCEWithLogitsLoss()

        generator = torch.Generator().manual_seed(self._config.random_seed)
        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(data.x_train), torch.from_numpy(data.y_train.astype(np.float32))),
            batch_size=self._config.batch_size,
            shuffle=True,
            generator=generator
        )
        y_val_tensor = torch.from_numpy(data.y_val.astype(np.float32)).to(self._device)

        logger.info(
            "Fitting %s for %d epochs (batch_size=%d, lr=%.4f, device=%s)...",
            builder.name, self._config.epochs, self._config.batch_size, self._config.learning_rate, self._device
        )

        history: Dict[str, List[float]] = {k: [] for k in ("loss", "auc", "pr_auc", "val_loss", "val_auc", "val_pr_auc")}
        fit_start = time.perf_counter()
        for epoch in range(1, self._config.epochs + 1):
            epoch_start = time.perf_counter()
            train_loss, y_seen, p_seen = self._train_one_epoch(model, train_loader, optimizer, criterion)

            val_logits = self._predict_logits(model, data.x_val)
            with torch.no_grad():
                val_loss = (criterion(val_logits, y_val_tensor) + model.regularization_loss()).item()
            val_proba = torch.sigmoid(val_logits).cpu().numpy()

            history["loss"].append(train_loss)
            history["auc"].append(float(roc_auc_score(y_seen, p_seen)))
            history["pr_auc"].append(float(average_precision_score(y_seen, p_seen)))
            history["val_loss"].append(float(val_loss))
            history["val_auc"].append(float(roc_auc_score(data.y_val, val_proba)))
            history["val_pr_auc"].append(float(average_precision_score(data.y_val, val_proba)))

            logger.info(
                "Epoch %2d/%d - %.1fs - loss: %.4f - pr_auc: %.4f - val_loss: %.4f - val_auc: %.4f - val_pr_auc: %.4f",
                epoch, self._config.epochs, time.perf_counter() - epoch_start, train_loss,
                history["pr_auc"][-1], val_loss, history["val_auc"][-1], history["val_pr_auc"][-1]
            )
        training_seconds = time.perf_counter() - fit_start
        logger.info("Training finished in %.1fs", training_seconds)

        # Tune the decision threshold on the validation set (never on test)
        val_pred_proba = self.predict_proba(model, data.x_val)
        tuned_threshold = MetricsEvaluator.find_best_f1_threshold(data.y_val, val_pred_proba)
        logger.info("F1-optimal threshold on validation set: %.3f", tuned_threshold)

        logger.info("Evaluating %s on independent Test set...", builder.name)
        test_pred_proba = self.predict_proba(model, data.x_test)
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

        if model_save_dir:
            model_save_dir.mkdir(parents=True, exist_ok=True)
            save_path = model_save_dir / f"{builder.name}.pt"
            torch.save(model.state_dict(), save_path)
            logger.info("Model saved to %s", save_path)

        return ExperimentRunResult(
            model_name=builder.name,
            activation=builder.activation,
            history=history,
            test_metrics=test_metrics,
            test_predictions=test_pred_proba.ravel(),
            test_metrics_tuned=test_metrics_tuned,
            tuned_threshold=tuned_threshold,
            training_seconds=training_seconds
        )
