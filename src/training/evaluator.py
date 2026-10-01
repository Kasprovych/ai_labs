"""Evaluation metrics computation module for imbalanced classification.

Adheres to Single Responsibility Principle (SRP):
Responsible solely for calculating performance metrics and generating summary reports.
"""

from dataclasses import dataclass
import logging
from typing import Dict, Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvaluationResult:
    """Immutable DTO containing all classification metrics for an evaluated model."""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    pr_auc: float
    confusion_matrix: np.ndarray
    tp: int
    fp: int
    tn: int
    fn: int
    threshold: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        """Converts result to serializable dictionary for tabular reporting."""
        return {
            "Model": self.model_name,
            "Threshold": f"{self.threshold:.3f}",
            "Accuracy": f"{self.accuracy:.4f}",
            "Precision": f"{self.precision:.4f}",
            "Recall": f"{self.recall:.4f}",
            "F1-Score": f"{self.f1_score:.4f}",
            "ROC-AUC": f"{self.roc_auc:.4f}",
            "PR-AUC (AP)": f"{self.pr_auc:.4f}",
            "TP": self.tp,
            "FP": self.fp,
            "TN": self.tn,
            "FN": self.fn,
        }


class MetricsEvaluator:
    """Evaluates classification models with specific focus on imbalanced distributions."""

    @staticmethod
    def find_best_f1_threshold(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Returns the decision threshold that maximizes F1 on the given (validation) data.

        SMOTE shifts the training class prior from 0.17% to ~17% fraud, so the network's
        probabilities are miscalibrated for the real distribution and 0.5 is rarely optimal.
        The threshold must be tuned on validation data only, never on the test set.
        """
        y_true = np.asarray(y_true).ravel()
        y_pred_proba = np.asarray(y_pred_proba).ravel()
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        # precision/recall have one more element than thresholds; drop the final (recall=0) point
        f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
        return float(thresholds[int(np.argmax(f1))])

    @staticmethod
    def evaluate(
        model_name: str,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        threshold: float = 0.5
    ) -> EvaluationResult:
        """Computes comprehensive set of metrics including ROC-AUC and PR-AUC.

        Args:
            model_name: Identifier of the model.
            y_true: Ground truth binary labels (0 or 1).
            y_pred_proba: Model output probabilities (continuous [0, 1]).
            threshold: Decision boundary for binary classification.

        Returns:
            Immutable EvaluationResult object.
        """
        # Ensure flat arrays
        y_true = np.asarray(y_true).ravel()
        y_pred_proba = np.asarray(y_pred_proba).ravel()
        y_pred_binary = (y_pred_proba >= threshold).astype(int)

        # Basic counts
        cm = confusion_matrix(y_true, y_pred_binary)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn, fp, fn, tp = 0, 0, 0, 0

        acc = float(accuracy_score(y_true, y_pred_binary))
        prec = float(precision_score(y_true, y_pred_binary, zero_division=0))
        rec = float(recall_score(y_true, y_pred_binary, zero_division=0))
        f1 = float(f1_score(y_true, y_pred_binary, zero_division=0))

        # Handle edge cases in probability metrics
        try:
            roc = float(roc_auc_score(y_true, y_pred_proba))
        except ValueError:
            roc = 0.5

        try:
            pr = float(average_precision_score(y_true, y_pred_proba))
        except ValueError:
            pr = 0.0

        logger.info(
            "[%s] Evaluation (thr=%.3f) -> F1: %.4f, Recall: %.4f, Precision: %.4f, ROC-AUC: %.4f, PR-AUC: %.4f (TP=%d, FP=%d, FN=%d)",
            model_name, threshold, f1, rec, prec, roc, pr, tp, fp, fn
        )

        return EvaluationResult(
            model_name=model_name,
            accuracy=acc,
            precision=prec,
            recall=rec,
            f1_score=f1,
            roc_auc=roc,
            pr_auc=pr,
            confusion_matrix=cm,
            tp=int(tp),
            fp=int(fp),
            tn=int(tn),
            fn=int(fn),
            threshold=float(threshold)
        )
