"""Visualization module for training curves, ROC/PR curves, and model comparison.

Adheres to Single Responsibility Principle (SRP):
Dedicated exclusively to rendering publication-quality diagnostic charts.
"""

import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve

from src.training.evaluator import EvaluationResult
from src.training.trainer import ExperimentRunResult

logger = logging.getLogger(__name__)


class TrainingVisualizer:
    """Generates comparative visualizations for neural network training and evaluation."""

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def _grid(self, n: int, cell_w: float = 5.5, cell_h: float = 4.2):
        """Creates a 2-row grid (Baseline row / Regularized row) holding n subplots."""
        n_cols = int(np.ceil(n / 2)) if n > 1 else 1
        n_rows = 2 if n > 1 else 1
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(cell_w * n_cols, cell_h * n_rows), squeeze=False)
        flat = axes.ravel()
        for ax in flat[n:]:
            ax.set_visible(False)
        return fig, flat

    def _plot_history_grid(self, results: List[ExperimentRunResult], metric: str, title: str, ylabel: str, filename: str) -> Path:
        """One subplot per model: train vs validation curve, best val-loss epoch marked."""
        fig, axes = self._grid(len(results))
        for ax, res in zip(axes, results):
            epochs = range(1, len(res.history[metric]) + 1)
            ax.plot(epochs, res.history[metric], label="Train (SMOTE)", color="#1f77b4", linestyle="--", marker="o", ms=3)
            ax.plot(epochs, res.history[f"val_{metric}"], label="Validation", color="#d62728", linewidth=2.0, marker="o", ms=3)
            ax.axvline(res.best_val_loss_epoch, color="gray", linestyle=":", lw=1.2,
                       label=f"Min val loss (epoch {res.best_val_loss_epoch})")
            ax.set_title(res.model_name, fontsize=11, weight="bold")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend(fontsize=8, loc="best")
        fig.suptitle(title, fontsize=14, weight="bold")
        plt.tight_layout()
        save_path = self._output_dir / filename
        fig.savefig(save_path, dpi=200)
        plt.close(fig)
        logger.info("Saved %s to %s", title, save_path)
        return save_path

    def plot_learning_curves(self, results: List[ExperimentRunResult]) -> Path:
        """Plots training and validation loss trajectories, one panel per model."""
        return self._plot_history_grid(
            results, "loss", "Training vs Validation Loss (Binary Cross-Entropy)", "Loss",
            "learning_curves_comparison.png"
        )

    def plot_metric_curves(self, results: List[ExperimentRunResult]) -> Path:
        """Plots training and validation PR-AUC trajectories, one panel per model."""
        return self._plot_history_grid(
            results, "pr_auc", "Training vs Validation PR-AUC", "PR-AUC",
            "learning_curves_pr_auc.png"
        )

    def plot_roc_and_pr_curves(
        self,
        y_true: np.ndarray,
        results: List[ExperimentRunResult]
    ) -> Path:
        """Plots Receiver Operating Characteristic (ROC) and Precision-Recall (PR) curves."""
        fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(16, 6))

        palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

        for idx, res in enumerate(results):
            color = palette[idx % len(palette)]
            y_pred = res.test_predictions

            # ROC Curve
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            ax_roc.plot(fpr, tpr, color=color, lw=2, label=f"{res.model_name} (AUC={res.test_metrics.roc_auc:.4f})")

            # PR Curve
            prec, rec, _ = precision_recall_curve(y_true, y_pred)
            ax_pr.plot(rec, prec, color=color, lw=2, label=f"{res.model_name} (AP={res.test_metrics.pr_auc:.4f})")

        # Baseline reference diagonals
        ax_roc.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Guess (0.50)")
        ax_roc.set_title("Receiver Operating Characteristic (ROC) Curves", fontsize=13, weight="bold")
        ax_roc.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
        ax_roc.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11)
        ax_roc.legend(loc="lower right", fontsize=8)
        ax_roc.grid(True, linestyle=":", alpha=0.6)

        # Baseline for PR curve is the prevalence of positive class
        baseline_pr = float(np.mean(y_true))
        ax_pr.plot([0, 1], [baseline_pr, baseline_pr], "k--", lw=1.2, label=f"No-skill Baseline ({baseline_pr:.4f})")
        ax_pr.set_title("Precision-Recall (PR) Curves [Primary Imbalance Metric]", fontsize=13, weight="bold")
        ax_pr.set_xlabel("Recall (Coverage)", fontsize=11)
        ax_pr.set_ylabel("Precision (Purity)", fontsize=11)
        ax_pr.legend(loc="lower left", fontsize=8)
        ax_pr.grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()
        save_path = self._output_dir / "roc_and_pr_curves.png"
        fig.savefig(save_path, dpi=200)
        plt.close(fig)
        logger.info("Saved ROC and PR curves to %s", save_path)
        return save_path

    def plot_confusion_matrices(self, results: List[ExperimentRunResult], tuned: bool = False) -> Path:
        """Renders a grid of test-set confusion matrices at threshold 0.5 or at the val-tuned threshold."""
        fig, axes = self._grid(len(results), cell_w=4.8, cell_h=4.2)

        for ax, res in zip(axes, results):
            metrics = res.test_metrics_tuned if tuned else res.test_metrics
            cm = metrics.confusion_matrix
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar=False,
                ax=ax,
                xticklabels=["Normal (0)", "Fraud (1)"],
                yticklabels=["Normal (0)", "Fraud (1)"]
            )
            ax.set_title(f"{res.model_name} (thr={metrics.threshold:.3f})\n"
                f"F1={metrics.f1_score:.3f} | P={metrics.precision:.3f} | R={metrics.recall:.3f}", fontsize=10, weight="bold")
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")

        suffix = "tuned threshold (selected on validation)" if tuned else "threshold 0.5"
        fig.suptitle(f"Test-set Confusion Matrices @ {suffix}", fontsize=14, weight="bold")
        plt.tight_layout()
        save_path = self._output_dir / ("confusion_matrices_tuned.png" if tuned else "confusion_matrices.png")
        fig.savefig(save_path, dpi=200)
        plt.close(fig)
        logger.info("Saved confusion matrices to %s", save_path)
        return save_path

    def plot_metrics_comparison_bar(self, results: List[ExperimentRunResult]) -> Path:
        """Plots comparative grouped bar chart for F1, Recall, Precision, PR-AUC."""
        metrics_names = ["Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC (AP)"]
        model_names = [r.model_name for r in results]

        data_matrix = []
        for r in results:
            m = r.test_metrics
            data_matrix.append([m.precision, m.recall, m.f1_score, m.roc_auc, m.pr_auc])

        data_matrix = np.array(data_matrix)  # shape: (n_models, n_metrics)

        fig, ax = plt.subplots(figsize=(14, 6))
        x = np.arange(len(metrics_names))
        total_width = 0.8
        single_width = total_width / len(model_names)

        palette = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"]

        for i, (name, color) in enumerate(zip(model_names, palette)):
            offset = (i - len(model_names) / 2) * single_width + single_width / 2
            bars = ax.bar(x + offset, data_matrix[i], width=single_width, label=name, color=color, edgecolor="black", alpha=0.85)
            for bar in bars:
                height = bar.get_height()
                if height > 0.05:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        height + 0.015,
                        f"{height:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=6,
                        rotation=90
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(metrics_names, fontsize=11, weight="bold")
        ax.set_ylabel("Score [0.0 - 1.0]", fontsize=11)
        ax.set_title("Test Metrics @ threshold 0.5 — All Architectures & Activation Functions", fontsize=13, weight="bold")
        ax.set_ylim(0, 1.2)
        ax.legend(loc="upper center", ncol=3, frameon=True, fontsize=8)
        ax.grid(axis="y", linestyle=":", alpha=0.7)

        plt.tight_layout()
        save_path = self._output_dir / "metrics_comparison_barchart.png"
        fig.savefig(save_path, dpi=200)
        plt.close(fig)
        logger.info("Saved metrics comparison bar chart to %s", save_path)
        return save_path
