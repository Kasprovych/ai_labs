"""Exploratory Data Analysis (EDA) module.

Adheres to Single Responsibility Principle (SRP):
Responsible solely for statistical inspection and visual representation of raw data.
"""

import logging
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# Modern theme settings
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10


class EdaAnalyzer:
    """Performs statistical analysis and generates exploratory plots for dataset inspection."""

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def generate_full_eda_report(self, df: pd.DataFrame, target_col: str = "Class") -> dict:
        """Executes all EDA steps and saves visual plots."""
        logger.info("Executing comprehensive EDA...")
        stats = self.compute_summary_statistics(df, target_col)
        self.plot_class_distribution(df, target_col)
        self.plot_feature_distributions(df, target_col)
        self.plot_correlation_heatmap(df, target_col)
        logger.info("EDA completed. Plots saved to %s", self._output_dir)
        return stats

    def compute_summary_statistics(self, df: pd.DataFrame, target_col: str = "Class") -> dict:
        """Calculates dataset dimensions, missingness, and class frequencies."""
        total_rows = len(df)
        missing_count = int(df.isnull().sum().sum())
        class_counts = df[target_col].value_counts().to_dict()
        class_percentages = (df[target_col].value_counts(normalize=True) * 100).to_dict()

        logger.info("Dataset statistics: Rows=%d, Missing Values=%d", total_rows, missing_count)
        logger.info("Class distribution: %s", {k: f"{v} ({class_percentages.get(k, 0):.3f}%)" for k, v in class_counts.items()})

        return {
            "total_rows": total_rows,
            "feature_count": len(df.columns) - 1,
            "missing_values": missing_count,
            "class_counts": class_counts,
            "class_percentages": class_percentages,
        }

    def plot_class_distribution(self, df: pd.DataFrame, target_col: str = "Class") -> Path:
        """Generates dual-view (linear and log scale) plot of class imbalance."""
        counts = df[target_col].value_counts()
        labels = ["Legitimate (0)", "Fraudulent (1)"]
        colors = ["#2b5c8f", "#d9534f"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # 1. Standard count bar plot
        bars = ax1.bar(labels, counts.values, color=colors, edgecolor="black", alpha=0.85)
        ax1.set_title("Class Distribution (Absolute Counts)", fontsize=13, weight="bold")
        ax1.set_ylabel("Number of Transactions")
        for bar, count in zip(bars, counts.values):
            pct = count / len(df) * 100
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + len(df) * 0.01,
                f"{count:,}\n({pct:.2f}%)",
                ha="center",
                va="bottom",
                fontweight="bold"
            )

        # 2. Log-scale plot to clearly reveal minority class
        ax2.bar(labels, counts.values, color=colors, edgecolor="black", alpha=0.85)
        ax2.set_yscale("log")
        ax2.set_title("Class Distribution (Log Scale)", fontsize=13, weight="bold")
        ax2.set_ylabel("Log(Count)")
        for bar, count in zip(bars, counts.values):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                count * 1.2,
                f"{count:,}",
                ha="center",
                va="bottom",
                fontweight="bold"
            )

        plt.tight_layout()
        save_path = self._output_dir / "eda_class_imbalance.png"
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        return save_path

    def plot_feature_distributions(
        self,
        df: pd.DataFrame,
        target_col: str = "Class",
        features: Optional[List[str]] = None
    ) -> Path:
        """Plots kernel density estimates (KDE) for the most discriminative features."""
        if features is None:
            # V14, V17, V12 have strong negative correlation; V4, V11 have strong positive correlation
            features = [f for f in ["V14", "V17", "V12", "V4", "Amount"] if f in df.columns]

        n_feats = len(features)
        fig, axes = plt.subplots(1, n_feats, figsize=(4 * n_feats, 4))
        if n_feats == 1:
            axes = [axes]

        for ax, feat in zip(axes, features):
            sns.kdeplot(
                data=df[df[target_col] == 0][feat],
                ax=ax,
                label="Legitimate",
                color="#2b5c8f",
                fill=True,
                alpha=0.3
            )
            sns.kdeplot(
                data=df[df[target_col] == 1][feat],
                ax=ax,
                label="Fraud",
                color="#d9534f",
                fill=True,
                alpha=0.3
            )
            ax.set_title(f"Distribution: {feat}", weight="bold")
            ax.set_xlabel(feat)
            ax.legend()

        plt.tight_layout()
        save_path = self._output_dir / "eda_feature_distributions.png"
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        return save_path

    def plot_correlation_heatmap(self, df: pd.DataFrame, target_col: str = "Class") -> Path:
        """Computes and visualizes correlation matrix highlighting correlation with target."""
        corr = df.corr()

        # Target correlation bar plot
        target_corr = corr[target_col].drop(target_col).sort_values()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [1.5, 1]})

        # Subplot 1: Full correlation heatmap
        sns.heatmap(
            corr,
            ax=ax1,
            cmap="vlag",
            center=0,
            cbar_kws={"shrink": 0.8},
            xticklabels=False,
            yticklabels=False
        )
        ax1.set_title("Full Feature Correlation Heatmap (30x30)", weight="bold", fontsize=12)

        # Subplot 2: Top correlated features with target
        top_corr = pd.concat([target_corr.head(6), target_corr.tail(6)])
        colors = ["#d9534f" if v > 0 else "#2b5c8f" for v in top_corr.values]
        top_corr.plot(kind="barh", ax=ax2, color=colors, edgecolor="black", alpha=0.85)
        ax2.set_title("Top Positively & Negatively Correlated Features with Class", weight="bold", fontsize=12)
        ax2.set_xlabel("Pearson Correlation Coefficient")

        plt.tight_layout()
        save_path = self._output_dir / "eda_correlation_analysis.png"
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        return save_path
