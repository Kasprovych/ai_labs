"""Main orchestrator for Laboratory Work #1 (TensorFlow Deep Learning).

Wires application dependencies (IoC / Dependency Injection) and coordinates the execution flow.
"""

import logging
import os
import sys
from pathlib import Path
from typing import List

import pandas as pd

# Suppress TF C++ verbose logs before importing TF
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from src.config.app_config import ExperimentConfig
from src.data.loader import CreditCardDataLoader
from src.data.preprocessor import DataPreprocessor, SmoteBalancingStrategy
from src.eda.analyzer import EdaAnalyzer
from src.models.builder import BaselineModelBuilder, RegularizedModelBuilder, IModelBuilder
from src.training.trainer import ModelTrainer, ExperimentRunResult
from src.visualization.plots import TrainingVisualizer


def setup_logging() -> None:
    """Configures structured console logging akin to SLF4J / Logback."""
    log_format = "%(asctime)s [%(levelname)-7s] %(name)-28s: %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def print_banner(title: str) -> None:
    """Prints a styled CLI section separator."""
    border = "=" * 80
    print(f"\n{border}\n >>> {title.upper()}\n{border}\n")


def main() -> None:
    """Main application execution pipeline."""
    setup_logging()
    logger = logging.getLogger("Application")
    logger.info("Initializing Laboratory Work #1 Deep Learning Pipeline...")

    # 1. Configuration (Inversion of Control container setup)
    config = ExperimentConfig()
    logger.info("Configuration loaded: Epochs=%d, BatchSize=%d, LearningRate=%.4f, Seed=%d",
                config.training.epochs, config.training.batch_size, config.training.learning_rate,
                config.training.random_seed)

    # Seed Python, NumPy and TensorFlow RNGs so the whole pipeline is reproducible
    import tensorflow as tf
    tf.keras.utils.set_random_seed(config.training.random_seed)

    # 2. Step 2 & 3: Data Loading & Exploratory Data Analysis (EDA)
    print_banner("Step 2 & 3: Data Ingestion and Exploratory Data Analysis (EDA)")
    loader = CreditCardDataLoader(data_config=config.data, path_config=config.paths)
    dataset = loader.load()

    eda_analyzer = EdaAnalyzer(output_dir=config.paths.plots_dir)
    eda_stats = eda_analyzer.generate_full_eda_report(dataset.data, target_col=config.data.target_column)

    # 3. Step 4 & 5: Data Preprocessing (Scaling, SMOTE balancing, Stratified Split)
    print_banner("Step 4 & 5: Preprocessing, Train/Val/Test Split & SMOTE Balancing")
    balancing_strategy = SmoteBalancingStrategy(
        sampling_strategy=config.data.smote_sampling_strategy,
        random_state=config.data.random_seed
    )
    preprocessor = DataPreprocessor(config=config.data, balancing_strategy=balancing_strategy)
    preprocessed_data = preprocessor.process(dataset.data)

    logger.info("Data Preprocessing Summary:")
    logger.info("  Training samples:   %d (minority class oversampled with SMOTE)", len(preprocessed_data.x_train))
    logger.info("  Validation samples: %d (realistic imbalanced distribution)", len(preprocessed_data.x_val))
    logger.info("  Testing samples:    %d (realistic imbalanced distribution)", len(preprocessed_data.x_test))
    logger.info("  Feature dimensions: %d", preprocessed_data.input_dim)

    # 4. Step 6 & 7: Model Builders Configuration
    print_banner("Step 6 & 7: Neural Network Architecture & Activation Setup")
    # 2 architectures x 3 activations (ReLU, LeakyReLU, GELU) = 6 experiments.
    # Order matters for the 2x3 plot grids: row 1 = Baseline, row 2 = Regularized.
    activations = ["relu", "leaky_relu", "gelu"]
    model_builders: List[IModelBuilder] = [
        *(BaselineModelBuilder(activation=act, hidden_units=[64, 32]) for act in activations),
        *(RegularizedModelBuilder(activation=act, hidden_units=[64, 32], dropout_rate=0.3, l2_factor=1e-4)
          for act in activations),
    ]

    logger.info("Configured %d distinct model experiments for training.", len(model_builders))

    # 5. Step 8 & 9: Training & Comprehensive Metric Evaluation
    print_banner("Step 8 & 9: Model Training (Fixed Epochs) & Independent Test Set Evaluation")
    trainer = ModelTrainer(config=config.training)
    experiment_results: List[ExperimentRunResult] = []

    models_dir = config.paths.artifacts_dir / "saved_models"
    for builder in model_builders:
        result = trainer.train_and_evaluate(
            builder=builder,
            data=preprocessed_data,
            model_save_dir=models_dir
        )
        experiment_results.append(result)

    # 6. Step 10: Comparative Visualizations
    print_banner("Step 10: Generating Publication-Quality Plots")
    visualizer = TrainingVisualizer(output_dir=config.paths.plots_dir)

    learning_curves_path = visualizer.plot_learning_curves(experiment_results)
    metric_curves_path = visualizer.plot_metric_curves(experiment_results)
    roc_pr_path = visualizer.plot_roc_and_pr_curves(preprocessed_data.y_test, experiment_results)
    cm_path = visualizer.plot_confusion_matrices(experiment_results)
    cm_tuned_path = visualizer.plot_confusion_matrices(experiment_results, tuned=True)
    bar_chart_path = visualizer.plot_metrics_comparison_bar(experiment_results)

    # 7. Step 11 & 12: Summary Report Generation
    print_banner("Step 11 & 12: Model Performance Comparison & Findings")
    default_md = pd.DataFrame([r.test_metrics.to_dict() for r in experiment_results]).to_markdown(index=False)
    tuned_md = pd.DataFrame([r.test_metrics_tuned.to_dict() for r in experiment_results]).to_markdown(index=False)
    overfit_md = pd.DataFrame([r.overfitting_summary for r in experiment_results]).to_markdown(index=False)

    print("\nTest metrics @ threshold 0.5:\n" + default_md)
    print("\nTest metrics @ F1-optimal threshold (tuned on validation):\n" + tuned_md)
    print("\nOver/underfitting diagnostics:\n" + overfit_md + "\n")

    # Persist summary report to artifacts
    report_file = config.paths.artifacts_dir / "metrics_summary.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# Laboratory Work #1: Model Evaluation Summary (TensorFlow)\n\n")
        f.write(f"Seed: {config.training.random_seed}, epochs: {config.training.epochs}, "
                f"optimizer: {config.training.optimizer_name}, lr: {config.training.learning_rate}, "
                f"batch size: {config.training.batch_size}\n\n")
        f.write("## Test Metrics @ threshold 0.5\n\n" + default_md + "\n\n")
        f.write("## Test Metrics @ F1-optimal threshold (tuned on validation set)\n\n" + tuned_md + "\n\n")
        f.write("## Over/Underfitting Diagnostics (train vs validation loss)\n\n" + overfit_md + "\n\n")
        f.write("## Visual Artifacts\n\n")
        f.write(f"- [Learning Curves (loss)](plots/{learning_curves_path.name})\n")
        f.write(f"- [Learning Curves (PR-AUC)](plots/{metric_curves_path.name})\n")
        f.write(f"- [ROC & PR Curves](plots/{roc_pr_path.name})\n")
        f.write(f"- [Confusion Matrices @ 0.5](plots/{cm_path.name})\n")
        f.write(f"- [Confusion Matrices @ tuned threshold](plots/{cm_tuned_path.name})\n")
        f.write(f"- [Metrics Comparison Bar Chart](plots/{bar_chart_path.name})\n")

    logger.info("Summary markdown table saved to %s", report_file)
    logger.info("All tasks completed successfully!")


if __name__ == "__main__":
    main()
