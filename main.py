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
    logger.info("Configuration loaded: Epochs=%d, BatchSize=%d, LearningRate=%.4f",
                config.training.epochs, config.training.batch_size, config.training.learning_rate)

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
    # We test:
    # 1. Baseline Feedforward (ReLU)
    # 2. Baseline Feedforward (GELU)
    # 3. Regularized (BatchNorm + Dropout + L2) (ReLU)
    # 4. Regularized (BatchNorm + Dropout + L2) (GELU)
    model_builders: List[IModelBuilder] = [
        BaselineModelBuilder(activation="relu", hidden_units=[64, 32]),
        BaselineModelBuilder(activation="gelu", hidden_units=[64, 32]),
        RegularizedModelBuilder(activation="relu", hidden_units=[64, 32], dropout_rate=0.3, l2_factor=1e-4),
        RegularizedModelBuilder(activation="gelu", hidden_units=[64, 32], dropout_rate=0.3, l2_factor=1e-4),
    ]

    logger.info("Configured %d distinct model experiments for training.", len(model_builders))

    # 5. Step 8 & 9: Training & Comprehensive Metric Evaluation
    print_banner("Step 8 & 9: Model Training (15 Epochs) & Independent Test Set Evaluation")
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
    roc_pr_path = visualizer.plot_roc_and_pr_curves(preprocessed_data.y_test, experiment_results)
    cm_path = visualizer.plot_confusion_matrices(experiment_results)
    bar_chart_path = visualizer.plot_metrics_comparison_bar(experiment_results)

    # 7. Step 11 & 12: Summary Report Generation
    print_banner("Step 11 & 12: Model Performance Comparison & Findings")
    metrics_table = [res.test_metrics.to_dict() for res in experiment_results]
    metrics_df = pd.DataFrame(metrics_table)

    table_markdown = metrics_df.to_markdown(index=False)
    print("\n" + table_markdown + "\n")

    # Persist summary report to artifacts
    report_file = config.paths.artifacts_dir / "metrics_summary.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# Laboratory Work #1: Model Evaluation Summary (TensorFlow)\n\n")
        f.write("## Performance Metrics Table\n\n")
        f.write(table_markdown + "\n\n")
        f.write("## Visual Artifacts\n\n")
        f.write(f"- [Learning Curves]({learning_curves_path.name})\n")
        f.write(f"- [ROC & PR Curves]({roc_pr_path.name})\n")
        f.write(f"- [Confusion Matrices]({cm_path.name})\n")
        f.write(f"- [Metrics Comparison Bar Chart]({bar_chart_path.name})\n")

    logger.info("Summary markdown table saved to %s", report_file)
    logger.info("All tasks completed successfully!")


if __name__ == "__main__":
    main()
