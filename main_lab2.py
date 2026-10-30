"""Main orchestrator for Laboratory Work #2 (PyTorch Deep Learning).

Reuses the Lab 1 data pipeline (loader, EDA, preprocessing), metrics and plots unchanged and swaps
only the framework-specific parts (model builders and trainer), then compares the results against
the TensorFlow results persisted by Lab 1. Console output is mirrored to artifacts/lab2/run_output.txt.
"""

import logging
import sys
from typing import List, TextIO

import pandas as pd
import torch

from src.comparison.framework_comparison import FrameworkComparator
from src.config.app_config import ExperimentConfig, PathConfig
from src.data.loader import CreditCardDataLoader
from src.data.preprocessor import DataPreprocessor, SmoteBalancingStrategy
from src.eda.analyzer import EdaAnalyzer
from src.models.torch_builder import ITorchModelBuilder, TorchBaselineModelBuilder, TorchRegularizedModelBuilder
from src.training.results import ExperimentRunResult
from src.training.torch_trainer import TorchModelTrainer, set_global_seed
from src.visualization.plots import TrainingVisualizer


class Tee:
    """Duplicates writes to the console and a log file (like `| tee`)."""

    def __init__(self, *streams: TextIO) -> None:
        self._streams = streams

    def write(self, text: str) -> None:
        for stream in self._streams:
            stream.write(text)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def setup_logging() -> None:
    """Configures structured console logging akin to SLF4J / Logback."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-7s] %(name)-28s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def print_banner(title: str) -> None:
    """Prints a styled CLI section separator."""
    border = "=" * 80
    print(f"\n{border}\n >>> {title.upper()}\n{border}\n")


def main() -> None:
    """Lab 2 execution pipeline."""
    config = ExperimentConfig(paths=PathConfig(artifacts_subdir="lab2"))
    run_log_path = config.paths.artifacts_dir / "run_output.txt"
    run_log = open(run_log_path, "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, run_log)

    setup_logging()
    logger = logging.getLogger("Application")
    logger.info("Initializing Laboratory Work #2 Deep Learning Pipeline (PyTorch %s)...", torch.__version__)
    logger.info("Configuration loaded: Epochs=%d, BatchSize=%d, LearningRate=%.4f, Seed=%d, Device=%s",
                config.training.epochs, config.training.batch_size, config.training.learning_rate,
                config.training.random_seed, config.training.device)
    set_global_seed(config.training.random_seed)

    print_banner("Step 2 & 3: Data Ingestion and Exploratory Data Analysis (EDA)")
    dataset = CreditCardDataLoader(data_config=config.data, path_config=config.paths).load()
    EdaAnalyzer(output_dir=config.paths.plots_dir).generate_full_eda_report(
        dataset.data, target_col=config.data.target_column)

    print_banner("Step 4 & 5: Preprocessing, Train/Val/Test Split & SMOTE Balancing")
    balancing_strategy = SmoteBalancingStrategy(
        sampling_strategy=config.data.smote_sampling_strategy,
        random_state=config.data.random_seed
    )
    data = DataPreprocessor(config=config.data, balancing_strategy=balancing_strategy).process(dataset.data)
    logger.info("Train=%d (SMOTE), Val=%d, Test=%d, features=%d",
                len(data.x_train), len(data.x_val), len(data.x_test), data.input_dim)

    print_banner("Step 6 & 7: Neural Network Architecture & Activation Setup (PyTorch)")
    # Same 2 architectures x 3 activations as Lab 1; order defines the 2x3 plot grids
    activations = ["relu", "leaky_relu", "gelu"]
    model_builders: List[ITorchModelBuilder] = [
        *(TorchBaselineModelBuilder(activation=act, hidden_units=[64, 32]) for act in activations),
        *(TorchRegularizedModelBuilder(activation=act, hidden_units=[64, 32], dropout_rate=0.3, l2_factor=1e-4)
          for act in activations),
    ]
    logger.info("Configured %d distinct model experiments for training.", len(model_builders))
    for builder in model_builders[::3]:
        logger.info("%s architecture:\n%s", builder.name, builder.build(data.input_dim))

    print_banner("Step 8 & 9: Model Training (Fixed Epochs) & Independent Test Set Evaluation")
    trainer = TorchModelTrainer(config=config.training)
    models_dir = config.paths.artifacts_dir / "saved_models"
    results: List[ExperimentRunResult] = [
        trainer.train_and_evaluate(builder=builder, data=data, model_save_dir=models_dir)
        for builder in model_builders
    ]

    print_banner("Step 10: Generating Publication-Quality Plots")
    visualizer = TrainingVisualizer(output_dir=config.paths.plots_dir, title_suffix=" [PyTorch]")
    plot_paths = [
        ("Learning Curves (loss)", visualizer.plot_learning_curves(results)),
        ("Learning Curves (PR-AUC)", visualizer.plot_metric_curves(results)),
        ("ROC & PR Curves", visualizer.plot_roc_and_pr_curves(data.y_test, results)),
        ("Confusion Matrices @ 0.5", visualizer.plot_confusion_matrices(results)),
        ("Confusion Matrices @ tuned threshold", visualizer.plot_confusion_matrices(results, tuned=True)),
        ("Metrics Comparison Bar Chart", visualizer.plot_metrics_comparison_bar(results)),
    ]

    print_banner("Step 11 & 12: Model Comparison and TensorFlow vs PyTorch")
    default_md = pd.DataFrame([r.test_metrics.to_dict() for r in results]).to_markdown(index=False)
    tuned_md = pd.DataFrame([r.test_metrics_tuned.to_dict() for r in results]).to_markdown(index=False)
    overfit_md = pd.DataFrame([r.overfitting_summary for r in results]).to_markdown(index=False)
    print("\nTest metrics @ threshold 0.5:\n" + default_md)
    print("\nTest metrics @ F1-optimal threshold (tuned on validation):\n" + tuned_md)
    print("\nOver/underfitting diagnostics:\n" + overfit_md + "\n")

    sys.stdout.flush()
    comparator = FrameworkComparator(
        tf_artifacts_dir=ExperimentConfig().paths.artifacts_dir, output_dir=config.paths.plots_dir)
    comparison = comparator.metrics_table(results)
    timing = comparator.timing_table(results, run_log_path)
    comparison_md = comparison.to_markdown(index=False)
    timing_md = timing.to_markdown(index=False)
    plot_paths.append(("TensorFlow vs PyTorch", comparator.plot_comparison(comparison)))
    print("\nTensorFlow (Lab 1) vs PyTorch (Lab 2) @ tuned threshold:\n" + comparison_md)
    print("\nWall-clock time per experiment:\n" + timing_md + "\n")

    report_file = config.paths.artifacts_dir / "metrics_summary.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# Laboratory Work #2: Model Evaluation Summary (PyTorch)\n\n")
        f.write(f"PyTorch {torch.__version__}, device: {config.training.device}, seed: {config.training.random_seed}, "
                f"epochs: {config.training.epochs}, optimizer: {config.training.optimizer_name}, "
                f"lr: {config.training.learning_rate}, batch size: {config.training.batch_size}\n\n")
        f.write("## Test Metrics @ threshold 0.5\n\n" + default_md + "\n\n")
        f.write("## Test Metrics @ F1-optimal threshold (tuned on validation set)\n\n" + tuned_md + "\n\n")
        f.write("## Over/Underfitting Diagnostics (train vs validation loss)\n\n" + overfit_md + "\n\n")
        f.write("## TensorFlow (Lab 1) vs PyTorch (Lab 2) @ tuned threshold\n\n" + comparison_md + "\n\n")
        f.write("## Wall-clock Time per Experiment\n\n" + timing_md + "\n\n")
        f.write("## Visual Artifacts\n\n")
        for title, path in plot_paths:
            f.write(f"- [{title}](plots/{path.name})\n")

    logger.info("Summary markdown table saved to %s", report_file)
    logger.info("All tasks completed successfully!")
    sys.stdout.flush()
    sys.stdout = sys.__stdout__
    run_log.close()


if __name__ == "__main__":
    main()
