"""TensorFlow (Lab 1) vs PyTorch (Lab 2) comparison.

Lab 1 results are read from its persisted artifacts (metrics_summary.md, run_output.txt), so the
comparison needs no TensorFlow installation and uses exactly the numbers reported in Lab 1.
"""

from datetime import datetime
import logging
from pathlib import Path
import re
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.training.results import ExperimentRunResult

logger = logging.getLogger(__name__)

TUNED_SECTION = "Test Metrics @ F1-optimal threshold (tuned on validation set)"
DEFAULT_SECTION = "Test Metrics @ threshold 0.5"
OVERFIT_SECTION = "Over/Underfitting Diagnostics (train vs validation loss)"

_LOG_LINE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) .*?: (Starting experiment: (\S+)|Model saved to .*/([^/]+)\.(?:keras|pt))")


def read_markdown_tables(path: Path) -> Dict[str, pd.DataFrame]:
    """Parses every '## Section' followed by a pipe table into {section title: DataFrame}."""
    tables: Dict[str, pd.DataFrame] = {}
    section, rows = None, []

    def flush() -> None:
        if section and len(rows) > 2:
            header = [c.strip() for c in rows[0].strip("|").split("|")]
            body = [[c.strip() for c in r.strip("|").split("|")] for r in rows[2:]]
            tables[section] = pd.DataFrame(body, columns=header)

    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            flush()
            section, rows = line[3:].strip(), []
        elif line.startswith("|"):
            rows.append(line.strip())
    flush()
    return tables


def experiment_durations(run_log: Path) -> Dict[str, float]:
    """Seconds from 'Starting experiment' to 'Model saved' per model (fit + predict + evaluation)."""
    starts: Dict[str, datetime] = {}
    durations: Dict[str, float] = {}
    for line in run_log.read_text(encoding="utf-8").splitlines():
        match = _LOG_LINE.match(line)
        if not match:
            continue
        ts = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
        if match.group(3):
            starts[match.group(3)] = ts
        elif match.group(4) in starts:
            durations[match.group(4)] = (ts - starts[match.group(4)]).total_seconds()
    return durations


class FrameworkComparator:
    """Builds side-by-side TensorFlow / PyTorch tables and charts for identical configurations."""

    def __init__(self, tf_artifacts_dir: Path, output_dir: Path) -> None:
        self._tf_tables = read_markdown_tables(tf_artifacts_dir / "metrics_summary.md")
        self._tf_log = tf_artifacts_dir / "run_output.txt"
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def metrics_table(self, results: List[ExperimentRunResult]) -> pd.DataFrame:
        """Per-configuration F1 (tuned threshold), PR-AUC, ROC-AUC for both frameworks."""
        tf_tuned = self._tf_tables[TUNED_SECTION].set_index("Model")
        rows = []
        for res in results:
            tf_row = tf_tuned.loc[res.model_name]
            pt = res.test_metrics_tuned
            rows.append({
                "Model": res.model_name,
                "TF F1": float(tf_row["F1-Score"]),
                "PT F1": round(pt.f1_score, 4),
                "ΔF1 (PT-TF)": round(pt.f1_score - float(tf_row["F1-Score"]), 4),
                "TF PR-AUC": float(tf_row["PR-AUC (AP)"]),
                "PT PR-AUC": round(pt.pr_auc, 4),
                "TF ROC-AUC": float(tf_row["ROC-AUC"]),
                "PT ROC-AUC": round(pt.roc_auc, 4),
                "TF TP/FP/FN": f"{tf_row['TP']}/{tf_row['FP']}/{tf_row['FN']}",
                "PT TP/FP/FN": f"{pt.tp}/{pt.fp}/{pt.fn}",
            })
        return pd.DataFrame(rows)

    def timing_table(self, results: List[ExperimentRunResult], pt_run_log: Path) -> pd.DataFrame:
        """Wall-clock time per experiment parsed identically from both run logs (1 s resolution)."""
        tf_time = experiment_durations(self._tf_log)
        pt_time = experiment_durations(pt_run_log)
        return pd.DataFrame([{
            "Model": r.model_name,
            "TF experiment, s": tf_time.get(r.model_name, np.nan),
            "PT experiment, s": pt_time.get(r.model_name, np.nan),
            "PT training loop, s": round(r.training_seconds, 1),
        } for r in results])

    def plot_comparison(self, table: pd.DataFrame) -> Path:
        """Grouped bars: F1 (tuned threshold) and PR-AUC, TensorFlow vs PyTorch per configuration."""
        labels = [m.replace("Regularized_BN_Dropout_L2_", "Reg. ").replace("Baseline_FFN_", "Base ")
                  for m in table["Model"]]
        x = np.arange(len(labels))
        width = 0.38
        fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
        for ax, metric, title in [(axes[0], "F1", "Test F1 @ validation-tuned threshold"),
                                  (axes[1], "PR-AUC", "Test PR-AUC (threshold-independent)")]:
            for offset, prefix, color, name in [(-width / 2, "TF", "#FF8F00", "TensorFlow 2.21 (Lab 1)"),
                                                (width / 2, "PT", "#4C72B0", "PyTorch (Lab 2)")]:
                values = table[f"{prefix} {metric}"].to_numpy(dtype=float)
                bars = ax.bar(x + offset, values, width, label=name, color=color, edgecolor="black", alpha=0.85)
                for bar, value in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width() / 2, value + 0.005, f"{value:.3f}",
                            ha="center", va="bottom", fontsize=8)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=15)
            ax.set_ylim(0.6, 0.9)
            ax.set_title(title, fontsize=13, weight="bold")
            ax.grid(axis="y", linestyle=":", alpha=0.7)
            ax.legend(loc="upper left", fontsize=9)
        fig.suptitle("TensorFlow vs PyTorch: identical architectures, data split and hyperparameters",
                     fontsize=14, weight="bold")
        plt.tight_layout()
        save_path = self._output_dir / "framework_comparison.png"
        fig.savefig(save_path, dpi=200)
        plt.close(fig)
        logger.info("Saved framework comparison chart to %s", save_path)
        return save_path
