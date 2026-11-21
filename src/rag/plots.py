"""Plots for Lab 4: chunk lengths, retrieval quality, answer quality, per-question results, latency."""

from pathlib import Path
from typing import Dict, Sequence

import matplotlib

matplotlib.use("Agg")  # render to files, no GUI window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.rag.evaluation import AnswerRecord

COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")


class RagVisualizer:
    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir

    def plot_chunk_lengths(self, tokens_by_chunking: Dict[str, Sequence[int]], token_limit: int) -> Path:
        """Distribution of chunk lengths in embedding-model tokens against the model input limit."""
        fig, axes = plt.subplots(1, len(tokens_by_chunking), figsize=(12, 3.6), sharey=False)
        for ax, (label, tokens), color in zip(axes, tokens_by_chunking.items(), COLORS):
            tokens = np.asarray(tokens)
            ax.hist(tokens, bins=30, color=color, alpha=0.8, edgecolor="black", linewidth=0.3)
            ax.axvline(token_limit, color="black", linestyle="--", label=f"model limit: {token_limit} tokens")
            share = 100 * (tokens > token_limit).mean()
            ax.set_title(f"{label}: {len(tokens)} chunks, {share:.0f}% truncated", fontsize=10)
            ax.set_xlabel("Chunk length, tokens")
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("Number of chunks")
        return self._save(fig, "chunk_lengths.png")

    def plot_retrieval_grid(self, grid: pd.DataFrame) -> Path:
        """Hit@k and MRR for every chunk size and ranking method."""
        metrics = ["Hit@1", "Hit@3", "Hit@5", "MRR"]
        labels = [f"{row['Chunk size, words']} words\n{row['Ranking']}" for _, row in grid.iterrows()]
        fig, ax = plt.subplots(figsize=(11, 4.2))
        width = 0.2
        positions = np.arange(len(grid))
        for index, (metric, color) in enumerate(zip(metrics, COLORS)):
            bars = ax.bar(positions + (index - 1.5) * width, grid[metric], width, label=metric, color=color,
                          edgecolor="black", linewidth=0.4)
            ax.bar_label(bars, fmt="%.2f", fontsize=6.5, padding=1)
        ax.set_xticks(positions, labels, fontsize=8)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("Score (answerable control questions)")
        ax.legend(ncol=4, fontsize=8, loc="lower right")
        ax.grid(axis="y", alpha=0.3)
        return self._save(fig, "retrieval_grid.png")

    def plot_answer_quality(self, summary: pd.DataFrame) -> Path:
        """LLM-judge grades (left) and objective key-fact coverage (right) per setup."""
        fig, (ax_judge, ax_facts) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [2, 1]})
        labels = [name.split(":")[0] for name in summary["Setup"]]
        positions = np.arange(len(summary))
        width = 0.27
        for index, (metric, color) in enumerate(zip(
                ["Accuracy (1-5)", "Completeness (1-5)", "Coherence (1-5)"], COLORS)):
            bars = ax_judge.bar(positions + (index - 1) * width, summary[metric], width, label=metric, color=color,
                                edgecolor="black", linewidth=0.4)
            ax_judge.bar_label(bars, fmt="%.1f", fontsize=6.5, padding=1)
        ax_judge.set_xticks(positions, labels)
        ax_judge.set_ylim(1, 5.9)
        ax_judge.set_ylabel("Mean grade of the LLM judge")
        ax_judge.legend(ncol=3, fontsize=8, loc="upper center")
        ax_judge.grid(axis="y", alpha=0.3)
        bars = ax_facts.bar(positions, summary["Key facts covered"], color="#7f7f7f", edgecolor="black",
                            linewidth=0.4)
        ax_facts.bar_label(bars, fmt="%.2f", fontsize=7, padding=1)
        ax_facts.set_xticks(positions, labels)
        ax_facts.set_ylim(0, 1.1)
        ax_facts.set_ylabel("Share of key facts in answers")
        ax_facts.grid(axis="y", alpha=0.3)
        fig.text(0.5, 0.005, "   ".join(summary["Setup"]), ha="center", fontsize=6.5, wrap=True)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        return self._save(fig, "answer_quality.png", tight=False)

    def plot_question_heatmap(self, records: Sequence[AnswerRecord]) -> Path:
        """Key-fact coverage of every answer: rows are setups, columns are control questions."""
        frame = pd.DataFrame([record.__dict__ for record in records])
        table = frame.pivot(index="setup", columns="question_id", values="fact_coverage")
        table = table.loc[list(dict.fromkeys(frame["setup"]))]  # keep the run order of setups
        fig, ax = plt.subplots(figsize=(12, 4.2))
        image = ax.imshow(table.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(table.columns)), table.columns, fontsize=8)
        ax.set_yticks(range(len(table.index)), table.index, fontsize=8)
        for row in range(table.shape[0]):
            for column in range(table.shape[1]):
                ax.text(column, row, f"{table.values[row, column]:.2f}", ha="center", va="center", fontsize=7)
        fig.colorbar(image, ax=ax, label="Share of key facts in the answer")
        return self._save(fig, "question_heatmap.png")

    def plot_latency(self, summary: pd.DataFrame) -> Path:
        """Mean time per question: retrieval (incl. re-ranking) and generation."""
        fig, ax = plt.subplots(figsize=(8, 4))
        labels = [name.split(":")[0] for name in summary["Setup"]]
        ax.bar(labels, summary["Retrieval, s"], label="Retrieval + ranking", color=COLORS[0], edgecolor="black",
               linewidth=0.4)
        bars = ax.bar(labels, summary["Generation, s"], bottom=summary["Retrieval, s"], label="LLM generation",
                      color=COLORS[1], edgecolor="black", linewidth=0.4)
        ax.bar_label(bars, labels=[f"{total:.1f}" for total in summary["Retrieval, s"] + summary["Generation, s"]],
                     fontsize=8, padding=2)
        ax.set_ylabel("Mean seconds per question")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        return self._save(fig, "latency.png")

    def _save(self, fig: plt.Figure, name: str, tight: bool = True) -> Path:
        path = self._output_dir / name
        if tight:
            fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path
