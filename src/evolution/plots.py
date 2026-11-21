"""Plots for Lab 3: convergence, GA vs random search, population evolution, final comparison."""

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # render to files, no GUI window
import matplotlib.pyplot as plt
import numpy as np

from src.evolution.evaluation import FinalEvaluation
from src.evolution.genetic_algorithm import SearchResult
from src.evolution.search_space import SearchSpace

GA_COLOR, RS_COLOR, DEFAULT_COLOR = "#1f77b4", "#d62728", "#7f7f7f"


class EvolutionVisualizer:
    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir

    def plot_convergence(self, ga_runs: Sequence[SearchResult]) -> Path:
        """Best / mean validation RMSE and population diversity per generation, one line per seed."""
        fig, (ax_fit, ax_div) = plt.subplots(1, 2, figsize=(12, 4.2))
        # One colour per seed; solid line = best individual, dashed = population mean
        seed_colors = plt.get_cmap("tab10").colors
        for run, color in zip(ga_runs, seed_colors):
            generations = [g.generation for g in run.generations]
            ax_fit.plot(generations, [g.best_rmse for g in run.generations], "-", color=color,
                        marker="o", markersize=3, label=f"seed {run.seed}: best")
            ax_fit.plot(generations, [g.mean_rmse for g in run.generations], "--", color=color, alpha=0.7,
                        label=f"seed {run.seed}: mean")
            ax_div.plot(generations, [g.diversity for g in run.generations], "-", color=color,
                        marker="o", markersize=3, label=f"seed {run.seed}")
        ax_fit.set(xlabel="Generation", ylabel="Validation RMSE (lower is better)",
                   title="Fitness per generation")
        # The mean of generation 0 contains very poor random models; zoom on the informative range
        best_values = [g.best_rmse for run in ga_runs for g in run.generations]
        final_means = [g.mean_rmse for run in ga_runs for g in run.generations[3:]]
        ax_fit.set_ylim(min(best_values) - 0.005, max(final_means) + 0.02)
        ax_fit.legend(fontsize=7, ncol=2)
        ax_fit.grid(alpha=0.3)
        ax_div.set(xlabel="Generation", ylabel="Mean per-gene std (normalized genes)",
                   title="Population diversity")
        ax_div.legend(fontsize=8)
        ax_div.grid(alpha=0.3)
        return self._save(fig, "ga_convergence.png")

    def plot_ga_vs_random(self, ga_runs: Sequence[SearchResult], rs_runs: Sequence[SearchResult],
                          default_val_rmse: float) -> Path:
        """Best-so-far validation RMSE against the number of fitness evaluations (mean and min-max band)."""
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for runs, color, label in ((ga_runs, GA_COLOR, "Genetic algorithm"), (rs_runs, RS_COLOR, "Random search")):
            curves = np.array([run.best_so_far for run in runs])
            evaluations = np.arange(1, curves.shape[1] + 1)
            ax.plot(evaluations, curves.mean(axis=0), color=color, label=f"{label} (mean of {len(runs)} seeds)")
            ax.fill_between(evaluations, curves.min(axis=0), curves.max(axis=0), color=color, alpha=0.2,
                            label=f"{label} (min-max)")
        ax.axhline(default_val_rmse, color=DEFAULT_COLOR, linestyle="--", label="Default hyperparameters")
        final = [run.best_so_far[-1] for run in (*ga_runs, *rs_runs)]
        ax.set_ylim(min(final) - 0.003, default_val_rmse + 0.01)
        ax.set(xlabel="Fitness evaluations (trained models)", ylabel="Best validation RMSE so far")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        return self._save(fig, "ga_vs_random_search.png")

    def plot_population_evolution(self, run: SearchResult, space: SearchSpace) -> Path:
        """Decoded hyperparameter values of every individual in every generation (one panel per gene)."""
        fig, axes = plt.subplots(2, 4, figsize=(13, 5.5))
        flat_axes = axes.ravel()
        for gene_index, spec in enumerate(space.specs):
            ax = flat_axes[gene_index]
            for stats in run.generations:
                values = [spec.decode(float(gene)) for gene in stats.population[:, gene_index]]
                ax.scatter([stats.generation] * len(values), values, s=9, alpha=0.45, color=GA_COLOR)
            ax.axhline(spec.decode(float(run.best_genome[gene_index])), color=RS_COLOR, linewidth=1,
                       linestyle="--")
            if spec.log_scale:
                ax.set_yscale("log")
            ax.set_title(spec.name, fontsize=10)
            ax.set_xlabel("Generation", fontsize=8)
            ax.grid(alpha=0.3)
        for ax in flat_axes[space.dimension:]:
            ax.axis("off")
        fig.suptitle(f"Population per generation (seed {run.seed}); dashed line = best individual", fontsize=11)
        return self._save(fig, "population_evolution.png")

    def plot_final_comparison(self, default: FinalEvaluation, rs_evals: Sequence[FinalEvaluation],
                              ga_evals: Sequence[FinalEvaluation]) -> Path:
        """Validation and test RMSE: default vs random search vs GA (bars = mean, whiskers = min-max)."""
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        groups = (("Default", [default], DEFAULT_COLOR), ("Random search", rs_evals, RS_COLOR),
                  ("Genetic algorithm", ga_evals, GA_COLOR))
        width = 0.36
        for position, (label, evaluations, color) in enumerate(groups):
            for offset, attribute, hatch in ((-width / 2, "val_rmse", "//"), (width / 2, "test_rmse", "")):
                values = np.array([getattr(e, attribute) for e in evaluations])
                mean = values.mean()
                ax.bar(position + offset, mean, width, color=color, alpha=0.55 if hatch else 1.0, hatch=hatch,
                       edgecolor="black", yerr=[[mean - values.min()], [values.max() - mean]], capsize=4)
                ax.text(position + offset, values.max() + 0.001, f"{mean:.4f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(groups)), [g[0] for g in groups])
        all_values = [v for _, evals, _ in groups for e in evals for v in (e.val_rmse, e.test_rmse)]
        ax.set_ylim(min(all_values) - 0.02, max(all_values) + 0.012)
        ax.set_ylabel("RMSE, $100k (lower is better)")
        ax.bar(0, 0, color="white", edgecolor="black", hatch="//", label="Validation RMSE")
        ax.bar(0, 0, color="white", edgecolor="black", label="Test RMSE (refit on train+val)")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        return self._save(fig, "final_comparison.png")

    def _save(self, fig: plt.Figure, name: str) -> Path:
        path = self._output_dir / name
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path
