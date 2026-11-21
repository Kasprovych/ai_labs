"""Main orchestrator for Laboratory Work #3 (Evolutionary Algorithms, variant 1).

A genetic algorithm tunes the hyperparameters of a gradient boosting regressor on the California
Housing dataset; fitness is the RMSE on the validation split. The GA is compared with the default
hyperparameters and with random search given the same number of fitness evaluations.
Console output is mirrored to artifacts/lab3/run_output.txt.
"""

import json
import logging
import time
from typing import List

import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

from src.config.lab3_config import Lab3Config
from src.evolution.evaluation import FinalEvaluation, FinalEvaluator
from src.evolution.fitness import ValidationRmseFitness
from src.evolution.genetic_algorithm import GeneticAlgorithm, RandomSearch, SearchResult
from src.evolution.housing_data import HousingDataLoader
from src.evolution.operators import GaussianMutation, TournamentSelection, UniformCrossover
from src.evolution.plots import EvolutionVisualizer
from src.utils.console import mirrored_stdout, print_banner, setup_logging

# scikit-learn defaults of HistGradientBoostingRegressor (the "no tuning" baseline)
DEFAULT_HYPERPARAMETERS = {"learning_rate": 0.1, "max_iter": 100, "max_leaf_nodes": 31, "min_samples_leaf": 20,
                           "l2_regularization": 0.0, "max_features": 1.0, "max_bins": 255}


def summarize(evaluations: List[FinalEvaluation], label: str) -> dict:
    """Mean and standard deviation over seeds for one search method."""
    val = np.array([e.val_rmse for e in evaluations])
    test = np.array([e.test_rmse for e in evaluations])
    r2 = np.array([e.test_r2 for e in evaluations])
    return {"Method": label, "Runs": len(evaluations),
            "Val RMSE (mean ± std)": f"{val.mean():.4f} ± {val.std():.4f}",
            "Test RMSE (mean ± std)": f"{test.mean():.4f} ± {test.std():.4f}",
            "Best test RMSE": f"{test.min():.4f}", "Test R2 (mean)": f"{r2.mean():.4f}"}


def run(config: Lab3Config) -> None:
    logger = logging.getLogger("Application")
    ga_config = config.ga
    logger.info("Initializing Laboratory Work #3: Genetic Algorithm for Hyperparameter Optimization "
                "(scikit-learn %s)", sklearn.__version__)

    print_banner("Step 1: Dataset and train / validation / test split")
    data = HousingDataLoader(config.data, cache_dir=config.paths.data_dir / "sklearn_cache").load()
    logger.info("Features: %s; target: median house value ($100k)", ", ".join(data.feature_names))

    print_banner("Step 2: Search space (genotype -> hyperparameters)")
    space = config.search_space
    space_md = pd.DataFrame([{"Gene": i, "Hyperparameter": s.name, "Type": s.kind.value, "Range": s.range_label,
                              "Default": DEFAULT_HYPERPARAMETERS[s.name]}
                             for i, s in enumerate(space.specs)]).to_markdown(index=False)
    print(space_md)

    print_banner("Step 3: Baseline with default hyperparameters")
    evaluator = FinalEvaluator(data, model_seed=config.model_seed)
    default_eval = evaluator.evaluate("Default", DEFAULT_HYPERPARAMETERS)
    logger.info("Default model: validation RMSE %.4f", default_eval.val_rmse)

    print_banner("Step 4: Genetic algorithm (selection -> crossover -> mutation -> elitism)")
    logger.info("Population=%d, generations=%d, tournament=%d, p_crossover=%.2f, p_mutation=%.2f, sigma=%.2f, "
                "elites=%d -> %d fitness evaluations per run", ga_config.population_size, ga_config.generations,
                ga_config.tournament_size, ga_config.crossover_probability, ga_config.mutation_probability,
                ga_config.mutation_sigma, ga_config.elite_count, ga_config.evaluation_budget)
    ga_runs: List[SearchResult] = []
    timing_rows = []
    for seed in config.search_seeds:
        logger.info("GA run, seed %d", seed)
        fitness = ValidationRmseFitness(space, data, model_seed=config.model_seed)
        algorithm = GeneticAlgorithm(
            search_space=space, fitness=fitness,
            selection=TournamentSelection(ga_config.tournament_size),
            crossover=UniformCrossover(ga_config.crossover_probability),
            mutation=GaussianMutation(ga_config.mutation_probability, ga_config.mutation_sigma),
            config=ga_config)
        started = time.perf_counter()
        ga_runs.append(algorithm.run(seed))
        timing_rows.append({"Method": "Genetic algorithm", "Seed": seed, "Fitness evaluations": len(ga_runs[-1].best_so_far),
                            "Models trained": fitness.model_fits, "Seconds": round(time.perf_counter() - started, 1)})

    print_banner("Step 5: Random search with the same evaluation budget")
    rs_runs: List[SearchResult] = []
    for seed in config.search_seeds:
        logger.info("Random search run, seed %d", seed)
        fitness = ValidationRmseFitness(space, data, model_seed=config.model_seed)
        started = time.perf_counter()
        rs_runs.append(RandomSearch(space, fitness, budget=ga_config.evaluation_budget).run(seed))
        timing_rows.append({"Method": "Random search", "Seed": seed, "Fitness evaluations": len(rs_runs[-1].best_so_far),
                            "Models trained": fitness.model_fits, "Seconds": round(time.perf_counter() - started, 1)})

    print_banner("Step 6: Final evaluation on the held-out test split")
    ga_evals = [evaluator.evaluate(f"GA, seed {r.seed}", space.decode(r.best_genome)) for r in ga_runs]
    rs_evals = [evaluator.evaluate(f"Random search, seed {r.seed}", space.decode(r.best_genome)) for r in rs_runs]
    all_evals = [default_eval, *rs_evals, *ga_evals]
    per_run_md = pd.DataFrame([e.to_row() for e in all_evals]).to_markdown(index=False)
    hyper_md = pd.DataFrame([{"Configuration": e.label,
                              **{k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in e.hyperparameters.items()}}
                             for e in all_evals]).to_markdown(index=False)
    summary_md = pd.DataFrame([summarize([default_eval], "Default hyperparameters"),
                               summarize(rs_evals, "Random search"),
                               summarize(ga_evals, "Genetic algorithm")]).to_markdown(index=False)
    timing_md = pd.DataFrame(timing_rows).to_markdown(index=False)
    best_ga_run = min(ga_runs, key=lambda r: r.best_val_rmse)
    generations_md = pd.DataFrame([{"Generation": g.generation, "Best RMSE": round(g.best_rmse, 4),
                                    "Mean RMSE": round(g.mean_rmse, 4), "Worst RMSE": round(g.worst_rmse, 4),
                                    "Diversity": round(g.diversity, 3), "Evaluations": g.evaluations}
                                   for g in best_ga_run.generations]).to_markdown(index=False)
    print("\nBest hyperparameters found in every run:\n" + hyper_md)
    print("\nValidation and test metrics per run:\n" + per_run_md)
    print("\nSummary over seeds (same evaluation budget for GA and random search):\n" + summary_md)
    print("\nSearch cost:\n" + timing_md + "\n")

    print_banner("Step 7: Plots")
    visualizer = EvolutionVisualizer(config.paths.plots_dir)
    plot_paths = [
        ("GA convergence and diversity", visualizer.plot_convergence(ga_runs)),
        ("GA vs random search", visualizer.plot_ga_vs_random(ga_runs, rs_runs, default_eval.val_rmse)),
        ("Population evolution", visualizer.plot_population_evolution(best_ga_run, space)),
        ("Final comparison", visualizer.plot_final_comparison(default_eval, rs_evals, ga_evals)),
    ]
    for title, path in plot_paths:
        logger.info("Saved plot '%s' to %s", title, path)

    artifacts = config.paths.artifacts_dir
    with open(artifacts / "metrics_summary.md", "w", encoding="utf-8") as f:
        f.write("# Laboratory Work #3: Genetic Algorithm for Hyperparameter Optimization\n\n")
        f.write(f"Model: HistGradientBoostingRegressor (scikit-learn {sklearn.__version__}); dataset: California "
                f"Housing; fitness: validation RMSE; GA seeds: {list(config.search_seeds)}\n\n")
        f.write("## Search Space\n\n" + space_md + "\n\n")
        f.write("## Summary over Seeds\n\n" + summary_md + "\n\n")
        f.write("## Validation and Test Metrics per Run\n\n" + per_run_md + "\n\n")
        f.write("## Best Hyperparameters per Run\n\n" + hyper_md + "\n\n")
        f.write(f"## Generations of the Best GA Run (seed {best_ga_run.seed})\n\n" + generations_md + "\n\n")
        f.write("## Search Cost\n\n" + timing_md + "\n\n")
        f.write("## Visual Artifacts\n\n")
        for title, path in plot_paths:
            f.write(f"- [{title}](plots/{path.name})\n")
    with open(artifacts / "results.json", "w", encoding="utf-8") as f:
        json.dump({"default": default_eval.__dict__, "random_search": [e.__dict__ for e in rs_evals],
                   "genetic_algorithm": [e.__dict__ for e in ga_evals],
                   "ga_best_so_far": {str(r.seed): r.best_so_far for r in ga_runs},
                   "rs_best_so_far": {str(r.seed): r.best_so_far for r in rs_runs}}, f, indent=2)
    logger.info("Summary saved to %s", artifacts / "metrics_summary.md")
    logger.info("All tasks completed successfully!")


def main() -> None:
    config = Lab3Config()
    with mirrored_stdout(config.paths.artifacts_dir / "run_output.txt"):
        setup_logging()
        # On ~12k rows OpenMP thread management costs more than it saves: one model fit takes ~0.7 s
        # on a single thread versus several seconds on all cores (measured on Apple M1 Pro).
        with threadpool_limits(limits=config.model_threads, user_api="openmp"):
            run(config)


if __name__ == "__main__":
    main()
