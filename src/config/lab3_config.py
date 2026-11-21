"""Configuration for Laboratory Work #3: genetic algorithm for hyperparameter optimization.

All numeric choices of the experiment live here, each with a short justification.
"""

from dataclasses import dataclass, field
from typing import Tuple

from src.config.app_config import PathConfig
from src.evolution.search_space import HyperparameterSpec, ParameterKind, SearchSpace


def default_search_space() -> SearchSpace:
    """Hyperparameters of HistGradientBoostingRegressor tuned by the genetic algorithm.

    Ranges are wide enough to contain both clearly underfitting (few shallow trees) and clearly
    overfitting (many big trees, no regularization) models, so the search is not trivial.
    """
    return SearchSpace(specs=(
        # Step size of boosting; acts multiplicatively, hence the log scale
        HyperparameterSpec("learning_rate", ParameterKind.FLOAT, 0.01, 0.5, log_scale=True),
        # Number of trees; the upper bound keeps one fitness evaluation within a few seconds
        HyperparameterSpec("max_iter", ParameterKind.INT, 50, 500),
        # Tree complexity
        HyperparameterSpec("max_leaf_nodes", ParameterKind.INT, 4, 64),
        # Minimum samples per leaf: larger values smooth the model
        HyperparameterSpec("min_samples_leaf", ParameterKind.INT, 1, 100),
        # L2 penalty on leaf values; spans several orders of magnitude, hence the log scale
        HyperparameterSpec("l2_regularization", ParameterKind.FLOAT, 1e-4, 10.0, log_scale=True),
        # Fraction of features considered per split (column subsampling)
        HyperparameterSpec("max_features", ParameterKind.FLOAT, 0.3, 1.0),
        # Histogram resolution used to bin feature values
        HyperparameterSpec("max_bins", ParameterKind.INT, 16, 255),
    ))


@dataclass(frozen=True)
class HousingDataConfig:
    """California Housing split: 60% train / 20% validation / 20% test."""
    test_size: float = 0.2
    # Share of the remaining 80% that becomes validation (0.25 * 0.8 = 0.2 of all data)
    val_size_of_rest: float = 0.25
    random_seed: int = 42


@dataclass(frozen=True)
class GeneticAlgorithmConfig:
    """Parameters of the genetic algorithm (genes are normalized to [0, 1])."""
    # 20 individuals x 15 generations = 272 model fits: enough to converge, minutes of CPU time
    population_size: int = 20
    generations: int = 15
    # Small tournaments keep selection pressure moderate and preserve diversity
    tournament_size: int = 3
    # Most children are recombined; the remaining 10% are mutated copies of one parent
    crossover_probability: float = 0.9
    # ~1.4 of 7 genes mutate per child on average
    mutation_probability: float = 0.2
    # Std of Gaussian noise in normalized gene units (10% of the gene range)
    mutation_sigma: float = 0.1
    # The 2 best individuals are copied unchanged, so the best fitness never degrades
    elite_count: int = 2

    @property
    def evaluation_budget(self) -> int:
        """Number of fitness evaluations per run (elites are not re-evaluated)."""
        return self.population_size + (self.generations - 1) * (self.population_size - self.elite_count)


@dataclass(frozen=True)
class Lab3Config:
    """Root configuration of Lab 3."""
    data: HousingDataConfig = field(default_factory=HousingDataConfig)
    ga: GeneticAlgorithmConfig = field(default_factory=GeneticAlgorithmConfig)
    search_space: SearchSpace = field(default_factory=default_search_space)
    paths: PathConfig = field(default_factory=lambda: PathConfig(artifacts_subdir="lab3"))
    # Independent repetitions of GA and random search: a single run could be just a lucky seed
    search_seeds: Tuple[int, ...] = (1, 2, 3, 4, 5)
    # Seed of the model itself (tree building); fixed so fitness is a deterministic function
    model_seed: int = 42
    # OpenMP threads used by the model; 1 is the fastest setting for a dataset this small
    model_threads: int = 1
