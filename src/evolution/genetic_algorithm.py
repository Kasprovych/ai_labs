"""Generational genetic algorithm with elitism, plus the random-search baseline.

One generation: evaluate fitness -> copy elites -> repeat (select two parents -> crossover ->
mutation) until the new population is full. Both searchers record their history as plain data
(DTOs); plotting and reporting are done elsewhere.
"""

import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from src.config.lab3_config import GeneticAlgorithmConfig
from src.evolution.fitness import IFitnessFunction
from src.evolution.operators import ICrossover, IMutation, ISelection
from src.evolution.search_space import SearchSpace

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GenerationStats:
    """Snapshot of one generation (fitness = -validation RMSE, so errors are stored as positive RMSE)."""
    generation: int
    best_rmse: float
    mean_rmse: float
    worst_rmse: float
    # Mean per-gene standard deviation: how spread out the population still is (0 = all identical)
    diversity: float
    # Fitness evaluations spent so far, including this generation
    evaluations: int
    population: np.ndarray  # shape (population_size, genes)


@dataclass(frozen=True)
class SearchResult:
    """Outcome of one search run (GA or random search)."""
    method: str
    seed: int
    best_genome: np.ndarray
    best_val_rmse: float
    # best_so_far[i] = lowest validation RMSE after i + 1 fitness evaluations
    best_so_far: Tuple[float, ...]
    generations: Tuple[GenerationStats, ...] = ()


class GeneticAlgorithm:
    """Maximizes a fitness function over genomes in [0, 1]^n."""

    def __init__(self, search_space: SearchSpace, fitness: IFitnessFunction, selection: ISelection,
                 crossover: ICrossover, mutation: IMutation, config: GeneticAlgorithmConfig) -> None:
        self._space = search_space
        self._fitness = fitness
        self._selection = selection
        self._crossover = crossover
        self._mutation = mutation
        self._config = config

    def run(self, seed: int) -> SearchResult:
        rng = np.random.default_rng(seed)
        config = self._config
        best_so_far: List[float] = []

        def evaluate(genome: np.ndarray) -> float:
            score = self._fitness.evaluate(genome)
            best_so_far.append(min(-score, best_so_far[-1]) if best_so_far else -score)
            return score

        # Generation 0: random individuals
        population = [self._space.random_genome(rng) for _ in range(config.population_size)]
        fitness = [evaluate(genome) for genome in population]
        history = [self._snapshot(0, population, fitness, len(best_so_far))]

        for generation in range(1, config.generations):
            # Elitism: the best individuals survive unchanged (their fitness is already known)
            ranked = sorted(range(len(population)), key=lambda index: fitness[index], reverse=True)
            next_population = [population[index].copy() for index in ranked[:config.elite_count]]
            next_fitness = [fitness[index] for index in ranked[:config.elite_count]]

            while len(next_population) < config.population_size:
                parent_a = self._selection.select(population, fitness, rng)
                parent_b = self._selection.select(population, fitness, rng)
                for child in self._crossover.recombine(parent_a, parent_b, rng):
                    if len(next_population) < config.population_size:
                        child = self._mutation.mutate(child, rng)
                        next_population.append(child)
                        next_fitness.append(evaluate(child))

            population, fitness = next_population, next_fitness
            history.append(self._snapshot(generation, population, fitness, len(best_so_far)))

        best_index = int(np.argmax(fitness))
        return SearchResult(method="Genetic algorithm", seed=seed, best_genome=population[best_index].copy(),
                            best_val_rmse=-fitness[best_index], best_so_far=tuple(best_so_far),
                            generations=tuple(history))

    @staticmethod
    def _snapshot(generation: int, population: List[np.ndarray], fitness: List[float],
                  evaluations: int) -> GenerationStats:
        rmse = -np.asarray(fitness)
        genomes = np.vstack(population)
        stats = GenerationStats(generation=generation, best_rmse=float(rmse.min()), mean_rmse=float(rmse.mean()),
                                worst_rmse=float(rmse.max()), diversity=float(genomes.std(axis=0).mean()),
                                evaluations=evaluations, population=genomes)
        logger.info("  generation %2d | best RMSE %.4f | mean %.4f | worst %.4f | diversity %.3f | evaluations %3d",
                    generation, stats.best_rmse, stats.mean_rmse, stats.worst_rmse, stats.diversity, evaluations)
        return stats


class RandomSearch:
    """Baseline: evaluates `budget` uniformly random genomes and keeps the best one.

    Given the same number of fitness evaluations as the GA, it shows how much the evolutionary
    operators actually contribute beyond "just trying many configurations".
    """

    def __init__(self, search_space: SearchSpace, fitness: IFitnessFunction, budget: int) -> None:
        self._space = search_space
        self._fitness = fitness
        self._budget = budget

    def run(self, seed: int) -> SearchResult:
        rng = np.random.default_rng(seed)
        best_genome, best_rmse = None, float("inf")
        best_so_far: List[float] = []
        for _ in range(self._budget):
            genome = self._space.random_genome(rng)
            rmse = -self._fitness.evaluate(genome)
            if rmse < best_rmse:
                best_genome, best_rmse = genome, rmse
            best_so_far.append(best_rmse)
        logger.info("  random search: best RMSE %.4f after %d evaluations", best_rmse, self._budget)
        return SearchResult(method="Random search", seed=seed, best_genome=best_genome,
                            best_val_rmse=best_rmse, best_so_far=tuple(best_so_far))
