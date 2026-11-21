"""Unit tests for the Lab 3 genetic algorithm building blocks (no model training, no network)."""

import numpy as np
import pytest

from src.config.lab3_config import GeneticAlgorithmConfig, default_search_space
from src.evolution.fitness import IFitnessFunction
from src.evolution.genetic_algorithm import GeneticAlgorithm, RandomSearch
from src.evolution.operators import GaussianMutation, TournamentSelection, UniformCrossover
from src.evolution.search_space import HyperparameterSpec, ParameterKind, SearchSpace


class SphereFitness(IFitnessFunction):
    """Toy fitness with a known optimum at 0.7 in every gene."""

    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, genome: np.ndarray) -> float:
        self.calls += 1
        return -float(np.sum((genome - 0.7) ** 2))


def test_decode_maps_gene_bounds_to_range_bounds():
    spec = HyperparameterSpec("max_iter", ParameterKind.INT, 50, 500)
    assert spec.decode(0.0) == 50
    assert spec.decode(1.0) == 500
    assert isinstance(spec.decode(0.5), int)


def test_decode_log_scale_midpoint_is_geometric_mean():
    spec = HyperparameterSpec("lr", ParameterKind.FLOAT, 0.01, 1.0, log_scale=True)
    assert spec.decode(0.5) == pytest.approx(0.1)


def test_decode_clips_genes_outside_unit_interval():
    spec = HyperparameterSpec("x", ParameterKind.FLOAT, 2.0, 4.0)
    assert spec.decode(-3.0) == 2.0
    assert spec.decode(7.0) == 4.0


def test_search_space_rejects_wrong_genome_length():
    with pytest.raises(ValueError):
        default_search_space().decode(np.zeros(2))


def test_tournament_of_whole_population_returns_the_fittest():
    population = [np.full(3, value) for value in (0.1, 0.2, 0.3, 0.4)]
    winner = TournamentSelection(4).select(population, [1.0, 5.0, 3.0, 2.0], np.random.default_rng(0))
    assert np.allclose(winner, 0.2)


def test_uniform_crossover_children_take_every_gene_from_a_parent():
    parent_a, parent_b = np.zeros(50), np.ones(50)
    child_a, child_b = UniformCrossover(1.0).recombine(parent_a, parent_b, np.random.default_rng(0))
    assert set(np.unique(child_a)) == {0.0, 1.0}  # genes are mixed
    assert np.allclose(child_a + child_b, 1.0)    # children are complementary


def test_crossover_with_zero_probability_copies_parents():
    parent_a, parent_b = np.zeros(5), np.ones(5)
    child_a, child_b = UniformCrossover(0.0).recombine(parent_a, parent_b, np.random.default_rng(0))
    assert np.array_equal(child_a, parent_a) and np.array_equal(child_b, parent_b)


def test_mutation_keeps_genes_within_bounds_and_does_not_modify_the_input():
    genome = np.array([0.0, 1.0, 0.5, 0.01, 0.99])
    original = genome.copy()
    mutated = GaussianMutation(probability=1.0, sigma=0.5).mutate(genome, np.random.default_rng(0))
    assert np.array_equal(genome, original)
    assert mutated.min() >= 0.0 and mutated.max() <= 1.0
    assert not np.array_equal(mutated, original)


def test_mutation_with_zero_probability_changes_nothing():
    genome = np.array([0.2, 0.4])
    assert np.array_equal(GaussianMutation(0.0, 0.5).mutate(genome, np.random.default_rng(0)), genome)


def _toy_ga(fitness: IFitnessFunction, config: GeneticAlgorithmConfig) -> GeneticAlgorithm:
    space = SearchSpace(tuple(HyperparameterSpec(f"x{i}", ParameterKind.FLOAT, 0.0, 1.0) for i in range(4)))
    return GeneticAlgorithm(space, fitness, TournamentSelection(config.tournament_size),
                            UniformCrossover(config.crossover_probability),
                            GaussianMutation(config.mutation_probability, config.mutation_sigma), config)


def test_ga_uses_exactly_the_evaluation_budget_and_elitism_never_loses_the_best():
    config = GeneticAlgorithmConfig(population_size=10, generations=8, elite_count=2)
    fitness = SphereFitness()
    result = _toy_ga(fitness, config).run(seed=0)
    assert fitness.calls == config.evaluation_budget == len(result.best_so_far)
    best_per_generation = [g.best_rmse for g in result.generations]
    assert all(later <= earlier for earlier, later in zip(best_per_generation, best_per_generation[1:]))
    assert result.best_val_rmse == pytest.approx(best_per_generation[-1])


def test_ga_is_reproducible_and_improves_over_the_initial_population():
    config = GeneticAlgorithmConfig(population_size=12, generations=15)
    first = _toy_ga(SphereFitness(), config).run(seed=5)
    second = _toy_ga(SphereFitness(), config).run(seed=5)
    assert np.array_equal(first.best_genome, second.best_genome)
    assert first.generations[-1].best_rmse < first.generations[0].best_rmse
    assert first.generations[-1].diversity < first.generations[0].diversity


def test_random_search_best_so_far_is_monotone():
    space = SearchSpace(tuple(HyperparameterSpec(f"x{i}", ParameterKind.FLOAT, 0.0, 1.0) for i in range(4)))
    result = RandomSearch(space, SphereFitness(), budget=30).run(seed=1)
    assert len(result.best_so_far) == 30
    assert all(b <= a for a, b in zip(result.best_so_far, result.best_so_far[1:]))
