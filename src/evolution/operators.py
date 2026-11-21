"""Evolutionary operators: selection, crossover and mutation.

Each operator is a small class behind an interface (Strategy pattern), so it can be unit-tested
in isolation and swapped without touching the genetic algorithm itself. All operators work on
genomes normalized to [0, 1] and take the random generator as a parameter (reproducibility).
"""

from abc import ABC, abstractmethod
from typing import Sequence, Tuple

import numpy as np


class ISelection(ABC):
    @abstractmethod
    def select(self, population: Sequence[np.ndarray], fitness: Sequence[float],
               rng: np.random.Generator) -> np.ndarray:
        """Returns one parent chosen from the population (higher fitness = better)."""


class ICrossover(ABC):
    @abstractmethod
    def recombine(self, parent_a: np.ndarray, parent_b: np.ndarray,
                  rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        """Returns two children built from two parents."""


class IMutation(ABC):
    @abstractmethod
    def mutate(self, genome: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Returns a mutated copy of the genome."""


class TournamentSelection(ISelection):
    """Picks `tournament_size` random individuals and returns the fittest of them.

    Selection pressure is controlled by the tournament size and does not depend on the scale of
    fitness values (unlike roulette-wheel selection), which matters for RMSE-based fitness where
    individuals differ only in the 2nd-3rd decimal place.
    """

    def __init__(self, tournament_size: int) -> None:
        if tournament_size < 1:
            raise ValueError("tournament_size must be >= 1")
        self._tournament_size = tournament_size

    def select(self, population: Sequence[np.ndarray], fitness: Sequence[float],
               rng: np.random.Generator) -> np.ndarray:
        size = min(self._tournament_size, len(population))
        contenders = rng.choice(len(population), size=size, replace=False)
        # max(..., key=...) is Java's Collections.max with a Comparator
        winner = max(contenders, key=lambda index: fitness[index])
        return population[winner].copy()


class UniformCrossover(ICrossover):
    """Every gene is taken from either parent with probability 0.5 (the second child gets the other one).

    Hyperparameters have no meaningful order in the genome, so position-based operators
    (one-point, two-point) would introduce an arbitrary linkage between neighbouring genes.
    """

    def __init__(self, probability: float) -> None:
        self._probability = probability

    def recombine(self, parent_a: np.ndarray, parent_b: np.ndarray,
                  rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        if rng.random() >= self._probability:
            return parent_a.copy(), parent_b.copy()
        take_from_a = rng.random(len(parent_a)) < 0.5  # boolean mask, one flag per gene
        child_a = np.where(take_from_a, parent_a, parent_b)
        child_b = np.where(take_from_a, parent_b, parent_a)
        return child_a, child_b


class GaussianMutation(IMutation):
    """Adds Gaussian noise N(0, sigma) to each gene with the given probability, clipped to [0, 1].

    Small steps are frequent and large steps are rare: the algorithm mostly refines good
    solutions but can still escape a local optimum.
    """

    def __init__(self, probability: float, sigma: float) -> None:
        self._probability = probability
        self._sigma = sigma

    def mutate(self, genome: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        mutated = genome.copy()
        mask = rng.random(len(genome)) < self._probability
        noise = rng.normal(0.0, self._sigma, size=len(genome))
        mutated[mask] += noise[mask]
        return np.clip(mutated, 0.0, 1.0)
