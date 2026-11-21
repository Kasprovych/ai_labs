"""Search space of hyperparameters and the genotype <-> phenotype mapping.

Genotype: a NumPy vector with one gene per hyperparameter, every gene normalized to [0, 1].
Phenotype: a dict of real hyperparameter values that can be passed to a scikit-learn model.
Normalizing genes lets the genetic operators treat all hyperparameters uniformly, no matter
whether they are integers, floats, or live on a logarithmic scale.
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Tuple, Union

import numpy as np

HyperparameterValue = Union[int, float]


class ParameterKind(Enum):
    INT = "int"
    FLOAT = "float"


@dataclass(frozen=True)
class HyperparameterSpec:
    """Description of one hyperparameter: name, type and allowed range."""
    name: str
    kind: ParameterKind
    low: float
    high: float
    # Log scale: equal gene steps multiply the value (0.01 -> 0.1 -> 1) instead of adding to it
    log_scale: bool = False

    def decode(self, gene: float) -> HyperparameterValue:
        """Maps a gene from [0, 1] to the real hyperparameter value."""
        gene = min(max(gene, 0.0), 1.0)
        if self.log_scale:
            value = math.exp(math.log(self.low) + gene * (math.log(self.high) - math.log(self.low)))
        else:
            value = self.low + gene * (self.high - self.low)
        if self.kind is ParameterKind.INT:
            return int(round(value))
        return float(value)

    @property
    def range_label(self) -> str:
        scale = ", log scale" if self.log_scale else ""
        return f"[{self.low:g}, {self.high:g}]{scale}"


@dataclass(frozen=True)
class SearchSpace:
    """Ordered collection of hyperparameter specs; gene i corresponds to specs[i]."""
    specs: Tuple[HyperparameterSpec, ...]

    @property
    def dimension(self) -> int:
        return len(self.specs)

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(spec.name for spec in self.specs)

    def decode(self, genome: np.ndarray) -> Dict[str, HyperparameterValue]:
        """Genotype -> phenotype."""
        if len(genome) != self.dimension:
            raise ValueError(f"Genome has {len(genome)} genes, expected {self.dimension}")
        # zip pairs each spec with its gene (like iterating two lists in parallel in Java)
        return {spec.name: spec.decode(float(gene)) for spec, gene in zip(self.specs, genome)}

    def random_genome(self, rng: np.random.Generator) -> np.ndarray:
        """Uniformly random individual."""
        return rng.random(self.dimension)
