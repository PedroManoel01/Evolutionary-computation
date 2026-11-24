"""Núcleo evolucionário: AG real no simplex, Evolução Diferencial (DE) e busca aleatória."""

from src.evolution.base import (
    FitnessFn,
    OptResult,
    mean_pairwise_distance,
    random_simplex,
)
from src.evolution.de import (
    DE_STRATEGIES,
    DEConfig,
    DifferentialEvolution,
)
from src.evolution.ga import (
    GAConfig,
    GeneticAlgorithm,
    RandomSearch,
)

__all__ = [
    "FitnessFn",
    "OptResult",
    "mean_pairwise_distance",
    "random_simplex",
    "GAConfig",
    "GeneticAlgorithm",
    "DEConfig",
    "DE_STRATEGIES",
    "DifferentialEvolution",
    "RandomSearch",
]
