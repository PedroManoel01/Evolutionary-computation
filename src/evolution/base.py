"""Tipos e utilitários compartilhados pelos otimizadores evolucionários no simplex Δ^M."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List

import numpy as np

FitnessFn = Callable[[np.ndarray], np.ndarray]


def random_simplex(n: int, m: int, rng: np.random.Generator) -> np.ndarray:
    """Amostragem uniforme no simplex: Dirichlet(1, ..., 1)."""
    return rng.dirichlet(np.ones(m), size=n)


def mean_pairwise_distance(pop: np.ndarray) -> float:
    """Diversidade: distância euclidiana média entre pares de indivíduos."""
    n = len(pop)
    if n < 2:
        return 0.0
    d = np.sqrt(((pop[:, None, :] - pop[None, :, :]) ** 2).sum(-1))
    return float(d[np.triu_indices(n, 1)].mean())


@dataclass
class OptResult:
    pop: np.ndarray            # população final (N, M)
    fit: np.ndarray            # fitness da população final (N,)
    best: np.ndarray           # melhor indivíduo (M,)
    best_fit: float
    history: List[float]       # melhor fitness por geração (gen 0 = população inicial)
    n_evals: int
    n_gens: int
    div_start: float
    div_end: float
    init_best_fit: float = field(default=0.0)
