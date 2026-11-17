"""Estratégias pós-drift S1..S6 (Seção 7 da proposta).

Cada estratégia monta a população inicial da nova otimização a partir da população final
anterior (``prev_pop`` com sua fitness antiga ``prev_fit``) e, em S5/S6, de um arquivo de
memória. O elitismo DENTRO do AG está sempre ativo; aqui varia só a população inicial.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from src.evolution.ga import random_simplex

STRATEGIES = {
    "S1": "Reinício total",
    "S2": "Warm start",
    "S3": "Warm start + imigrantes",
    "S4": "Warm start + hipermutação",
    "S5": "Memória",
    "S6": "Memória + imigrantes",
}


@dataclass
class MemoryArchive:
    """Arquivo de memória (Branke, 1999): até ``capacity`` indivíduos.

    Com o arquivo cheio, o novo indivíduo substitui o mais parecido com ele
    (menor distância euclidiana), mantendo o arquivo diverso.
    """

    capacity: int = 10
    items: List[np.ndarray] = field(default_factory=list)

    def insert(self, w: np.ndarray) -> None:
        w = np.asarray(w, dtype=float).copy()
        if len(self.items) < self.capacity:
            self.items.append(w)
            return
        d = [float(np.linalg.norm(m - w)) for m in self.items]
        self.items[int(np.argmin(d))] = w

    def as_array(self, m: int) -> np.ndarray:
        return np.array(self.items) if self.items else np.empty((0, m))


@dataclass
class PostDriftStrategy:
    name: str = "S1"
    pop_size: int = 30
    immigrant_rate: float = 0.3
    memory_capacity: int = 10
    memory: MemoryArchive = field(init=False)

    def __post_init__(self) -> None:
        if self.name not in STRATEGIES:
            raise ValueError(f"Estratégia desconhecida: {self.name}")
        self.memory = MemoryArchive(self.memory_capacity)

    @property
    def uses_memory(self) -> bool:
        return self.name in {"S5", "S6"}

    @property
    def hypermutation(self) -> bool:
        return self.name == "S4"

    def on_drift(self, current_best: Optional[np.ndarray]) -> None:
        """Chamado no drift, antes de reotimizar: guarda o melhor do conceito que terminou."""
        if self.uses_memory and current_best is not None:
            self.memory.insert(current_best)

    def initial_population(self, prev_pop: Optional[np.ndarray], prev_fit: Optional[np.ndarray],
                           m: int, rng: np.random.Generator) -> np.ndarray:
        N = self.pop_size
        if prev_pop is None or self.name == "S1":
            return random_simplex(N, m, rng)

        order = np.argsort(-prev_fit, kind="stable")  # melhores primeiro
        prev_sorted = prev_pop[order]
        n_imm = int(round(self.immigrant_rate * N))

        if self.name in {"S2", "S4"}:
            return prev_sorted[:N].copy()
        if self.name == "S3":
            keep = prev_sorted[: N - n_imm]
            return np.concatenate([keep, random_simplex(n_imm, m, rng)])

        mem = self.memory.as_array(m)
        if self.name == "S5":
            rest = prev_sorted[: N - len(mem)]
            return np.concatenate([mem, rest])
        # S6
        n_rest = max(N - len(mem) - n_imm, 0)
        parts = [mem, random_simplex(n_imm, m, rng), prev_sorted[:n_rest]]
        return np.concatenate(parts)[:N]
