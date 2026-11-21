"""Benchmarks estado da arte em ensembles para data streams (river).

Referência: Seção 9.2 de Proposta_Projeto_CE.pdf.
"""

from __future__ import annotations

from typing import Dict, List

from river import base

from src.models.benchmarks.arf import create_arf
from src.models.benchmarks.leveraging_bagging import create_leveraging_bagging
from src.models.benchmarks.srp import create_srp

BENCHMARK_NAMES: List[str] = [
    "ARF",
    "SRP",
    "LeveragingBagging",
]


def get_benchmark_names() -> List[str]:
    """Retorna os nomes dos modelos de benchmark na ordem padrão."""
    return list(BENCHMARK_NAMES)


def build_benchmarks(seed: int = 42) -> Dict[str, base.Classifier]:
    """Instancia os benchmarks da literatura no river.

    Args:
        seed: Semente aleatória para reprodutibilidade.

    Returns:
        Dicionário ordenado {nome_do_benchmark: classificador_river}.
    """
    return {
        "ARF": create_arf(seed=seed),
        "SRP": create_srp(seed=seed),
        "LeveragingBagging": create_leveraging_bagging(seed=seed),
    }


__all__ = [
    "BENCHMARK_NAMES",
    "get_benchmark_names",
    "build_benchmarks",
    "create_arf",
    "create_srp",
    "create_leveraging_bagging",
]
