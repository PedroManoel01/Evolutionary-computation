"""Benchmark 1: Adaptive Random Forest (river.forest.ARFClassifier)."""

from __future__ import annotations

from typing import Any

from river import forest


def create_arf(
    n_models: int = 10,
    seed: int = 42,
    **kwargs: Any,
) -> forest.ARFClassifier:
    """Instancia um ARFClassifier com 10 árvores por padrão."""
    return forest.ARFClassifier(n_models=n_models, seed=seed, **kwargs)
