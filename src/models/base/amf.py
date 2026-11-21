"""Modelo 8 do pool: Aggregated Mondrian Forest (river.forest.AMFClassifier)."""

from __future__ import annotations

from typing import Any

from river import forest


def create_amf(
    n_estimators: int = 10,
    seed: int = 42,
    **kwargs: Any,
) -> forest.AMFClassifier:
    """Instancia um AMFClassifier (Floresta de Mondrian online)."""
    return forest.AMFClassifier(n_estimators=n_estimators, seed=seed, **kwargs)
