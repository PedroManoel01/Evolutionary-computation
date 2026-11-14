"""Modelo 7 do pool: Bagging online com Hoeffding Trees (river.ensemble.BaggingClassifier)."""

from __future__ import annotations

from typing import Any

from river import ensemble, tree


def create_bagging_ht(
    n_models: int = 5,
    seed: int = 42,
    **kwargs: Any,
) -> ensemble.BaggingClassifier:
    """Instancia um BaggingClassifier com 5 Hoeffding Trees por padrão."""
    base_model = kwargs.pop("model", tree.HoeffdingTreeClassifier())
    return ensemble.BaggingClassifier(
        model=base_model,
        n_models=n_models,
        seed=seed,
        **kwargs,
    )
