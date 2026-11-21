"""Benchmark 3: Leveraging Bagging (river.ensemble.LeveragingBaggingClassifier)."""

from __future__ import annotations

from typing import Any

from river import ensemble, tree


def create_leveraging_bagging(
    n_models: int = 10,
    seed: int = 42,
    **kwargs: Any,
) -> ensemble.LeveragingBaggingClassifier:
    """Instancia um LeveragingBaggingClassifier com 10 Hoeffding Trees por padrão."""
    base_model = kwargs.pop("model", tree.HoeffdingTreeClassifier())
    return ensemble.LeveragingBaggingClassifier(
        model=base_model,
        n_models=n_models,
        seed=seed,
        **kwargs,
    )
