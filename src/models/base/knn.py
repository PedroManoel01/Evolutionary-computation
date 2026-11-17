"""Modelo 6 do pool: KNN incremental com janela deslizante (river.neighbors.KNNClassifier)."""

from __future__ import annotations

from typing import Any

from river import neighbors


def create_knn(window_size: int = 1000, **kwargs: Any) -> neighbors.KNNClassifier:
    """Instancia um KNNClassifier com janela de tamanho padrão 1000."""
    engine = kwargs.pop("engine", neighbors.LazySearch(window_size=window_size))
    return neighbors.KNNClassifier(engine=engine, **kwargs)
