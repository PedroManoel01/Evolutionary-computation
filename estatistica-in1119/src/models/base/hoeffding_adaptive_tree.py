"""Modelo 3 do pool: Hoeffding Adaptive Tree (river.tree.HoeffdingAdaptiveTreeClassifier)."""

from __future__ import annotations

from typing import Any

from river import tree


def create_hoeffding_adaptive_tree(
    seed: int = 42, **kwargs: Any
) -> tree.HoeffdingAdaptiveTreeClassifier:
    """Instancia um HoeffdingAdaptiveTreeClassifier com adaptação interna a drift."""
    return tree.HoeffdingAdaptiveTreeClassifier(seed=seed, **kwargs)
