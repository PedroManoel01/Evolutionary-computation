"""Modelo 4 do pool: Extremely Fast Decision Tree (river.tree.ExtremelyFastDecisionTreeClassifier)."""

from __future__ import annotations

from typing import Any

from river import tree


def create_efdt(**kwargs: Any) -> tree.ExtremelyFastDecisionTreeClassifier:
    """Instancia um ExtremelyFastDecisionTreeClassifier (revisa splits)."""
    return tree.ExtremelyFastDecisionTreeClassifier(**kwargs)
