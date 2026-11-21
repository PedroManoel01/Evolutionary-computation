"""Modelo 2 do pool: Hoeffding Tree (river.tree.HoeffdingTreeClassifier)."""

from __future__ import annotations

from typing import Any

from river import tree


def create_hoeffding_tree(**kwargs: Any) -> tree.HoeffdingTreeClassifier:
    """Instancia um HoeffdingTreeClassifier incremental."""
    return tree.HoeffdingTreeClassifier(**kwargs)
