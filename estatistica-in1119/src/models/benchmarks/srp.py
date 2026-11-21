"""Benchmark 2: Streaming Random Patches (river.ensemble.SRPClassifier)."""

from __future__ import annotations

from typing import Any

from river import ensemble


def create_srp(
    n_models: int = 10,
    seed: int = 42,
    **kwargs: Any,
) -> ensemble.SRPClassifier:
    """Instancia um SRPClassifier com 10 modelos por padrão."""
    return ensemble.SRPClassifier(n_models=n_models, seed=seed, **kwargs)
