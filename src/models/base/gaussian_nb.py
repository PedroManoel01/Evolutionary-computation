"""Modelo 1 do pool: Gaussian Naive Bayes (river.naive_bayes.GaussianNB)."""

from __future__ import annotations

from typing import Any

from river import naive_bayes


def create_gaussian_nb(**kwargs: Any) -> naive_bayes.GaussianNB:
    """Instancia um classificador GaussianNB incremental."""
    return naive_bayes.GaussianNB(**kwargs)
