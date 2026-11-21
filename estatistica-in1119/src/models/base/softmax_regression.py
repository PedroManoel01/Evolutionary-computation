"""Modelo 5 do pool: StandardScaler + SoftmaxRegression (river)."""

from __future__ import annotations

from typing import Any, Optional

from river import compose, linear_model, preprocessing


def create_softmax_regression(
    scaler_kwargs: Optional[dict[str, Any]] = None,
    **kwargs: Any,
) -> compose.Pipeline:
    """Instancia um pipeline StandardScaler -> SoftmaxRegression."""
    scaler = preprocessing.StandardScaler(**(scaler_kwargs or {}))
    clf = linear_model.SoftmaxRegression(**kwargs)
    return scaler | clf
