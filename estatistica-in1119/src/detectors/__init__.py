"""Detectores de concept drift e buffer de warning."""

from __future__ import annotations

from typing import Any

from src.detectors.adwin_pair import ADWINPairDetector, create_adwin_pair
from src.detectors.base import BaseDriftDetector
from src.detectors.ddm import DDMDetector, create_ddm
from src.detectors.oracle import OracleDetector, create_oracle
from src.detectors.warning_buffer import WarningBuffer


def build_detector(name: str, **kwargs: Any) -> BaseDriftDetector:
    """Fábrica unificada para instanciar detectores pelo nome.

    Args:
        name: Um de {'adwin_pair', 'ddm', 'oracle'}.
        **kwargs: Parâmetros repassados ao detector escolhido.
    """
    key = name.lower()
    if key in {"adwin_pair", "adwin"}:
        return create_adwin_pair(**kwargs)
    if key == "ddm":
        return create_ddm(**kwargs)
    if key == "oracle":
        return create_oracle(**kwargs)
    raise ValueError(
        f"Detector desconhecido: '{name}'. Use 'adwin_pair', 'ddm' ou 'oracle'."
    )


__all__ = [
    "BaseDriftDetector",
    "ADWINPairDetector",
    "DDMDetector",
    "OracleDetector",
    "WarningBuffer",
    "create_adwin_pair",
    "create_ddm",
    "create_oracle",
    "build_detector",
]
