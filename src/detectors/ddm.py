"""Detector comparativo: DDM (Drift Detection Method).

Referência: Seção 6.1 de Proposta_Projeto_CE.pdf.
Usa river.drift.binary.DDM com limiares padrão e estado de warning nativo.
"""

from __future__ import annotations

from typing import Any

from river import drift

from src.detectors.base import BaseDriftDetector


class DDMDetector(BaseDriftDetector):
    """Wrapper para river.drift.binary.DDM."""

    def __init__(self, **kwargs: Any) -> None:
        self._kwargs = dict(kwargs)
        self._detector = drift.binary.DDM(**self._kwargs)

    @property
    def warning_detected(self) -> bool:
        return bool(self._detector.warning_detected)

    @property
    def drift_detected(self) -> bool:
        return bool(self._detector.drift_detected)

    def update(self, value: int | float) -> None:
        self._detector.update(int(value))

    def reset(self) -> None:
        self._detector = drift.binary.DDM(**self._kwargs)


def create_ddm(**kwargs: Any) -> DDMDetector:
    """Instancia o detector DDM com limiares padrão do river."""
    return DDMDetector(**kwargs)
