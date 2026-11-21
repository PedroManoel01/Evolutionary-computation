"""Detector principal: Par de ADWINs (warning + drift).

Referência: Seção 6.1 de Proposta_Projeto_CE.pdf.
Como o ADWIN não possui nível de warning nativo, utiliza-se um segundo ADWIN
mais sensível (delta_w = 0.01) para warning e outro (delta_d = 0.001) para drift.
"""

from __future__ import annotations

from typing import Any

from river import drift

from src.detectors.base import BaseDriftDetector


class ADWINPairDetector(BaseDriftDetector):
    """Par de detectores ADWIN para sinalização de warning e confirmação de drift."""

    def __init__(
        self,
        delta_warning: float = 0.01,
        delta_drift: float = 0.001,
        **kwargs: Any,
    ) -> None:
        self.delta_warning = delta_warning
        self.delta_drift = delta_drift
        self._kwargs = dict(kwargs)
        self._in_warning = False
        self._drift_detected = False
        self._init_detectors()

    def _init_detectors(self) -> None:
        self.warning_detector = drift.ADWIN(delta=self.delta_warning, **self._kwargs)
        self.drift_detector = drift.ADWIN(delta=self.delta_drift, **self._kwargs)

    @property
    def warning_detected(self) -> bool:
        return self._in_warning and not self._drift_detected

    @property
    def drift_detected(self) -> bool:
        return self._drift_detected

    def update(self, value: int | float) -> None:
        self.warning_detector.update(value)
        self.drift_detector.update(value)

        if self.warning_detector.drift_detected:
            self._in_warning = True

        self._drift_detected = bool(self.drift_detector.drift_detected)

    def reset(self) -> None:
        self._init_detectors()
        self._in_warning = False
        self._drift_detected = False


def create_adwin_pair(
    delta_warning: float = 0.01,
    delta_drift: float = 0.001,
    **kwargs: Any,
) -> ADWINPairDetector:
    """Instancia o par de ADWINs padrão do projeto."""
    return ADWINPairDetector(
        delta_warning=delta_warning,
        delta_drift=delta_drift,
        **kwargs,
    )
