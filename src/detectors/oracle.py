"""Detector Oráculo para bases com posições de drift conhecidas.

Referência: Seção 6.1 de Proposta_Projeto_CE.pdf.
Utilizado como controle e limite superior em dados sintéticos (e INSECTS).
"""

from __future__ import annotations

from typing import Iterable, Set

from src.detectors.base import BaseDriftDetector


class OracleDetector(BaseDriftDetector):
    """Sinaliza drift exatamente nas posições reais informadas."""

    def __init__(
        self,
        drift_positions: Iterable[int],
        warning_window: int = 0,
    ) -> None:
        self.drift_positions: Set[int] = set(int(p) for p in drift_positions)
        self.warning_window = int(warning_window)
        self._step = -1
        self._warning_detected = False
        self._drift_detected = False

    @property
    def warning_detected(self) -> bool:
        return self._warning_detected and not self._drift_detected

    @property
    def drift_detected(self) -> bool:
        return self._drift_detected

    def update(self, value: int | float) -> None:  # noqa: ARG002
        self._step += 1
        self._drift_detected = self._step in self.drift_positions

        if self.warning_window > 0 and not self._drift_detected:
            self._warning_detected = any(
                (pos - self.warning_window) <= self._step < pos
                for pos in self.drift_positions
            )
        else:
            self._warning_detected = False

    def reset(self) -> None:
        """Limpa apenas as flags atuais, preservando o contador de instâncias."""
        self._warning_detected = False
        self._drift_detected = False


def create_oracle(
    drift_positions: Iterable[int],
    warning_window: int = 0,
) -> OracleDetector:
    """Instancia o detector oráculo a partir das posições conhecidas de drift."""
    return OracleDetector(
        drift_positions=drift_positions,
        warning_window=warning_window,
    )
