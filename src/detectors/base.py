"""Interface base para os detectores de concept drift."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseDriftDetector(ABC):
    """Interface comum para detectores que monitoram o erro 0/1 do ensemble."""

    @property
    @abstractmethod
    def warning_detected(self) -> bool:
        """Indica se o detector está em estado de warning."""

    @property
    @abstractmethod
    def drift_detected(self) -> bool:
        """Indica se um drift foi confirmado na última atualização."""

    @abstractmethod
    def update(self, value: int | float) -> None:
        """Atualiza o detector com uma nova observação (ex.: erro 0/1)."""

    @abstractmethod
    def reset(self) -> None:
        """Reinicializa o detector após a confirmação de um drift."""
