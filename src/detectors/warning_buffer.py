"""Buffer do warning para coleta de dados de avaliação pós-drift.

Referência: Seção 6.2 de Proposta_Projeto_CE.pdf.
- No warning: guarda (P_t, y_t), limitado às B_max = 2000 instâncias mais recentes.
- Warning falso (saída do estado de warning ou B_max passos sem drift): descarta buffer.
- No drift: E = buffer U instâncias pós-drift até |E| >= n_min = 100.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Tuple

import numpy as np


class WarningBuffer:
    """Armazena pares (P_t, y_t) entre o warning e a confirmação do drift."""

    def __init__(self, max_size: int = 2000, min_eval_size: int = 100) -> None:
        self.max_size = max_size
        self.min_eval_size = min_eval_size
        self._buffer: Deque[Tuple[np.ndarray, Any]] = deque(maxlen=max_size)
        self._steps_in_warning = 0
        self._awaiting_post_drift = False

    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def awaiting_post_drift(self) -> bool:
        """True quando o drift foi confirmado, mas ainda faltam instâncias para n_min."""
        return self._awaiting_post_drift

    def add_warning_sample(self, probs: np.ndarray, y: Any) -> bool:
        """Adiciona uma instância durante o estado de warning.

        Returns:
            True se o buffer foi descartado por exceder max_size passos sem drift.
        """
        if self._awaiting_post_drift:
            return False

        self._steps_in_warning += 1
        if self._steps_in_warning > self.max_size:
            self.clear()
            return True

        self._buffer.append((np.asarray(probs, dtype=float), y))
        return False

    def on_false_warning(self) -> None:
        """Descarta o buffer quando o detector sai do estado de warning sem drift."""
        if not self._awaiting_post_drift and len(self._buffer) > 0:
            self.clear()

    def on_drift(self, probs: np.ndarray | None = None, y: Any = None) -> None:
        """Marca confirmação do drift e, se fornecida, inclui a instância atual."""
        self._awaiting_post_drift = True
        if probs is not None:
            self._buffer.append((np.asarray(probs, dtype=float), y))

    def add_post_drift_sample(self, probs: np.ndarray, y: Any) -> None:
        """Acumula instâncias pós-drift enquanto |E| < min_eval_size."""
        if self._awaiting_post_drift:
            self._buffer.append((np.asarray(probs, dtype=float), y))

    def is_ready(self) -> bool:
        """True quando o drift foi confirmado e |E| >= min_eval_size."""
        return self._awaiting_post_drift and len(self._buffer) >= self.min_eval_size

    def get_evaluation_set(self) -> Tuple[np.ndarray, np.ndarray]:
        """Retorna (P, y) com P de shape (n, M, C) e y de shape (n,)."""
        if not self._buffer:
            raise ValueError("O buffer de avaliação está vazio.")
        probs_list = [item[0] for item in self._buffer]
        labels_list = [item[1] for item in self._buffer]
        return np.stack(probs_list, axis=0), np.asarray(labels_list)

    def clear(self) -> None:
        """Limpa o buffer e reinicia os contadores internos."""
        self._buffer.clear()
        self._steps_in_warning = 0
        self._awaiting_post_drift = False
