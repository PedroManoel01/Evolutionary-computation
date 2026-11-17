"""Adaptador do DASE mantendo seu detector (HCDD) e otimizador (DE + RBFN) originais.

Usa diretamente `initial_archives` e `DASE.run(..., true_drift=None)` de
`src/baselines/dase/src/DASE.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np

from src.evaluation.objectives import repair_simplex

_DASE_SRC = str(Path(__file__).resolve().parent / "dase" / "src")


def _ensure_dase_imports() -> None:
    if _DASE_SRC in sys.path:
        sys.path.remove(_DASE_SRC)
    sys.path.insert(0, _DASE_SRC)
    existing_utils = sys.modules.get("utils")
    if existing_utils is not None and _DASE_SRC not in str(getattr(existing_utils, "__file__", "")):
        sys.modules.pop("utils", None)


class DASEAdapter:
    """Executa o DASE original com HCDD + RBFN + DE.current_to_best_1."""

    def __init__(
        self,
        n_models: int = 8,
        chunk_size: int = 50,
        pop_size: int = 50,
        max_iter: int = 30,
        seed: int = 42,
    ) -> None:
        _ensure_dase_imports()
        from configuration import initial_parameters
        from detector import HCDD

        self.n_models = n_models
        self.seed = seed
        self.lb = np.zeros(n_models, dtype=float)
        self.ub = np.ones(n_models, dtype=float)
        self.config = initial_parameters(
            d=n_models,
            cs=chunk_size,
            pop_size=pop_size,
            max_iter=max_iter,
        )
        self.detector = HCDD(beta=self.config["beta"])
        self.env_archive = None
        self.cur_window = None
        self._step_idx = 0

    def step(
        self,
        objective_fn: Callable[[np.ndarray], np.ndarray],
    ) -> Tuple[np.ndarray, float, bool]:
        """Executa um passo (chunk) do DASE original com detecção própria via HCDD.

        Args:
            objective_fn: Função de minimização `f(W) -> (N,)` (ex.: `1.0 - macro_f1`).

        Returns:
            Tupla `(melhores_pesos_simplex, macro_f1_do_melhor, drift_detectado_hcdd)`.
        """
        _ensure_dase_imports()
        from DASE import initial_archives, run as dase_run
        from sampling import lhs

        np.random.seed(self.seed + self._step_idx)
        self._step_idx += 1

        # Evita divisão por zero em DASE.calculate_metrics: (y - y_pred) / y
        def _positive_obj(W: np.ndarray) -> np.ndarray:
            return np.asarray(objective_fn(W), dtype=float) + 1.0

        n_samples = int(self.config["cs"])
        X_chunk = lhs(n_samples, self.n_models, self.lb, self.ub)
        Y_chunk = _positive_obj(X_chunk).reshape(-1, 1)
        samples = np.hstack([X_chunk, Y_chunk])

        if self.env_archive is None:
            self.env_archive = initial_archives(
                samples=samples,
                lb=self.lb,
                ub=self.ub,
                config=self.config,
            )
            best_raw = self.env_archive[-1]["solutions"][0]
            drift_detected = False
        else:
            iter_best, self.env_archive, self.cur_window, self.detector, drift_detected = dase_run(
                samples=samples,
                lb=self.lb,
                ub=self.ub,
                env_archive=self.env_archive,
                config=self.config,
                detector=self.detector,
                cur_window=self.cur_window,
                true_drift=None,
            )
            best_raw = iter_best[-1]

        best_weights = repair_simplex(best_raw)
        best_macro_f1 = 1.0 - float(objective_fn(best_weights))
        return best_weights, best_macro_f1, bool(drift_detected)
