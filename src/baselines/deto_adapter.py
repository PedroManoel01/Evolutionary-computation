"""Adaptador do DETO (Dynamic Evolutionary / Transfer Bayesian Optimization)
para ponderação de ensembles no simplex Δ^M.
"""

from __future__ import annotations

import contextlib
import io
import sys
import warnings
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from src.evaluation.objectives import repair_simplex

_DETO_ROOT = str(Path(__file__).resolve().parent / "deto")
if _DETO_ROOT not in sys.path:
    sys.path.append(_DETO_ROOT)


class DETOAdapter:
    """Envolve o modelo DETO com transferência entre ambientes (drifts)."""

    def __init__(
        self,
        n_models: int = 8,
        n_init: int = 16,
        n_iter: int = 20,
        seed: int = 42,
    ) -> None:
        self.n_models = n_models
        self.n_init = n_init
        self.n_iter = n_iter
        self.seed = seed
        self.source_datasets: Dict[int, object] = {}
        self._env_idx = 0

    def optimize(
        self,
        objective_fn: Callable[[np.ndarray], np.ndarray],
    ) -> Tuple[np.ndarray, float]:
        """Executa o DETO no ambiente atual e registra os dados para o próximo drift.

        Args:
            objective_fn: Função de minimização `f(W) -> (N,)` (ex.: `1.0 - macro_f1`).

        Returns:
            Tupla `(melhores_pesos_simplex, melhor_macro_f1)`.
        """
        import GPy
        from emukit.bayesian_optimization.acquisitions import (
            NegativeLowerConfidenceBound as UCB,
        )
        from emukit.core import ContinuousParameter, ParameterSpace
        from emukit.core.optimization import GradientAcquisitionOptimizer
        from models import DETO, RBO, TaskData, WrapperBase

        np.random.seed(self.seed + self._env_idx)

        space = ParameterSpace(
            [ContinuousParameter(f"w_{i}", 0.0, 1.0) for i in range(self.n_models)]
        )
        kernel = GPy.kern.RBF(input_dim=self.n_models, ARD=True)

        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            if self.source_datasets:
                model = DETO(kernel=kernel, noise_variance=1e-4, normalize=True)
                model.meta_fit(self.source_datasets)
                X_init = model.meta_initial(self.n_init)
            else:
                model = RBO(kernel=kernel, noise_variance=1e-4, normalize=True)
                X_init = None

            if X_init is None:
                X_init = space.sample_uniform(self.n_init)

            Y_init = np.atleast_2d(objective_fn(X_init)).reshape(-1, 1)
            X, Y = X_init, Y_init

            emukit_model = WrapperBase(model)
            for _ in range(self.n_iter):
                try:
                    model.fit(TaskData(X, Y), optimize=True)
                except np.linalg.LinAlgError:
                    break
                af = UCB(emukit_model, beta=np.float64(3.0))
                optimizer = GradientAcquisitionOptimizer(space)
                try:
                    X_new, _ = optimizer.optimize(af)
                except np.linalg.LinAlgError:
                    break
                Y_new = np.atleast_2d(objective_fn(X_new)).reshape(-1, 1)
                X = np.vstack([X, X_new])
                Y = np.vstack([Y, Y_new])

            self.source_datasets[self._env_idx] = TaskData(X, Y)
            self._env_idx += 1

        best_idx = int(np.argmin(Y[:, 0]))
        best_weights = repair_simplex(X[best_idx])
        best_macro_f1 = 1.0 - float(Y[best_idx, 0])
        return best_weights, best_macro_f1
