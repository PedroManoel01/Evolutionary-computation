"""Ensemble ponderado sobre o pool de modelos incrementais.

Referência: Seções 3 e 4 de Proposta_Projeto_CE.pdf.
Os modelos base nunca dependem dos pesos: são sempre atualizados via learn_one(x, y),
enquanto o ensemble combina suas probabilidades usando o vetor de pesos w no simplex.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
from river import base

from src.models.pool import build_model_pool


class WeightedEnsemble:
    """Ensemble online com ponderação externa sobre um pool de classificadores river."""

    def __init__(
        self,
        models: Optional[Dict[str, base.Classifier]] = None,
        weights: Optional[Sequence[float]] = None,
        seed: int = 42,
    ) -> None:
        self.models: Dict[str, base.Classifier] = (
            dict(models) if models is not None else build_model_pool(seed=seed)
        )
        self.model_names = list(self.models.keys())
        self.n_models = len(self.model_names)

        if self.n_models == 0:
            raise ValueError("O pool de modelos não pode estar vazio.")

        if weights is None:
            self.weights = np.full(self.n_models, 1.0 / self.n_models, dtype=float)
        else:
            self.set_weights(weights)

    def set_weights(self, weights: Sequence[float]) -> None:
        """Atualiza e normaliza o vetor de pesos w no simplex (soma 1, >= 0)."""
        w = np.asarray(weights, dtype=float)
        if w.shape != (self.n_models,):
            raise ValueError(
                f"Esperado vetor de pesos com tamanho {self.n_models}, recebido {w.shape}."
            )
        if np.any(w < 0):
            raise ValueError("Os pesos do ensemble devem ser não negativos.")
        total = float(w.sum())
        if total <= 0.0:
            self.weights = np.full(self.n_models, 1.0 / self.n_models, dtype=float)
        else:
            self.weights = w / total

    def get_weights(self) -> Dict[str, float]:
        """Retorna os pesos atuais indexados pelo nome de cada modelo."""
        return {
            name: float(w) for name, w in zip(self.model_names, self.weights)
        }

    def predict_proba_pool(self, x: Dict[str, Any]) -> Dict[str, Dict[Any, float]]:
        """Retorna as distribuições de probabilidade de cada modelo do pool."""
        return {
            name: model.predict_proba_one(x)
            for name, model in self.models.items()
        }

    def predict_proba_one(self, x: Dict[str, Any]) -> Dict[Any, float]:
        """Combina as probabilidades dos modelos segundo os pesos atuais."""
        combined: Dict[Any, float] = {}
        for weight, model in zip(self.weights, self.models.values()):
            if weight == 0.0:
                continue
            probs = model.predict_proba_one(x)
            for cls, prob in probs.items():
                combined[cls] = combined.get(cls, 0.0) + float(weight) * float(prob)
        return combined

    def predict_one(self, x: Dict[str, Any]) -> Any:
        """Retorna a classe com maior probabilidade ponderada."""
        probs = self.predict_proba_one(x)
        if not probs:
            return None
        return max(probs.items(), key=lambda item: item[1])[0]

    def learn_one(self, x: Dict[str, Any], y: Any) -> None:
        """Atualiza todos os modelos do pool incrementalmente (test-then-train)."""
        for model in self.models.values():
            model.learn_one(x, y)
