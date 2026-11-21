"""Pool de modelos incrementais (M = 8) da biblioteca river.

Referência: Seção 4 de Proposta_Projeto_CE.pdf.
Todos os modelos são atualizados incrementalmente no esquema test-then-train.
"""

from __future__ import annotations

from typing import Dict, List

from river import base

from src.models.base import (
    create_amf,
    create_bagging_ht,
    create_efdt,
    create_gaussian_nb,
    create_hoeffding_adaptive_tree,
    create_hoeffding_tree,
    create_knn,
    create_softmax_regression,
)

MODEL_NAMES: List[str] = [
    "GaussianNB",
    "HoeffdingTree",
    "HoeffdingAdaptiveTree",
    "EFDT",
    "SoftmaxRegression",
    "KNN",
    "BaggingHT",
    "AMF",
]


def get_model_names() -> List[str]:
    """Retorna os nomes dos M = 8 modelos do pool na ordem padrão."""
    return list(MODEL_NAMES)


def build_model_pool(seed: int = 42) -> Dict[str, base.Classifier]:
    """Instancia o pool padrão de M = 8 classificadores online do river.

    Args:
        seed: Semente aleatória para os modelos estocásticos do pool.

    Returns:
        Dicionário ordenado {nome_do_modelo: classificador_river}.
    """
    return {
        "GaussianNB": create_gaussian_nb(),
        "HoeffdingTree": create_hoeffding_tree(),
        "HoeffdingAdaptiveTree": create_hoeffding_adaptive_tree(seed=seed),
        "EFDT": create_efdt(),
        "SoftmaxRegression": create_softmax_regression(),
        "KNN": create_knn(window_size=1000),
        "BaggingHT": create_bagging_ht(n_models=5, seed=seed),
        "AMF": create_amf(n_estimators=10, seed=seed),
    }
