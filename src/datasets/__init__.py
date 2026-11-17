"""Datasets e streams do projeto disponíveis no river."""

from __future__ import annotations

from typing import List

from src.datasets.base import StreamDataset, chain_concept_drift_streams
from src.datasets.real import create_credit_card, create_elec2
from src.datasets.synthetic import (
    create_agr_abr,
    create_agr_grad,
    create_rbf_inc,
    create_sea_rec,
    create_stagger_rec,
)

SYNTHETIC_DATASET_NAMES: List[str] = [
    "SEA-Rec",
    "STAGGER-Rec",
    "AGR-Abr",
    "AGR-Grad",
    "RBF-Inc",
]

REAL_RIVER_DATASET_NAMES: List[str] = [
    "Elec2",
    "CreditCard",
]

DATASET_NAMES: List[str] = SYNTHETIC_DATASET_NAMES + REAL_RIVER_DATASET_NAMES


def get_dataset_names(include_optional: bool = False) -> List[str]:
    """Retorna os nomes dos datasets do river configurados no projeto."""
    if include_optional:
        return list(DATASET_NAMES)
    return SYNTHETIC_DATASET_NAMES + ["Elec2"]


def build_dataset(name: str, seed: int = 42) -> StreamDataset:
    """Instancia um dataset do river pelo nome.

    Args:
        name: Nome da base (ex.: 'SEA-Rec', 'STAGGER-Rec', 'AGR-Abr',
              'AGR-Grad', 'RBF-Inc', 'Elec2', 'CreditCard').
        seed: Semente usada na geração dos streams sintéticos.
    """
    key = name.lower().replace("_", "-")
    if key == "sea-rec":
        return create_sea_rec(seed=seed)
    if key == "stagger-rec":
        return create_stagger_rec(seed=seed)
    if key == "agr-abr":
        return create_agr_abr(seed=seed)
    if key == "agr-grad":
        return create_agr_grad(seed=seed)
    if key == "rbf-inc":
        return create_rbf_inc(seed=seed)
    if key == "elec2":
        return create_elec2()
    if key == "creditcard":
        return create_credit_card()

    raise ValueError(
        f"Dataset desconhecido: '{name}'. Disponíveis: {', '.join(DATASET_NAMES)}."
    )


__all__ = [
    "StreamDataset",
    "DATASET_NAMES",
    "SYNTHETIC_DATASET_NAMES",
    "REAL_RIVER_DATASET_NAMES",
    "chain_concept_drift_streams",
    "get_dataset_names",
    "build_dataset",
    "create_sea_rec",
    "create_stagger_rec",
    "create_agr_abr",
    "create_agr_grad",
    "create_rbf_inc",
    "create_elec2",
    "create_credit_card",
]
