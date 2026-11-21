"""Datasets do projeto de Estatística: apenas a base real do INMET."""

from __future__ import annotations

from src.datasets.base import StreamDataset
from src.datasets.real import create_inmet_recife

DATASET_NAMES = ["INMET-Recife"]


def build_dataset(name: str, seed: int = 42) -> StreamDataset:  # noqa: ARG001 (base real: sem semente)
    if name.lower().replace("_", "-") == "inmet-recife":
        return create_inmet_recife()
    raise ValueError(f"Dataset desconhecido: '{name}'. Disponível: INMET-Recife.")


__all__ = ["StreamDataset", "DATASET_NAMES", "build_dataset", "create_inmet_recife"]
