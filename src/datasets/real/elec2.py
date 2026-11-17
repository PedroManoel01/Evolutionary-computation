"""Base real Elec2 (river.datasets.Elec2).

Referência: Seção 8.2 de Proposta_Projeto_CE.pdf.
45.312 instâncias, 2 classes, drift desconhecido.
"""

from __future__ import annotations

from river import datasets

from src.datasets.base import StreamDataset


def create_elec2() -> StreamDataset:
    """Instancia o dataset real Elec2 do river."""
    return StreamDataset(
        name="Elec2",
        stream=datasets.Elec2(),
        n_classes=2,
        n_samples=45312,
        drift_positions=[],
        drift_type="unknown",
    )
