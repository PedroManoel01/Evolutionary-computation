"""Base real CreditCard (river.datasets.CreditCard) — opcional.

Referência: Seção 8.2 de Proposta_Projeto_CE.pdf.
Desbalanceamento extremo, 2 classes.
"""

from __future__ import annotations

from river import datasets

from src.datasets.base import StreamDataset


def create_credit_card() -> StreamDataset:
    """Instancia o dataset real CreditCard do river."""
    return StreamDataset(
        name="CreditCard",
        stream=datasets.CreditCard(),
        n_classes=2,
        n_samples=284807,
        drift_positions=[],
        drift_type="unknown",
    )
