"""Stream sintética AGR-Grad (gradual, novo, 2 classes desbalanceado, 50k instâncias).

Referência: Seção 8.1 de Proposta_Projeto_CE.pdf.
Igual ao AGR-Abr, com largura de transição de 2.000 instâncias.
"""

from __future__ import annotations

from river.datasets import synth

from src.datasets.base import StreamDataset, chain_concept_drift_streams
from src.datasets.synthetic.agr_abr import (
    AGR_FUNCTIONS,
    N_SAMPLES,
    PERTURBATION,
    SEGMENT_LENGTH,
)

TRANSITION_WIDTH = 2000


def create_agr_grad(seed: int = 42) -> StreamDataset:
    """Instancia a stream sintética AGR-Grad."""
    factories = [
        (
            lambda s, fn=func: synth.Agrawal(
                classification_function=fn,
                perturbation=PERTURBATION,
                balance_classes=False,
                seed=s,
            )
        )
        for func in AGR_FUNCTIONS
    ]
    stream, drift_positions = chain_concept_drift_streams(
        concept_factories=factories,
        segment_length=SEGMENT_LENGTH,
        width=TRANSITION_WIDTH,
        seed=seed,
    )
    return StreamDataset(
        name="AGR-Grad",
        stream=stream,
        n_classes=2,
        n_samples=N_SAMPLES,
        drift_positions=drift_positions,
        drift_type="gradual_new",
    )
