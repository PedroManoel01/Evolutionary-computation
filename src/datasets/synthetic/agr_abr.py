"""Stream sintética AGR-Abr (abrupto, novo, 2 classes desbalanceado, 50k instâncias).

Referência: Seção 8.1 de Proposta_Projeto_CE.pdf.
Funções: 0 -> 2 -> 4 -> 6 -> 8;
5 segmentos de 10.000 instâncias (total 50.000); perturbação 0.05; balance_classes=False.
"""

from __future__ import annotations

from river.datasets import synth

from src.datasets.base import StreamDataset, chain_concept_drift_streams

AGR_FUNCTIONS = [0, 2, 4, 6, 8]
SEGMENT_LENGTH = 10000
N_SAMPLES = SEGMENT_LENGTH * len(AGR_FUNCTIONS)  # 50.000
PERTURBATION = 0.05


def create_agr_abr(seed: int = 42) -> StreamDataset:
    """Instancia a stream sintética AGR-Abr."""
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
        width=1,
        seed=seed,
    )
    return StreamDataset(
        name="AGR-Abr",
        stream=stream,
        n_classes=2,
        n_samples=N_SAMPLES,
        drift_positions=drift_positions,
        drift_type="abrupt_new",
    )
