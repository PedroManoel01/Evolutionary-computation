"""Stream sintética SEA-Rec (abrupto, recorrente, 2 classes, 50k instâncias).

Referência: Seção 8.1 de Proposta_Projeto_CE.pdf.
Variantes: 0 -> 1 -> 2 -> 3 -> 0 -> 1 -> 2 -> 3;
8 segmentos de 6.250 instâncias (total 50.000); ruído de 10%.
"""

from __future__ import annotations

from river.datasets import synth

from src.datasets.base import StreamDataset, chain_concept_drift_streams

SEA_VARIANTS = [0, 1, 2, 3, 0, 1, 2, 3]
SEGMENT_LENGTH = 6250
N_SAMPLES = SEGMENT_LENGTH * len(SEA_VARIANTS)  # 50.000
NOISE = 0.10


def create_sea_rec(seed: int = 42) -> StreamDataset:
    """Instancia a stream sintética SEA-Rec."""
    factories = [
        (lambda s, v=variant: synth.SEA(variant=v, noise=NOISE, seed=s))
        for variant in SEA_VARIANTS
    ]
    stream, drift_positions = chain_concept_drift_streams(
        concept_factories=factories,
        segment_length=SEGMENT_LENGTH,
        width=1,
        seed=seed,
    )
    return StreamDataset(
        name="SEA-Rec",
        stream=stream,
        n_classes=2,
        n_samples=N_SAMPLES,
        drift_positions=drift_positions,
        drift_type="abrupt_recurrent",
    )
