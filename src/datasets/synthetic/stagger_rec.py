"""Stream sintética STAGGER-Rec (abrupto, recorrente, 2 classes, 30k instâncias).

Referência: Seção 8.1 de Proposta_Projeto_CE.pdf.
Funções: 0 -> 1 -> 2 -> 0 -> 1 -> 2;
6 segmentos de 5.000 instâncias (total 30.000); 10% de ruído nos rótulos aplicado manualmente.
"""

from __future__ import annotations

import random
from typing import Any, Dict, Iterable, Iterator, Tuple

from river.datasets import synth

from src.datasets.base import StreamDataset, chain_concept_drift_streams

STAGGER_FUNCTIONS = [0, 1, 2, 0, 1, 2]
SEGMENT_LENGTH = 5000
N_SAMPLES = SEGMENT_LENGTH * len(STAGGER_FUNCTIONS)  # 30.000
LABEL_NOISE = 0.10


def _apply_binary_label_noise(
    stream: Iterable[Tuple[Dict[str, Any], Any]],
    noise_rate: float,
    seed: int,
) -> Iterator[Tuple[Dict[str, Any], int]]:
    rng = random.Random(seed)
    for x, y in stream:
        label = int(y)
        if rng.random() < noise_rate:
            label = 1 - label
        yield x, label


def create_stagger_rec(seed: int = 42) -> StreamDataset:
    """Instancia a stream sintética STAGGER-Rec com 10% de ruído nos rótulos."""
    factories = [
        (
            lambda s, fn=func: synth.STAGGER(
                classification_function=fn,
                seed=s,
                balance_classes=False,
            )
        )
        for func in STAGGER_FUNCTIONS
    ]
    base_stream, drift_positions = chain_concept_drift_streams(
        concept_factories=factories,
        segment_length=SEGMENT_LENGTH,
        width=1,
        seed=seed,
    )
    noisy_stream = _apply_binary_label_noise(
        base_stream,
        noise_rate=LABEL_NOISE,
        seed=seed + 999,
    )
    return StreamDataset(
        name="STAGGER-Rec",
        stream=noisy_stream,
        n_classes=2,
        n_samples=N_SAMPLES,
        drift_positions=drift_positions,
        drift_type="abrupt_recurrent",
    )
