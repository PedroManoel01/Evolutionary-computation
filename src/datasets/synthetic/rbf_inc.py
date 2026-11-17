"""Stream sintética RBF-Inc (incremental, 5 classes, 10 atributos, 50k instâncias).

Referência: Seção 8.1 de Proposta_Projeto_CE.pdf.
RandomRBFDrift: 5 classes, 10 atributos, 50 centróides (25 em movimento), change_speed=0.0001.
"""

from __future__ import annotations

from river.datasets import synth

from src.datasets.base import StreamDataset

N_SAMPLES = 50000


def create_rbf_inc(seed: int = 42) -> StreamDataset:
    """Instancia a stream sintética RBF-Inc."""
    stream = synth.RandomRBFDrift(
        seed_model=seed,
        seed_sample=seed + 1,
        n_classes=5,
        n_features=10,
        n_centroids=50,
        n_drift_centroids=25,
        change_speed=0.0001,
    )
    return StreamDataset(
        name="RBF-Inc",
        stream=stream,
        n_classes=5,
        n_samples=N_SAMPLES,
        drift_positions=[],
        drift_type="incremental",
    )
