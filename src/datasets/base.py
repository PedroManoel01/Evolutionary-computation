"""Estrutura base e utilitários para streams de dados (river)."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import islice
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple

from river.datasets import synth


@dataclass
class StreamDataset:
    """Representa um stream de classificação com metadados de drift."""

    name: str
    stream: Iterable[Tuple[Dict[str, Any], Any]]
    n_classes: int
    n_samples: Optional[int] = None
    drift_positions: List[int] = field(default_factory=list)
    drift_type: str = "unknown"

    def __iter__(self) -> Iterator[Tuple[Dict[str, Any], Any]]:
        if self.n_samples is not None:
            return islice(self.stream, self.n_samples)
        return iter(self.stream)


def chain_concept_drift_streams(
    concept_factories: List[Callable[[int], Any]],
    segment_length: int,
    width: int = 1,
    seed: int = 42,
) -> Tuple[Any, List[int]]:
    """Encadeia múltiplos conceitos usando river.datasets.synth.ConceptDriftStream.

    Args:
        concept_factories: Lista de funções `fn(seed) -> SyntheticDataset` na ordem dos segmentos.
        segment_length: Número de instâncias por segmento.
        width: Largura da transição do ConceptDriftStream (1 = abrupto).
        seed: Semente base para reprodutibilidade.

    Returns:
        Tupla `(stream_encadeado, posicoes_de_drift)`.
    """
    if len(concept_factories) < 2:
        raise ValueError("São necessários pelo menos 2 conceitos para encadear.")

    streams = [factory(seed + idx) for idx, factory in enumerate(concept_factories)]
    drift_positions = [
        segment_length * i for i in range(1, len(concept_factories))
    ]

    if width <= 1:
        def _abrupt_generator() -> Iterator[Tuple[Dict[str, Any], Any]]:
            for stream in streams:
                yield from islice(stream, segment_length)

        return _abrupt_generator(), drift_positions

    chained = streams[-1]

    for idx in range(len(streams) - 2, -1, -1):
        chained = synth.ConceptDriftStream(
            stream=streams[idx],
            drift_stream=chained,
            position=segment_length,
            width=width,
            seed=seed + 100 + idx,
        )

    return chained, drift_positions
