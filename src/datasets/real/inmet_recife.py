"""Stream real: estação automática do INMET A301 – Recife/PE, dados horários.

Gerado por ``scripts/prepare_inmet.py``. Cada instância é uma hora; o alvo é 1 se chover
na hora seguinte. Drift real (estação chuvosa abr–jul vs. estiagem), posições desconhecidas.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, Tuple

import pandas as pd

from src.datasets.base import StreamDataset

ROOT = Path(__file__).resolve().parents[3]
CSV = ROOT / "data" / "processed" / "recife_a301.csv"


def load_frame() -> pd.DataFrame:
    return pd.read_csv(CSV, parse_dates=["ts"])


def create_inmet_recife() -> StreamDataset:
    df = load_frame()
    feats = [c for c in df.columns if c not in {"ts", "alvo"}]
    X = df[feats].to_dict("records")
    y = df["alvo"].astype(int).tolist()

    def _gen() -> Iterator[Tuple[Dict[str, Any], int]]:
        yield from zip(X, y)

    return StreamDataset(name="INMET-Recife", stream=_gen(), n_classes=2, n_samples=len(df),
                         drift_positions=[], drift_type="real")
