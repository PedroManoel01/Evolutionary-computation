"""Geração e leitura do cache das matrizes de probabilidade P (Seção 10.5 da proposta).

Os modelos base não dependem dos pesos do ensemble. Portanto o pool roda uma única
vez por base x semente (test-then-train) e salvamos:

* ``P``  : (n, M, C) float32 — probabilidades de cada modelo, ANTES de treinar em (x_t, y_t);
* ``y``  : (n,) int16 — rótulos inteiros 0..C-1;
* ``bench``: (n, 3) int16 — predições prequenciais de ARF, SRP e Leveraging Bagging;
* ``meta``: json com nomes dos modelos, posições de drift e tempos de execução.

Todo o restante (AG, estratégias, detectores e baselines) é replay em numpy sobre P.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def cache_dir() -> Path:
    """Diretório do cache. Pode ser sobrescrito por ``EVOCE_CACHE`` (ex.: fora do Google Drive)."""
    return Path(os.environ.get("EVOCE_CACHE", PROJECT_ROOT / "data" / "cache"))


def cache_path(dataset: str, seed: int) -> Path:
    return cache_dir() / f"{dataset}_s{seed:02d}.npz"


def _label_to_int(y: Any) -> int:
    if isinstance(y, (bool, np.bool_)):
        return int(y)
    return int(y)


def build_cache(
    dataset_name: str,
    seed: int,
    with_benchmarks: bool = True,
    overwrite: bool = False,
) -> Path:
    """Roda o pool (e os benchmarks) prequencialmente e salva o resultado em .npz."""
    from src.datasets import build_dataset
    from src.models.benchmarks import build_benchmarks
    from src.models.pool import build_model_pool

    out = cache_path(dataset_name, seed)
    if out.exists() and not overwrite:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)

    ds = build_dataset(dataset_name, seed=seed)
    pool = build_model_pool(seed=seed)
    names = list(pool.keys())
    benches = build_benchmarks(seed=seed) if with_benchmarks else {}
    bench_names = list(benches.keys())

    n_max = ds.n_samples or 10**7
    C = ds.n_classes
    M = len(names)
    P = np.zeros((n_max, M, C), dtype=np.float32)
    y_arr = np.zeros(n_max, dtype=np.int16)
    bench = np.full((n_max, max(len(bench_names), 1)), -1, dtype=np.int16)
    t_pool = 0.0
    t_bench = {b: 0.0 for b in bench_names}

    n = 0
    for x, y in ds:
        if n >= n_max:
            break
        yi = _label_to_int(y)
        s = time.perf_counter()
        for m, name in enumerate(names):
            proba = pool[name].predict_proba_one(x)
            for c, p in proba.items():
                ci = _label_to_int(c)
                if 0 <= ci < C:
                    P[n, m, ci] = p
        for model in pool.values():
            model.learn_one(x, y)
        t_pool += time.perf_counter() - s

        for b, name in enumerate(bench_names):
            s = time.perf_counter()
            pred = benches[name].predict_one(x)
            if pred is not None:
                bench[n, b] = _label_to_int(pred)
            benches[name].learn_one(x, y)
            t_bench[name] += time.perf_counter() - s

        y_arr[n] = yi
        n += 1

    P = P[:n]
    sums = P.sum(axis=2, keepdims=True)
    P = np.where(sums > 0, P / np.where(sums > 0, sums, 1), P).astype(np.float32)

    meta: Dict[str, Any] = {
        "dataset": dataset_name,
        "seed": seed,
        "n": n,
        "n_classes": C,
        "models": names,
        "benchmarks": bench_names,
        "drift_positions": list(ds.drift_positions),
        "drift_type": ds.drift_type,
        "time_pool_s": t_pool,
        "time_bench_s": t_bench,
    }
    np.savez_compressed(
        out, P=P, y=y_arr[:n], bench=bench[:n], meta=np.array(json.dumps(meta))
    )
    return out


def load_cache(dataset: str, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Carrega (P float64, y int64, bench int64, meta)."""
    with np.load(cache_path(dataset, seed)) as d:
        meta = json.loads(str(d["meta"]))
        return (
            d["P"].astype(np.float64),
            d["y"].astype(np.int64),
            d["bench"].astype(np.int64),
            meta,
        )


def maybe_load(dataset: str, seed: int) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]]:
    p = cache_path(dataset, seed)
    return load_cache(dataset, seed) if p.exists() else None
