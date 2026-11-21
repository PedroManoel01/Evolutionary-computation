"""Experimento: 7 métodos x 30 sementes no stream real INMET-Recife (replay sobre o cache).

Fator de interesse  : método de combinação do ensemble (7 níveis)
Réplicas            : 30 sementes (aleatoriedade dos modelos do pool e do AG/busca aleatória)
Blocos              : ano (2015..2025) e estação (chuvosa abr–jul x seca)

Para cada execução guarda o macro-F1 global e o macro-F1 por ano e por estação, que
alimentam a análise estatística em ``scripts/analyze.py``.

Uso:
    python scripts/run_experiments.py --seeds 30 --workers 8
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.real import load_frame  # noqa: E402
from src.evaluation.cache import load_cache  # noqa: E402
from src.evaluation.metrics import kappa, macro_f1  # noqa: E402
from src.evaluation.replay import ReplayConfig, run_online_baseline, run_triggered  # noqa: E402

DATASET = "INMET-Recife"
METHODS = {
    "B1": "Média uniforme",
    "B4": "Seleção online",
    "B7": "Busca aleatória",
    "S1": "AG reinício",
    "S2": "AG warm start",
    "S6": "AG memória+imigr.",
    "ARF": "Adaptive Random Forest",
}
N_INIT = 500  # instâncias de aquecimento, fora da avaliação por bloco


def _blocks() -> pd.DataFrame:
    ts = load_frame()["ts"]
    mes = ts.dt.month
    return pd.DataFrame({"ano": ts.dt.year.values,
                         "estacao": np.where(mes.between(4, 7), "chuvosa", "seca")})


def job(seed: int) -> List[Dict[str, Any]]:
    P, y, bench, meta = load_cache(DATASET, seed)
    blk = _blocks()
    assert len(blk) == len(y), "cache e CSV processado fora de sincronia"
    preds: Dict[str, np.ndarray] = {}
    info: Dict[str, Dict[str, Any]] = {}

    for b in ("B1", "B4"):
        preds[b] = run_online_baseline(b, P, y)["y_hat"]
    for name, cfg in {
        "B7": ReplayConfig(strategy="S1", optimizer="random"),
        "S1": ReplayConfig(strategy="S1"),
        "S2": ReplayConfig(strategy="S2"),
        "S6": ReplayConfig(strategy="S6"),
    }.items():
        out = run_triggered(P, y, [], cfg, seed=seed)
        preds[name] = out.y_hat
        info[name] = {"n_reotimizacoes": sum(not e["initial"] for e in out.events), "n_avaliacoes": out.n_evals}
    preds["ARF"] = bench[:, meta["benchmarks"].index("ARF")]

    rows: List[Dict[str, Any]] = []
    sl = slice(N_INIT, None)
    for m, yh in preds.items():
        base = {"seed": seed, "metodo": m, **info.get(m, {})}
        rows.append({**base, "bloco": "global", "nivel": "global",
                     "macro_f1": macro_f1(y[sl], yh[sl]), "kappa": kappa(y[sl], yh[sl], 2),
                     "acuracia": float((y[sl] == yh[sl]).mean())})
        for col in ("ano", "estacao"):
            g = blk[col].values
            for lv in np.unique(g):
                idx = np.where(g == lv)[0]
                idx = idx[idx >= N_INIT]
                rows.append({**base, "bloco": col, "nivel": str(lv),
                             "macro_f1": macro_f1(y[idx], yh[idx]), "kappa": kappa(y[idx], yh[idx], 2),
                             "acuracia": float((y[idx] == yh[idx]).mean())})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    out = ROOT / "results" / "raw"
    out.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(job, s): s for s in range(args.seeds)}
        for i, f in enumerate(as_completed(futs), 1):
            rows += f.result()
            print(f"[{i}/{len(futs)}] semente {futs[f]:02d} ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows).sort_values(["bloco", "nivel", "metodo", "seed"])
    df.to_csv(out / "runs.csv", index=False)
    print(f"OK: {len(df)} linhas -> {out / 'runs.csv'}")


if __name__ == "__main__":
    main()
