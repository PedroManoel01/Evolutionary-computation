"""Gera o cache das matrizes P (pool) e das predições dos benchmarks para todas as bases x sementes.

Uso:
    python scripts/build_cache.py --datasets SEA-Rec AGR-Abr --seeds 30 --workers 8
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.cache import build_cache  # noqa: E402

DEFAULT_DATASETS = ["SEA-Rec", "STAGGER-Rec", "AGR-Abr", "AGR-Grad", "RBF-Inc", "Elec2"]


def _job(ds: str, seed: int, bench: bool) -> str:
    s = time.time()
    p = build_cache(ds, seed, with_benchmarks=bench)
    return f"{ds} s{seed:02d} -> {p.name} ({time.time() - s:.0f}s)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-bench", action="store_true")
    args = ap.parse_args()

    jobs = [(d, s) for s in range(args.seeds) for d in args.datasets]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_job, d, s, not args.no_bench) for d, s in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            print(f"[{i}/{len(futs)}] {f.result()}", flush=True)


if __name__ == "__main__":
    main()
