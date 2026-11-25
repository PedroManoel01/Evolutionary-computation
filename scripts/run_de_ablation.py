"""Estudo de ablação da Evolução Diferencial (DE) sobre o cache prequencial.

Avalia sistematicamente:
  1) Bloco `strategies`: todas as 8 estratégias de mutação/cruzamento do DE
     (rand/1, best/1, current-to-best/1, rand/2 × bin, exp) × S1..S6;
  2) Bloco `params`: ablação de F (0.3, 0.5, 0.8, dither), CR (0.3, 0.5, 0.7, 0.9),
     pop_size (15, 30, 50) e hyper_sigma em S4 (0.0, 0.1, 0.25, 0.5).

Salva checkpoints incrementais por (dataset, seed) em results/de_ablation/checkpoints/
com retomada automática.

Uso:
    python scripts/run_de_ablation.py --seeds 30 --workers 12 --n-min 500
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.cache import cache_path, load_cache  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    drift_reaction,
    prequential_summary,
    windowed_macro_f1,
)
from src.evaluation.replay import ReplayConfig, run_triggered  # noqa: E402
from src.evolution.de import DE_STRATEGIES, DEConfig  # noqa: E402

DATASETS = ["SEA-Rec", "STAGGER-Rec", "AGR-Abr", "AGR-Grad", "RBF-Inc", "Elec2"]
STRATS = ["S1", "S2", "S3", "S4", "S5", "S6"]


def _nanmean(vals: List[float]) -> float:
    arr = np.asarray(vals, dtype=float)
    valid = arr[~np.isnan(arr)]
    return float(valid.mean()) if len(valid) else float("nan")


def _run_one(
    P: np.ndarray,
    y: np.ndarray,
    drifts: List[int],
    dataset: str,
    seed: int,
    block: str,
    post_drift: str,
    de_cfg: DEConfig,
    n_min: int,
    detector: str,
    ablation_factor: str = "",
    ablation_value: str = "",
) -> Dict[str, Any]:
    n, _, C = P.shape
    cfg = ReplayConfig(
        n_min=n_min,
        strategy=post_drift,
        detector=detector,
        optimizer="de",
        de=de_cfg,
    )
    out = run_triggered(P, y, drifts, cfg, seed=seed)
    wf1 = windowed_macro_f1(y, out.y_hat, C)
    row: Dict[str, Any] = {
        "dataset": dataset,
        "seed": seed,
        "block": block,
        "de_strategy": de_cfg.strategy,
        "post_drift": post_drift,
        "F": de_cfg.F,
        "dither": de_cfg.dither,
        "CR": de_cfg.CR,
        "pop_size": de_cfg.pop_size,
        "hyper_sigma": de_cfg.hyper_sigma,
        "n_min": n_min,
        "detector": detector,
        "ablation_factor": ablation_factor,
        "ablation_value": ablation_value,
        **prequential_summary(y, out.y_hat, C),
    }
    if drifts:
        row.update(drift_reaction(wf1, drifts))

    dr_events = [e for e in out.events if not e.get("initial", False)]
    row.update({
        "n_events": len(dr_events),
        "n_evals": out.n_evals,
        "replay_time_s": out.replay_time,
        "gens_to_95": _nanmean([e["gens_to_95"] for e in dr_events]) if dr_events else np.nan,
        "n_gens": _nanmean([e["n_gens"] for e in dr_events]) if dr_events else np.nan,
        "best_fit": _nanmean([e["best_fit"] for e in dr_events]) if dr_events else np.nan,
        "div_start": _nanmean([e["div_start"] for e in dr_events]) if dr_events else np.nan,
        "div_end": _nanmean([e["div_end"] for e in dr_events]) if dr_events else np.nan,
    })
    return row


def job(dataset: str, seed: int, blocks: List[str], params: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not cache_path(dataset, seed).exists():
        raise FileNotFoundError(
            f"Cache ausente: {cache_path(dataset, seed)}. Execute scripts/build_cache.py primeiro."
        )

    P, y, _, meta = load_cache(dataset, seed)
    drifts = [int(d) for d in meta["drift_positions"] if int(d) < len(y)]
    n_min = params["n_min"]
    detector = params["detector"]
    rows: List[Dict[str, Any]] = []

    # 1) Grade completa: 8 estratégias DE x 6 estratégias pós-drift (S1..S6)
    if "strategies" in blocks:
        for de_strat in sorted(DE_STRATEGIES):
            de_cfg = DEConfig(
                pop_size=params["pop_size"],
                max_gens=params["max_gens"],
                stagnation=params["stagnation"],
                F=params["base_F"],
                CR=params["base_CR"],
                strategy=de_strat,
            )
            for s in STRATS:
                rows.append(_run_one(
                    P, y, drifts, dataset, seed,
                    block="strategies", post_drift=s, de_cfg=de_cfg,
                    n_min=n_min, detector=detector,
                    ablation_factor="de_strategy", ablation_value=de_strat,
                ))

    # 2) Ablação de hiperparâmetros do DE (F/dither, CR, pop_size e hyper_sigma em S4)
    if "params" in blocks:
        param_strats = ["rand/1/bin", "current-to-best/1/bin"]
        param_post = ["S1", "S4", "S6"]

        for de_strat in param_strats:
            # Ablação de F + dither
            for label, f_val, dith in [
                ("0.3", 0.3, False),
                ("0.5", 0.5, False),
                ("0.8", 0.8, False),
                ("dither", 0.5, True),
            ]:
                cfg = DEConfig(
                    pop_size=params["pop_size"], max_gens=params["max_gens"],
                    stagnation=params["stagnation"], F=f_val, CR=params["base_CR"],
                    strategy=de_strat, dither=dith,
                )
                for s in param_post:
                    rows.append(_run_one(
                        P, y, drifts, dataset, seed,
                        block="params", post_drift=s, de_cfg=cfg,
                        n_min=n_min, detector=detector,
                        ablation_factor="F", ablation_value=label,
                    ))

            # Ablação de CR
            for cr_val in (0.3, 0.5, 0.7, 0.9):
                cfg = DEConfig(
                    pop_size=params["pop_size"], max_gens=params["max_gens"],
                    stagnation=params["stagnation"], F=params["base_F"], CR=cr_val,
                    strategy=de_strat,
                )
                for s in param_post:
                    rows.append(_run_one(
                        P, y, drifts, dataset, seed,
                        block="params", post_drift=s, de_cfg=cfg,
                        n_min=n_min, detector=detector,
                        ablation_factor="CR", ablation_value=str(cr_val),
                    ))

            # Ablação de pop_size
            for npop in (15, 30, 50):
                cfg = DEConfig(
                    pop_size=npop, max_gens=params["max_gens"],
                    stagnation=params["stagnation"], F=params["base_F"], CR=params["base_CR"],
                    strategy=de_strat,
                )
                for s in param_post:
                    rows.append(_run_one(
                        P, y, drifts, dataset, seed,
                        block="params", post_drift=s, de_cfg=cfg,
                        n_min=n_min, detector=detector,
                        ablation_factor="pop_size", ablation_value=str(npop),
                    ))

            # Ablação de hyper_sigma especificamente em S4 (0.0 = sem ruído extra)
            for hsig in (0.0, 0.1, 0.25, 0.5):
                cfg = DEConfig(
                    pop_size=params["pop_size"], max_gens=params["max_gens"],
                    stagnation=params["stagnation"], F=params["base_F"], CR=params["base_CR"],
                    strategy=de_strat, hyper_sigma=hsig,
                )
                rows.append(_run_one(
                    P, y, drifts, dataset, seed,
                    block="params", post_drift="S4", de_cfg=cfg,
                    n_min=n_min, detector=detector,
                    ablation_factor="hyper_sigma", ablation_value=str(hsig),
                ))

    return rows


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Estudo de ablação da Evolução Diferencial (DE).")
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--blocks", nargs="+", default=["strategies", "params"], choices=["strategies", "params"])
    ap.add_argument("--detector", choices=["adwin", "ddm", "oracle"], default="adwin")
    ap.add_argument("--n-min", type=int, default=500)
    ap.add_argument("--pop-size", type=int, default=30)
    ap.add_argument("--max-gens", type=int, default=50)
    ap.add_argument("--stagnation", type=int, default=10)
    ap.add_argument("--base-F", type=float, default=0.5)
    ap.add_argument("--base-CR", type=float, default=0.9)
    ap.add_argument("--out", default=str(PROJECT_ROOT / "results" / "de_ablation"))
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def _ckpt_path(out_dir: Path, dataset: str, seed: int) -> Path:
    return out_dir / "checkpoints" / f"{dataset}_s{seed:02d}.csv"


def main() -> None:
    args = parse_args()
    params = vars(args).copy()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8")

    tasks = [(d, s) for s in range(args.seeds) for d in args.datasets]
    rows: List[Dict[str, Any]] = []
    pending: List[tuple[str, int]] = []
    for d, s in tasks:
        p = _ckpt_path(out_dir, d, s)
        if not args.overwrite and p.exists():
            rows.extend(pd.read_csv(p).to_dict(orient="records"))
        else:
            pending.append((d, s))

    done_prev = len(tasks) - len(pending)
    if done_prev:
        print(f"Retomando: {done_prev}/{len(tasks)} jobs concluídos; restam {len(pending)}.", flush=True)

    t0 = time.time()
    try:
        if pending:
            pbar = tqdm(total=len(pending), desc="Ablação DE", unit="job", dynamic_ncols=True) if tqdm else None
            try:
                if args.workers <= 1:
                    for d, s in pending:
                        r = job(d, s, args.blocks, params)
                        cp = _ckpt_path(out_dir, d, s)
                        cp.parent.mkdir(parents=True, exist_ok=True)
                        pd.DataFrame(r).to_csv(cp, index=False)
                        rows.extend(r)
                        if pbar:
                            pbar.set_postfix_str(f"{d} s{s:02d}")
                            pbar.update(1)
                else:
                    with ProcessPoolExecutor(max_workers=args.workers) as ex:
                        futs = {ex.submit(job, d, s, args.blocks, params): (d, s) for d, s in pending}
                        for f in as_completed(futs):
                            d, s = futs[f]
                            r = f.result()
                            cp = _ckpt_path(out_dir, d, s)
                            cp.parent.mkdir(parents=True, exist_ok=True)
                            pd.DataFrame(r).to_csv(cp, index=False)
                            rows.extend(r)
                            if pbar:
                                pbar.set_postfix_str(f"{d} s{s:02d}")
                                pbar.update(1)
            finally:
                if pbar:
                    pbar.close()
    finally:
        if rows:
            pd.DataFrame(rows).to_csv(out_dir / "ablation.csv", index=False)

    print(f"OK: {len(rows)} execuções em {time.time() - t0:.1f}s -> {out_dir / 'ablation.csv'}")


if __name__ == "__main__":
    main()
