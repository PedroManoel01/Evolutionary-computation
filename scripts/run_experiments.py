"""Executa a grade experimental (Seção 10.4) como replay sobre o cache das matrizes P.

Blocos: main (S1..S6 x ADWIN), baselines (B1..B8), benchmarks (ARF/SRP/LB),
detector (S1..S6 x {DDM, Oráculo} nas sintéticas) e sensibilidade (SEA-Rec e AGR-Abr).

Saídas em results/raw/: runs.csv, events.csv, curves.npz.

Uso:
    python scripts/run_experiments.py --seeds 30 --workers 8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.cache import load_cache  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    detector_metrics,
    drift_reaction,
    prequential_summary,
    windowed_macro_f1,
)
from src.evaluation.replay import ReplayConfig, run_online_baseline, run_triggered  # noqa: E402
from src.evolution.ga import GAConfig  # noqa: E402

DATASETS = ["SEA-Rec", "STAGGER-Rec", "AGR-Abr", "AGR-Grad", "RBF-Inc", "Elec2"]
SYNTH = ["SEA-Rec", "STAGGER-Rec", "AGR-Abr", "AGR-Grad", "RBF-Inc"]
SENS_DATASETS = ["SEA-Rec", "AGR-Abr"]
STRATS = ["S1", "S2", "S3", "S4", "S5", "S6"]
TRANSITION = {"AGR-Grad": 2000}
CURVE_STEP = 100


def sensitivity_grid() -> List[Tuple[str, Any, Dict[str, Any]]]:
    g: List[Tuple[str, Any, Dict[str, Any]]] = []
    for N in (10, 50):
        g.append(("N", N, {"ga": {"pop_size": N}}))
    for G in (20, 100):
        g.append(("G", G, {"ga": {"max_gens": G}}))
    for nm in (50, 200):
        g.append(("n_min", nm, {"n_min": nm}))
    for ir in (0.1, 0.5):
        g.append(("immigrant_rate", ir, {"immigrant_rate": ir}))
    g.append(("survivor", "mu+lambda", {"ga": {"survivor": "mu+lambda"}}))
    return g


def _cfg(**kw: Any) -> ReplayConfig:
    ga_kw = kw.pop("ga", {})
    return ReplayConfig(ga=GAConfig(**ga_kw), **kw)


def _summarize(y: np.ndarray, y_hat: np.ndarray, C: int, drifts: List[int], wf1: np.ndarray) -> Dict[str, Any]:
    row = prequential_summary(y, y_hat, C)
    if drifts:
        row.update(drift_reaction(wf1, drifts))
    return row


def job(dataset: str, seed: int, blocks: List[str]) -> Dict[str, Any]:
    P, y, bench, meta = load_cache(dataset, seed)
    n, M, C = P.shape
    drifts = list(meta["drift_positions"])
    tol_early = TRANSITION.get(dataset, 0) // 2
    pool_tpi = meta["time_pool_s"] / n
    rows: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []
    curves: Dict[str, np.ndarray] = {}

    def record(block: str, method: str, y_hat: np.ndarray, extra: Dict[str, Any], keep_curve: bool,
               out=None, detector: str = "") -> None:
        wf1 = windowed_macro_f1(y, y_hat, C)
        row = {"dataset": dataset, "seed": seed, "block": block, "method": method,
               "detector": detector, **_summarize(y, y_hat, C, drifts, wf1), **extra}
        if out is not None:
            dr = [e for e in out.events if not e["initial"]]
            row.update({
                "n_events": len(dr), "n_evals": out.n_evals,
                "replay_time_per_inst_ms": 1000 * out.replay_time / n,
                "time_per_inst_ms": 1000 * (out.replay_time / n + pool_tpi),
                "gens_to_95": np.mean([e["gens_to_95"] for e in dr]) if dr else np.nan,
                "n_gens": np.mean([e["n_gens"] for e in dr]) if dr else np.nan,
                "init_best": np.mean([e["init_best"] for e in dr]) if dr else np.nan,
                "best_fit": np.mean([e["best_fit"] for e in dr]) if dr else np.nan,
                "gain_over_old_w": np.mean([e["best_fit"] - e["old_w_fit"] for e in dr]) if dr else np.nan,
                "div_start": np.mean([e["div_start"] for e in dr]) if dr else np.nan,
                "div_end": np.mean([e["div_end"] for e in dr]) if dr else np.nan,
                "eval_size": np.mean([e["eval_size"] for e in dr]) if dr else np.nan,
            })
            if detector and drifts:
                row.update(detector_metrics(out.detections, drifts, n, early=tol_early))
            elif detector:
                row["n_detections"] = len(out.detections)
            if block in {"main", "baselines"}:
                for e in out.events:
                    events.append({"dataset": dataset, "seed": seed, "block": block, "method": method,
                                   "detector": detector, **{k: v for k, v in e.items()
                                                            if k not in {"history", "weights"}},
                                   "history": json.dumps([round(h, 5) for h in e["history"]]),
                                   "weights": json.dumps([round(w, 4) for w in e["weights"]])})
        rows.append(row)
        if keep_curve:
            curves[f"{dataset}|{method}|{detector}|{seed}"] = wf1[::CURVE_STEP].astype(np.float32)

    if "main" in blocks or "detector" in blocks:
        dets = ["adwin"] if "detector" not in blocks else ["adwin", "ddm"] + (["oracle"] if drifts else [])
        if dataset not in SYNTH:
            dets = ["adwin"]
        for s in STRATS:
            for d in dets:
                out = run_triggered(P, y, drifts, _cfg(strategy=s, detector=d), seed=seed)
                block = "main" if d == "adwin" else "detector"
                record(block, s, out.y_hat, {}, keep_curve=(d == "adwin"), out=out, detector=d)

    if "baselines" in blocks:
        for b in ["B1", "B3", "B4", "B5", "B6"]:
            r = run_online_baseline(b, P, y)
            extra = {"time_per_inst_ms": 1000 * (r["replay_time"] / n + pool_tpi)}
            if b == "B3":
                extra["best_model"] = meta["models"][r["best_model"]]
            record("baselines", b, r["y_hat"], extra, keep_curve=True)
        out = run_triggered(P, y, drifts, _cfg(strategy="S1", frozen=True), seed=seed)
        record("baselines", "B2", out.y_hat, {}, True, out=out)
        out = run_triggered(P, y, drifts, _cfg(strategy="S1", optimizer="random"), seed=seed)
        record("baselines", "B7", out.y_hat, {}, True, out=out, detector="adwin")
        out = run_triggered(P, y, drifts, _cfg(strategy="S2", optimizer="de"), seed=seed)
        record("baselines", "B8", out.y_hat, {}, True, out=out, detector="adwin")
        for m in range(M):
            record("models", meta["models"][m], P[:, m, :].argmax(1), {}, keep_curve=False)
        # diversidade do pool: discordância média entre pares de modelos
        pred = P.argmax(2)
        dis = [(pred[:, i] != pred[:, j]).mean() for i in range(M) for j in range(i + 1, M)]
        rows[-1]["pool_disagreement"] = float(np.mean(dis))

    if "benchmarks" in blocks:
        for b, name in enumerate(meta["benchmarks"]):
            record("benchmarks", name, bench[:, b], {"time_per_inst_ms": 1000 * meta["time_bench_s"][name] / n},
                   keep_curve=True)

    if "sensitivity" in blocks and dataset in SENS_DATASETS:
        for s in STRATS:
            for factor, value, kw in sensitivity_grid():
                out = run_triggered(P, y, drifts, _cfg(strategy=s, **json.loads(json.dumps(kw))), seed=seed)
                record("sensitivity", s, out.y_hat, {"factor": factor, "value": str(value)},
                       keep_curve=False, out=out, detector="adwin")

    return {"rows": rows, "events": events, "curves": curves}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--blocks", nargs="+", default=["main", "baselines", "benchmarks", "detector", "sensitivity"])
    ap.add_argument("--out", default=str(PROJECT_ROOT / "results" / "raw"))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, events, curves = [], [], {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(job, d, s, args.blocks): (d, s) for s in range(args.seeds) for d in args.datasets}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            rows += r["rows"]
            events += r["events"]
            curves.update(r["curves"])
            d, s = futs[f]
            print(f"[{i}/{len(futs)}] {d} s{s:02d} ({time.time() - t0:.0f}s)", flush=True)

    pd.DataFrame(rows).to_csv(out_dir / "runs.csv", index=False)
    pd.DataFrame(events).to_csv(out_dir / "events.csv", index=False)
    np.savez_compressed(out_dir / "curves.npz", **curves)
    print(f"OK: {len(rows)} execuções, {len(events)} eventos de otimização -> {out_dir}")


if __name__ == "__main__":
    main()
