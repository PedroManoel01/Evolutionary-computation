"""Executa experimentos comparando Evolução Diferencial (S1–S6), AG (S1–S6),
DETO, DASE, baselines online (B1–B7) e benchmarks (ARF, SRP, LeveragingBagging)
sobre o cache prequencial de matrizes P.

Saídas em results/de_deto_dase/ (ou --out):
  - runs.csv    : métricas agregadas por (dataset, seed, block, method)
  - events.csv  : eventos de otimização pós-drift (fitness, avaliações, pesos, etc.)
  - curves.npz  : curvas de macro-F1 em janela deslizante (subamostradas)
  - config.json : parâmetros utilizados na execução

Uso:
    python scripts/run_de_deto_dase.py --seeds 10 --workers 4
    python scripts/run_de_deto_dase.py --datasets SEA-Rec AGR-Abr --seeds 5 --blocks de deto dase
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

from src.evaluation.cache import build_cache, cache_path, load_cache  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    detector_metrics,
    drift_reaction,
    prequential_summary,
    windowed_macro_f1,
)
from src.evaluation.replay import (  # noqa: E402
    ReplayConfig,
    run_dase_replay,
    run_deto_replay,
    run_online_baseline,
    run_triggered,
)
from src.evolution.de import DE_STRATEGIES, DEConfig  # noqa: E402
from src.evolution.ga import GAConfig  # noqa: E402

DATASETS = ["SEA-Rec", "STAGGER-Rec", "AGR-Abr", "AGR-Grad", "RBF-Inc", "Elec2"]
SYNTH = {"SEA-Rec", "STAGGER-Rec", "AGR-Abr", "AGR-Grad", "RBF-Inc"}
SENS_DATASETS = {"SEA-Rec", "AGR-Abr"}
STRATS = ["S1", "S2", "S3", "S4", "S5", "S6"]
TRANSITION = {"AGR-Grad": 2000}
CURVE_STEP = 100


def _de_sensitivity_grid(base_strategy: str, base_n_min: int = 500) -> List[tuple[str, Any, Dict[str, Any]]]:
    g: List[tuple[str, Any, Dict[str, Any]]] = []
    for N in (10, 50):
        g.append(("N", N, {"de": {"pop_size": N}}))
    for G in (20, 100):
        g.append(("G", G, {"de": {"max_gens": G}}))
    for nm in (100, 200, 500):
        if nm != base_n_min:
            g.append(("n_min", nm, {"n_min": nm}))
    for ir in (0.1, 0.5):
        g.append(("immigrant_rate", ir, {"immigrant_rate": ir}))
    for strat in ("rand/1/bin", "best/1/bin", "current-to-best/1/bin"):
        if strat != base_strategy:
            g.append(("de_strategy", strat, {"de": {"strategy": strat}}))
    for f_val in (0.3, 0.8):
        g.append(("de_F", f_val, {"de": {"F": f_val}}))
    for cr_val in (0.5, 0.7):
        g.append(("de_CR", cr_val, {"de": {"CR": cr_val}}))
    return g


def _nanmean(vals: List[float]) -> float:
    arr = np.asarray(vals, dtype=float)
    valid = arr[~np.isnan(arr)]
    return float(valid.mean()) if len(valid) else float("nan")


def _summarize(
    y: np.ndarray,
    y_hat: np.ndarray,
    C: int,
    drifts: List[int],
    wf1: np.ndarray,
) -> Dict[str, Any]:
    row = prequential_summary(y, y_hat, C)
    if drifts:
        row.update(drift_reaction(wf1, drifts))
    return row


def job(
    dataset: str,
    seed: int,
    blocks: List[str],
    params: Dict[str, Any],
) -> Dict[str, Any]:
    if not cache_path(dataset, seed).exists():
        if params.get("build_missing_cache", False):
            build_cache(dataset, seed=seed, with_benchmarks=True)
        else:
            raise FileNotFoundError(
                f"Cache ausente: {cache_path(dataset, seed)}. "
                "Execute scripts/build_cache.py ou passe --build-missing-cache."
            )

    P, y, bench, meta = load_cache(dataset, seed)
    max_inst = params.get("max_instances")
    if max_inst is not None and max_inst < len(y):
        P = P[:max_inst]
        y = y[:max_inst]
        bench = bench[:max_inst]

    n, M, C = P.shape
    drifts = [int(d) for d in meta["drift_positions"] if int(d) < n]
    tol_early = TRANSITION.get(dataset, 0) // 2
    pool_tpi = meta["time_pool_s"] / max(int(meta["n"]), 1)
    detector = params["detector"]

    rows: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []
    curves: Dict[str, np.ndarray] = {}

    def record(
        block: str,
        method: str,
        y_hat: np.ndarray,
        extra: Dict[str, Any],
        out=None,
        det_label: str = "",
        keep_curve: bool = True,
        save_events: bool = True,
    ) -> None:
        wf1 = windowed_macro_f1(y, y_hat, C)
        row = {
            "dataset": dataset,
            "seed": seed,
            "block": block,
            "method": method,
            "detector": det_label,
            **_summarize(y, y_hat, C, drifts, wf1),
            **extra,
        }
        if out is not None:
            dr_events = [e for e in out.events if not e.get("initial", False)]
            row.update({
                "n_events": len(dr_events),
                "n_evals": out.n_evals,
                "replay_time_s": out.replay_time,
                "replay_time_per_inst_ms": 1000.0 * out.replay_time / n,
                "time_per_inst_ms": 1000.0 * (out.replay_time / n + pool_tpi),
                "gens_to_95": _nanmean([e["gens_to_95"] for e in dr_events]) if dr_events else np.nan,
                "n_gens": _nanmean([e["n_gens"] for e in dr_events]) if dr_events else np.nan,
                "init_best": _nanmean([e["init_best"] for e in dr_events]) if dr_events else np.nan,
                "best_fit": _nanmean([e["best_fit"] for e in dr_events]) if dr_events else np.nan,
                "gain_over_old_w": (
                    _nanmean([e["best_fit"] - e["old_w_fit"] for e in dr_events])
                    if dr_events else np.nan
                ),
                "div_start": _nanmean([e["div_start"] for e in dr_events]) if dr_events else np.nan,
                "div_end": _nanmean([e["div_end"] for e in dr_events]) if dr_events else np.nan,
                "eval_size": _nanmean([e["eval_size"] for e in dr_events]) if dr_events else np.nan,
            })
            if det_label and drifts:
                row.update(detector_metrics(out.detections, drifts, n, early=tol_early))
            elif det_label:
                row["n_detections"] = len(out.detections)

            if save_events:
                for e in out.events:
                    events.append({
                        "dataset": dataset,
                        "seed": seed,
                        "block": block,
                        "method": method,
                        "detector": det_label,
                        **{k: v for k, v in e.items() if k not in {"history", "weights"}},
                        "history": json.dumps([round(float(h), 5) for h in e.get("history", [])]),
                        "weights": json.dumps([round(float(w), 4) for w in e.get("weights", [])]),
                    })

        rows.append(row)
        if keep_curve:
            curves[f"{dataset}|{method}|{seed}"] = wf1[::CURVE_STEP].astype(np.float32)

    de_cfg = DEConfig(
        pop_size=params["pop_size"],
        max_gens=params["max_gens"],
        stagnation=params["stagnation"],
        F=params["de_F"],
        CR=params["de_CR"],
        strategy=params["de_strategy"],
        dither=params["de_dither"],
        hyper_sigma=params["de_hyper_sigma"],
    )
    ga_cfg = GAConfig(
        pop_size=params["pop_size"],
        max_gens=params["max_gens"],
        stagnation=params["stagnation"],
        blx_alpha=params["ga_blx_alpha"],
    )

    n_min = params["n_min"]

    if "de" in blocks:
        for s in STRATS:
            cfg = ReplayConfig(
                n_min=n_min,
                strategy=s,
                detector=detector,
                optimizer="de",
                de=de_cfg,
                ga=ga_cfg,
            )
            out = run_triggered(P, y, drifts, cfg, seed=seed)
            record("de", f"DE-{s}", out.y_hat, {}, out=out, det_label=detector)

    if "ga" in blocks:
        for s in STRATS:
            cfg = ReplayConfig(
                n_min=n_min,
                strategy=s,
                detector=detector,
                optimizer="ga",
                ga=ga_cfg,
                de=de_cfg,
            )
            out = run_triggered(P, y, drifts, cfg, seed=seed)
            record("ga", f"GA-{s}", out.y_hat, {}, out=out, det_label=detector)

    if "deto" in blocks:
        out = run_deto_replay(
            P,
            y,
            drifts,
            n_min=n_min,
            detector=detector,
            deto_n_init=params["deto_n_init"],
            deto_n_iter=params["deto_n_iter"],
            seed=seed,
        )
        record("deto", "DETO", out.y_hat, {}, out=out, det_label=detector)

    if "dase" in blocks:
        out = run_dase_replay(
            P,
            y,
            chunk_interval=params["dase_chunk_interval"],
            window_size=params["dase_window"],
            chunk_size=params["dase_chunk_size"],
            pop_size=params["dase_pop_size"],
            max_iter=params["dase_max_iter"],
            seed=seed,
        )
        record("dase", "DASE", out.y_hat, {}, out=out, det_label="hcdd")

    if "baselines" in blocks:
        for b in ["B1", "B3", "B4", "B5", "B6"]:
            r = run_online_baseline(b, P, y)
            extra: Dict[str, Any] = {
                "replay_time_s": r["replay_time"],
                "replay_time_per_inst_ms": 1000.0 * r["replay_time"] / n,
                "time_per_inst_ms": 1000.0 * (r["replay_time"] / n + pool_tpi),
            }
            if b == "B3":
                extra["best_model"] = meta["models"][r["best_model"]]
            record("baselines", b, r["y_hat"], extra)
        out = run_triggered(
            P, y, drifts,
            ReplayConfig(n_min=n_min, strategy="S1", detector=detector, optimizer="random", ga=ga_cfg),
            seed=seed,
        )
        record("baselines", "B7", out.y_hat, {}, out=out, det_label=detector)

    if "benchmarks" in blocks:
        for b_idx, name in enumerate(meta.get("benchmarks", [])):
            if bench.ndim == 2 and b_idx < bench.shape[1] and (bench[:, b_idx] >= 0).all():
                t_b = meta.get("time_bench_s", {}).get(name, 0.0)
                record(
                    "benchmarks",
                    name,
                    bench[:, b_idx],
                    {"time_per_inst_ms": 1000.0 * t_b / max(int(meta["n"]), 1)},
                )

    if "detector" in blocks and dataset in SYNTH:
        extra_dets = [d for d in (["ddm"] + (["oracle"] if drifts else [])) if d != detector]
        for s in STRATS:
            for d in extra_dets:
                cfg = ReplayConfig(
                    n_min=n_min,
                    strategy=s,
                    detector=d,
                    optimizer="de",
                    de=de_cfg,
                    ga=ga_cfg,
                )
                out = run_triggered(P, y, drifts, cfg, seed=seed)
                record(
                    "detector",
                    f"DE-{s}",
                    out.y_hat,
                    {},
                    out=out,
                    det_label=d,
                    keep_curve=False,
                    save_events=False,
                )

    if "sensitivity" in blocks and dataset in SENS_DATASETS:
        de_base_dict = {
            "pop_size": params["pop_size"],
            "max_gens": params["max_gens"],
            "stagnation": params["stagnation"],
            "F": params["de_F"],
            "CR": params["de_CR"],
            "strategy": params["de_strategy"],
            "dither": params["de_dither"],
            "hyper_sigma": params["de_hyper_sigma"],
        }
        for s in STRATS:
            for factor, value, kw in _de_sensitivity_grid(params["de_strategy"], n_min):
                de_override = {**de_base_dict, **kw.get("de", {})}
                replay_kw = {"n_min": n_min, **{k: v for k, v in kw.items() if k != "de"}}
                cfg = ReplayConfig(
                    strategy=s,
                    detector=detector,
                    optimizer="de",
                    de=DEConfig(**de_override),
                    ga=ga_cfg,
                    **replay_kw,
                )
                out = run_triggered(P, y, drifts, cfg, seed=seed)
                record(
                    "sensitivity",
                    f"DE-{s}",
                    out.y_hat,
                    {"factor": factor, "value": str(value)},
                    out=out,
                    det_label=detector,
                    keep_curve=False,
                    save_events=False,
                )

    return {"rows": rows, "events": events, "curves": curves}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Executa experimentos de DE (S1–S6), AG (S1–S6), DETO, DASE e baselines sobre o cache."
    )
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument(
        "--blocks",
        nargs="+",
        default=["de", "ga", "deto", "dase", "baselines", "benchmarks", "detector", "sensitivity"],
        choices=["de", "ga", "deto", "dase", "baselines", "benchmarks", "detector", "sensitivity"],
    )
    ap.add_argument("--detector", choices=["adwin", "ddm", "oracle"], default="adwin")
    ap.add_argument(
        "--n-min",
        type=int,
        default=500,
        help="Mínimo de instâncias no conjunto de avaliação E após drift para DE, AG, DETO e B7 (padrão: 500).",
    )
    ap.add_argument("--max-instances", type=int, default=None, help="Opcional: trunca a stream para testes rápidos.")
    ap.add_argument("--build-missing-cache", action="store_true", help="Gera o cache automaticamente se estiver ausente.")

    # Hiperparâmetros de DE / AG
    ap.add_argument("--pop-size", type=int, default=30)
    ap.add_argument("--max-gens", type=int, default=50)
    ap.add_argument("--stagnation", type=int, default=10)
    ap.add_argument("--de-strategy", choices=sorted(DE_STRATEGIES), default="current-to-best/1/exp")
    ap.add_argument("--de-F", type=float, default=0.8)
    ap.add_argument("--de-CR", type=float, default=0.5)
    ap.add_argument("--de-dither", action="store_true")
    ap.add_argument("--de-hyper-sigma", type=float, default=0.1)
    ap.add_argument("--ga-blx-alpha", type=float, default=0.25)

    # Hiperparâmetros do DETO
    ap.add_argument("--deto-n-init", type=int, default=12)
    ap.add_argument("--deto-n-iter", type=int, default=15)

    # Hiperparâmetros do DASE
    ap.add_argument("--dase-chunk-interval", type=int, default=1000)
    ap.add_argument("--dase-window", type=int, default=500)
    ap.add_argument("--dase-chunk-size", type=int, default=50)
    ap.add_argument("--dase-pop-size", type=int, default=40)
    ap.add_argument("--dase-max-iter", type=int, default=20)

    ap.add_argument("--out", default=str(PROJECT_ROOT / "results" / "de_deto_dase"))
    ap.add_argument("--overwrite", action="store_true", help="Ignora checkpoints salvos e reexecuta tudo.")
    return ap.parse_args()


def _build_cache_task(dataset: str, seed: int) -> tuple[str, int, float]:
    t0 = time.perf_counter()
    build_cache(dataset, seed=seed, with_benchmarks=True)
    return dataset, seed, time.perf_counter() - t0


class _SimpleProgress:
    """Fallback minimalista de barra de progresso caso tqdm não esteja instalado."""

    def __init__(self, total: int, desc: str) -> None:
        self.total = total
        self.desc = desc
        self.n = 0
        self.t0 = time.perf_counter()

    def __enter__(self) -> "_SimpleProgress":
        print(f"{self.desc}: 0/{self.total}", flush=True)
        return self

    def __exit__(self, *exc: object) -> None:
        pass

    def set_postfix_str(self, s: str) -> None:
        self._last = s

    def update(self, inc: int = 1) -> None:
        self.n += inc
        elapsed = time.perf_counter() - self.t0
        pct = 100.0 * self.n / max(self.total, 1)
        eta = (elapsed / self.n) * (self.total - self.n) if self.n else 0.0
        extra = getattr(self, "_last", "")
        print(
            f"[{self.desc}] {self.n}/{self.total} ({pct:5.1f}%) | "
            f"{elapsed:.1f}s decorridos | ETA {eta:.1f}s | {extra}",
            flush=True,
        )


def _progress(total: int, desc: str):
    if tqdm is not None:
        return tqdm(total=total, desc=desc, unit="job", dynamic_ncols=True)
    return _SimpleProgress(total=total, desc=desc)


def _ckpt_paths(out_dir: Path, dataset: str, seed: int) -> tuple[Path, Path, Path]:
    ckpt_dir = out_dir / "checkpoints"
    stem = f"{dataset}_s{seed:02d}"
    return (
        ckpt_dir / f"{stem}_runs.csv",
        ckpt_dir / f"{stem}_events.csv",
        ckpt_dir / f"{stem}_curves.npz",
    )


def _save_ckpt(out_dir: Path, dataset: str, seed: int, res: Dict[str, Any]) -> None:
    runs_p, events_p, curves_p = _ckpt_paths(out_dir, dataset, seed)
    runs_p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(res["rows"]).to_csv(runs_p, index=False)
    pd.DataFrame(res["events"]).to_csv(events_p, index=False)
    np.savez_compressed(curves_p, **res["curves"])


def _load_ckpt(out_dir: Path, dataset: str, seed: int) -> Dict[str, Any] | None:
    runs_p, events_p, curves_p = _ckpt_paths(out_dir, dataset, seed)
    if not (runs_p.exists() and events_p.exists() and curves_p.exists()):
        return None
    rows = pd.read_csv(runs_p).to_dict(orient="records")
    events = pd.read_csv(events_p).to_dict(orient="records") if events_p.stat().st_size > 2 else []
    with np.load(curves_p) as z:
        curves = {k: z[k] for k in z.files}
    return {"rows": rows, "events": events, "curves": curves}


def _save_consolidated(
    out_dir: Path,
    rows: List[Dict[str, Any]],
    events: List[Dict[str, Any]],
    curves: Dict[str, np.ndarray],
    include_heavy: bool = True,
) -> None:
    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "runs.csv", index=False)
    if include_heavy:
        if events:
            pd.DataFrame(events).to_csv(out_dir / "events.csv", index=False)
        if curves:
            np.savez_compressed(out_dir / "curves.npz", **curves)


def main() -> None:
    args = parse_args()
    params = vars(args).copy()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8")

    tasks = [(d, s) for s in range(args.seeds) for d in args.datasets]

    # 1) Carrega checkpoints já finalizados (retomada automática, salvo --overwrite)
    rows: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []
    curves: Dict[str, np.ndarray] = {}
    pending: List[tuple[str, int]] = []
    for d, s in tasks:
        loaded = None if args.overwrite else _load_ckpt(out_dir, d, s)
        if loaded is not None:
            rows.extend(loaded["rows"])
            events.extend(loaded["events"])
            curves.update(loaded["curves"])
        else:
            pending.append((d, s))

    done_prev = len(tasks) - len(pending)
    if done_prev:
        print(f"Retomando: {done_prev}/{len(tasks)} jobs já concluídos em checkpoints; restam {len(pending)}.", flush=True)

    # 2) Se solicitado, constrói caches ausentes primeiro com barra de progresso própria
    missing = [(d, s) for d, s in pending if not cache_path(d, s).exists()]
    if missing:
        if not args.build_missing_cache:
            d0, s0 = missing[0]
            raise FileNotFoundError(
                f"{len(missing)} arquivo(s) de cache ausente(s), ex.: {cache_path(d0, s0)}. "
                "Execute scripts/build_cache.py ou passe --build-missing-cache."
            )
        with _progress(len(missing), desc="Construindo cache") as pbar:
            if args.workers <= 1:
                for d, s in missing:
                    _, _, dt = _build_cache_task(d, s)
                    pbar.set_postfix_str(f"{d} s{s:02d} ({dt:.1f}s)")
                    pbar.update(1)
            else:
                with ProcessPoolExecutor(max_workers=args.workers) as ex:
                    futs = {ex.submit(_build_cache_task, d, s): (d, s) for d, s in missing}
                    for f in as_completed(futs):
                        d, s, dt = f.result()
                        pbar.set_postfix_str(f"{d} s{s:02d} ({dt:.1f}s)")
                        pbar.update(1)

    # 3) Executa os experimentos pendentes com salvamento incremental por (dataset, seed)
    t0 = time.time()
    failed: List[tuple[str, int, str]] = []
    try:
        if pending:
            with _progress(len(pending), desc="Experimentos") as pbar:
                if args.workers <= 1:
                    for d, s in pending:
                        tj = time.perf_counter()
                        try:
                            r = job(d, s, args.blocks, params)
                        except Exception as exc:
                            msg = f"{type(exc).__name__}: {exc}"
                            failed.append((d, s, msg))
                            print(f"\n[ERRO] {d} s{s:02d}: {msg}", flush=True)
                            pbar.update(1)
                            continue
                        _save_ckpt(out_dir, d, s, r)
                        rows.extend(r["rows"])
                        events.extend(r["events"])
                        curves.update(r["curves"])
                        _save_consolidated(out_dir, rows, events, curves, include_heavy=False)
                        pbar.set_postfix_str(f"{d} s{s:02d} ({time.perf_counter() - tj:.1f}s)")
                        pbar.update(1)
                else:
                    with ProcessPoolExecutor(max_workers=args.workers) as ex:
                        futs = {ex.submit(job, d, s, args.blocks, params): (d, s) for d, s in pending}
                        for f in as_completed(futs):
                            d, s = futs[f]
                            try:
                                r = f.result()
                            except Exception as exc:
                                msg = f"{type(exc).__name__}: {exc}"
                                failed.append((d, s, msg))
                                print(f"\n[ERRO] {d} s{s:02d}: {msg}", flush=True)
                                pbar.update(1)
                                continue
                            _save_ckpt(out_dir, d, s, r)
                            rows.extend(r["rows"])
                            events.extend(r["events"])
                            curves.update(r["curves"])
                            _save_consolidated(out_dir, rows, events, curves, include_heavy=False)
                            pbar.set_postfix_str(f"{d} s{s:02d}")
                            pbar.update(1)
    finally:
        _save_consolidated(out_dir, rows, events, curves, include_heavy=True)
        if failed:
            (out_dir / "errors.log").write_text(
                "\n".join(f"{d} s{s:02d}: {msg}" for d, s, msg in failed) + "\n",
                encoding="utf-8",
            )

    print(
        f"OK: {len(rows)} execuções, {len(events)} eventos, {len(failed)} falhas "
        f"em {time.time() - t0:.1f}s -> {out_dir}"
    )


if __name__ == "__main__":
    main()
