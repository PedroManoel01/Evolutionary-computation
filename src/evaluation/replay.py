"""Motor de replay prequencial sobre as matrizes P em cache (Seções 6, 9, 10 e 11 da proposta).

``run_triggered``: AG (S1..S6), B2 (AG congelado), B7 (busca aleatória) e B8 (DE) —
métodos reacionados pelo detector de drift, com o buffer do warning como conjunto E.

``run_online_baseline``: B1 (uniforme), B3 (melhor modelo a posteriori),
B4 (seleção online), B5 (ponderação por desempenho) e B6 (Hedge) — totalmente vetorizados.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from river import drift

from src.evaluation.objectives import macro_f1_batch, make_macro_f1_objective
from src.evolution.de import DEConfig, DifferentialEvolution
from src.evolution.ga import (
    GAConfig,
    GeneticAlgorithm,
    RandomSearch,
    random_simplex,
)
from src.strategies.post_drift import PostDriftStrategy


# ---------------------------------------------------------------------------
# Detectores (sinal = erro 0/1 do ensemble)
# ---------------------------------------------------------------------------
class _ADWINPair:
    """Par de ADWINs (δw=0,01 warning; δd=0,001 drift). Só conta AUMENTO do erro.

    O ADWIN detecta mudança nas duas direções; uma queda do erro (por exemplo, logo
    após aplicar novos pesos) não é concept drift e é ignorada.
    """

    has_warning_state = False

    def __init__(self, delta_w: float = 0.01, delta_d: float = 0.001) -> None:
        self.dw, self.dd = delta_w, delta_d
        self.reset()

    def reset(self) -> None:
        self.w = drift.ADWIN(delta=self.dw)
        self.d = drift.ADWIN(delta=self.dd)

    @staticmethod
    def _up(det: drift.ADWIN, x: int) -> bool:
        before = det.estimation
        det.update(x)
        return bool(det.drift_detected and det.estimation > before)

    def update(self, x: int) -> tuple[bool, bool]:
        warn = self._up(self.w, x)
        dr = self._up(self.d, x)
        return warn, dr


class _DDM:
    has_warning_state = True

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.det = drift.binary.DDM()

    def update(self, x: int) -> tuple[bool, bool]:
        self.det.update(bool(x))
        return bool(self.det.warning_detected), bool(self.det.drift_detected)


class _Oracle:
    has_warning_state = False

    def __init__(self, positions: Sequence[int]) -> None:
        self.pos = set(int(p) for p in positions)
        self.t = -1

    def reset(self) -> None:  # o contador global é preservado
        pass

    def update_at(self, t: int) -> tuple[bool, bool]:
        return False, t in self.pos


def make_detector(name: str, positions: Sequence[int]):
    name = name.lower()
    if name in {"adwin", "adwin_pair"}:
        return _ADWINPair()
    if name == "ddm":
        return _DDM()
    if name == "oracle":
        return _Oracle(positions)
    raise ValueError(name)


# ---------------------------------------------------------------------------
@dataclass
class ReplayConfig:
    n_init: int = 500
    n_min: int = 100
    b_max: int = 2000
    detector: str = "adwin"
    strategy: str = "S1"
    optimizer: str = "ga"          # ga | random | de
    frozen: bool = False           # B2
    immigrant_rate: float = 0.3
    memory_capacity: int = 10
    ga: GAConfig = field(default_factory=GAConfig)
    de: DEConfig = field(default_factory=DEConfig)


@dataclass
class RunOutput:
    y_hat: np.ndarray
    events: List[Dict[str, Any]]
    detections: List[int]
    replay_time: float
    n_evals: int
    weights_final: np.ndarray


def _predict(P: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.einsum("m,nmc->nc", w, P).argmax(axis=1)


def _gens_to_95(hist: Sequence[float]) -> int:
    h = np.asarray(hist)
    gain = h[-1] - h[0]
    if gain <= 1e-12:
        return 0
    return int(np.nonzero(h - h[0] >= 0.95 * gain)[0][0])


def run_triggered(P: np.ndarray, y: np.ndarray, drift_positions: Sequence[int],
                  cfg: ReplayConfig, seed: int) -> RunOutput:
    """Replay de um método reacionado por drift (AG + estratégia, B2, B7 ou B8)."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    n, M, _ = P.shape

    if cfg.optimizer == "ga":
        opt = GeneticAlgorithm(cfg.ga)
        pop_size = cfg.ga.pop_size
    elif cfg.optimizer == "de":
        opt = DifferentialEvolution(cfg.de)
        pop_size = cfg.de.pop_size
    elif cfg.optimizer == "random":
        opt = RandomSearch(budget=cfg.ga.pop_size * cfg.ga.max_gens, keep=cfg.ga.pop_size)
        pop_size = cfg.ga.pop_size
    else:
        raise ValueError(cfg.optimizer)
    strat = PostDriftStrategy(cfg.strategy, pop_size=pop_size,
                              immigrant_rate=cfg.immigrant_rate,
                              memory_capacity=cfg.memory_capacity)
    det = make_detector(cfg.detector, drift_positions)

    w = np.full(M, 1.0 / M)
    y_hat = np.empty(n, dtype=np.int64)
    y_hat[:] = _predict(P, w)
    events: List[Dict[str, Any]] = []
    detections: List[int] = []
    n_evals = 0
    prev_pop: Optional[np.ndarray] = None
    prev_fit: Optional[np.ndarray] = None

    def optimize(idx: np.ndarray, t_detect: int, t_now: int, initial: bool) -> None:
        nonlocal w, prev_pop, prev_fit, n_evals
        PE, yE = P[idx], y[idx]
        fitness = lambda pop: macro_f1_batch(pop, PE, yE)  # noqa: E731
        if initial:
            pop0 = random_simplex(pop_size, M, rng)
        else:
            pop0 = strat.initial_population(prev_pop, prev_fit, M, rng)
        res = opt.run(pop0, fitness, rng, hypermutation=(strat.hypermutation and not initial))
        n_evals += res.n_evals
        old_w_fit = float(fitness(w[None, :])[0])
        prev_pop, prev_fit = res.pop, res.fit
        w = res.best
        if t_now + 1 < n:
            y_hat[t_now + 1:] = _predict(P[t_now + 1:], w)
        events.append({
            "t_detect": t_detect, "t_apply": t_now + 1, "eval_size": len(idx),
            "initial": initial, "init_best": res.init_best_fit, "best_fit": res.best_fit,
            "old_w_fit": old_w_fit, "n_gens": res.n_gens, "n_evals": res.n_evals,
            "gens_to_95": _gens_to_95(res.history), "div_start": res.div_start,
            "div_end": res.div_end, "history": res.history, "weights": res.best.tolist(),
        })

    # 1) primeiras n_init instâncias com pesos uniformes, depois AG no "conceito 0"
    n_init = min(cfg.n_init, n)
    optimize(np.arange(n_init), n_init - 1, n_init - 1, initial=True)
    if cfg.frozen:
        return RunOutput(y_hat, events, detections, time.perf_counter() - t0, n_evals, w)

    err_all = None
    buffer: deque = deque(maxlen=cfg.b_max)
    in_warning = False
    warn_steps = 0
    awaiting = False
    E: List[int] = []
    t_detect = -1
    is_oracle = isinstance(det, _Oracle)

    t = n_init
    while t < n:
        if err_all is None:
            err_all = (y_hat != y).astype(np.int64)
        err = int(err_all[t])
        if is_oracle:
            warn, dr = det.update_at(t)
        else:
            warn, dr = det.update(err)

        if awaiting:
            E.append(t)
        elif dr:
            detections.append(t)
            t_detect = t
            strat.on_drift(w)
            E = list(buffer) + [t]
            buffer.clear()
            in_warning = False
            awaiting = True
            det.reset()
        else:
            if det.has_warning_state:
                if warn:
                    in_warning = True
                elif in_warning:          # DDM saiu do warning sem drift: warning falso
                    in_warning = False
                    buffer.clear()
            elif warn and not in_warning:
                in_warning = True
                warn_steps = 0
            if in_warning:
                buffer.append(t)
                warn_steps += 1
                if not det.has_warning_state and warn_steps > cfg.b_max:
                    in_warning = False    # B_max instâncias sem drift: warning falso
                    buffer.clear()

        if awaiting and len(E) >= cfg.n_min:
            optimize(np.asarray(E[-cfg.b_max:]), t_detect, t, initial=False)
            awaiting = False
            E = []
            det.reset()
            err_all = None                # pesos mudaram: recalcula erros futuros
        t += 1

    return RunOutput(y_hat, events, detections, time.perf_counter() - t0, n_evals, w)


def run_deto_replay(
    P: np.ndarray,
    y: np.ndarray,
    drift_positions: Sequence[int],
    *,
    n_init: int = 500,
    n_min: int = 100,
    b_max: int = 2000,
    detector: str = "adwin",
    deto_n_init: int = 16,
    deto_n_iter: int = 20,
    seed: int = 42,
) -> RunOutput:
    """Replay prequencial do baseline DETO (Transfer BO) sobre as matrizes P em cache."""
    from src.baselines.deto_adapter import DETOAdapter

    t0 = time.perf_counter()
    n, M, _ = P.shape
    deto = DETOAdapter(n_models=M, n_init=deto_n_init, n_iter=deto_n_iter, seed=seed)
    det = make_detector(detector, drift_positions)

    w = np.full(M, 1.0 / M)
    y_hat = np.empty(n, dtype=np.int64)
    y_hat[:] = _predict(P, w)
    events: List[Dict[str, Any]] = []
    detections: List[int] = []
    n_evals = 0

    def optimize(idx: np.ndarray, t_detect: int, t_now: int, initial: bool) -> None:
        nonlocal w, n_evals
        PE, yE = P[idx], y[idx]
        raw_obj = make_macro_f1_objective(PE, yE, minimize=True)
        ev_evals = 0

        def counted_obj(W: np.ndarray) -> np.ndarray:
            nonlocal ev_evals
            arr = np.asarray(W, dtype=float)
            ev_evals += 1 if arr.ndim == 1 else len(arr)
            return raw_obj(W)

        old_w_fit = float(macro_f1_batch(w[None, :], PE, yE)[0])
        w_new, best_f1 = deto.optimize(counted_obj)
        n_evals += ev_evals
        w = w_new
        if t_now + 1 < n:
            y_hat[t_now + 1:] = _predict(P[t_now + 1:], w)
        events.append({
            "t_detect": t_detect,
            "t_apply": t_now + 1,
            "eval_size": len(idx),
            "initial": initial,
            "init_best": float("nan"),
            "best_fit": float(best_f1),
            "old_w_fit": old_w_fit,
            "n_gens": deto_n_iter,
            "n_evals": ev_evals,
            "gens_to_95": float("nan"),
            "div_start": float("nan"),
            "div_end": float("nan"),
            "history": [],
            "weights": w.tolist(),
        })

    n_init_eff = min(n_init, n)
    optimize(np.arange(n_init_eff), n_init_eff - 1, n_init_eff - 1, initial=True)

    err_all = None
    buffer: deque = deque(maxlen=b_max)
    in_warning = False
    warn_steps = 0
    awaiting = False
    E: List[int] = []
    t_detect = -1
    is_oracle = isinstance(det, _Oracle)

    t = n_init_eff
    while t < n:
        if err_all is None:
            err_all = (y_hat != y).astype(np.int64)
        err = int(err_all[t])
        if is_oracle:
            warn, dr = det.update_at(t)
        else:
            warn, dr = det.update(err)

        if awaiting:
            E.append(t)
        elif dr:
            detections.append(t)
            t_detect = t
            E = list(buffer) + [t]
            buffer.clear()
            in_warning = False
            awaiting = True
            det.reset()
        else:
            if det.has_warning_state:
                if warn:
                    in_warning = True
                elif in_warning:
                    in_warning = False
                    buffer.clear()
            elif warn and not in_warning:
                in_warning = True
                warn_steps = 0
            if in_warning:
                buffer.append(t)
                warn_steps += 1
                if not det.has_warning_state and warn_steps > b_max:
                    in_warning = False
                    buffer.clear()

        if awaiting and len(E) >= n_min:
            optimize(np.asarray(E[-b_max:]), t_detect, t, initial=False)
            awaiting = False
            E = []
            det.reset()
            err_all = None
        t += 1

    return RunOutput(y_hat, events, detections, time.perf_counter() - t0, n_evals, w)


def run_dase_replay(
    P: np.ndarray,
    y: np.ndarray,
    *,
    n_init: int = 500,
    chunk_interval: int = 1000,
    window_size: int = 500,
    chunk_size: int = 50,
    pop_size: int = 50,
    max_iter: int = 30,
    seed: int = 42,
) -> RunOutput:
    """Replay prequencial do baseline DASE (HCDD + RBFN + DE) sobre as matrizes P em cache."""
    from src.baselines.dase_adapter import DASEAdapter

    t0 = time.perf_counter()
    n, M, _ = P.shape
    dase = DASEAdapter(
        n_models=M,
        chunk_size=chunk_size,
        pop_size=pop_size,
        max_iter=max_iter,
        seed=seed,
    )

    w = np.full(M, 1.0 / M)
    y_hat = np.empty(n, dtype=np.int64)
    y_hat[:] = _predict(P, w)
    events: List[Dict[str, Any]] = []
    detections: List[int] = []
    n_evals = 0

    def step_at(idx: np.ndarray, t_now: int, initial: bool) -> None:
        nonlocal w, n_evals
        PE, yE = P[idx], y[idx]
        raw_obj = make_macro_f1_objective(PE, yE, minimize=True)
        ev_evals = 0

        def counted_obj(W: np.ndarray) -> np.ndarray:
            nonlocal ev_evals
            arr = np.asarray(W, dtype=float)
            ev_evals += 1 if arr.ndim == 1 else len(arr)
            return raw_obj(W)

        old_w_fit = float(macro_f1_batch(w[None, :], PE, yE)[0])
        w_new, best_f1, hcdd_drift = dase.step(counted_obj)
        n_evals += ev_evals
        w = w_new
        if hcdd_drift:
            detections.append(t_now)
        if t_now + 1 < n:
            y_hat[t_now + 1:] = _predict(P[t_now + 1:], w)
        events.append({
            "t_detect": t_now,
            "t_apply": t_now + 1,
            "eval_size": len(idx),
            "initial": initial,
            "hcdd_drift": bool(hcdd_drift),
            "init_best": float("nan"),
            "best_fit": float(best_f1),
            "old_w_fit": old_w_fit,
            "n_gens": max_iter,
            "n_evals": ev_evals,
            "gens_to_95": float("nan"),
            "div_start": float("nan"),
            "div_end": float("nan"),
            "history": [],
            "weights": w.tolist(),
        })

    n_init_eff = min(n_init, n)
    step_at(np.arange(n_init_eff), n_init_eff - 1, initial=True)

    for t in range(n_init_eff, n):
        if (t + 1) % chunk_interval == 0:
            start = max(0, t + 1 - window_size)
            step_at(np.arange(start, t + 1), t, initial=False)

    return RunOutput(y_hat, events, detections, time.perf_counter() - t0, n_evals, w)


# ---------------------------------------------------------------------------
# Baselines online vetorizados
# ---------------------------------------------------------------------------
def model_predictions(P: np.ndarray) -> np.ndarray:
    """(n, M) predição de cada modelo individual."""
    return P.argmax(axis=2)


def rolling_model_f1(P: np.ndarray, y: np.ndarray, window: int = 500) -> np.ndarray:
    """(n, M) macro-F1 de cada modelo nas ``window`` instâncias ANTERIORES a t (estritamente passadas)."""
    n, M, C = P.shape
    pred = model_predictions(P)                                   # (n, M)
    yc = (y[:, None] == np.arange(C))[:, None, :]                 # (n, 1, C)
    pc = pred[:, :, None] == np.arange(C)[None, None, :]          # (n, M, C)
    tp = np.cumsum(yc & pc, axis=0, dtype=np.int32)
    fp = np.cumsum(~yc & pc, axis=0, dtype=np.int32)
    fn = np.cumsum(yc & ~pc, axis=0, dtype=np.int32)

    def past_window(a: np.ndarray) -> np.ndarray:
        # soma em [t-window, t-1]
        z = np.zeros((1,) + a.shape[1:], dtype=a.dtype)
        cs = np.concatenate([z, a])                                # cs[t] = soma até t-1
        lagged = np.concatenate([np.zeros((window,) + a.shape[1:], dtype=a.dtype), cs[:-window]])
        return (cs - lagged)[:n]

    tpw, fpw, fnw = past_window(tp), past_window(fp), past_window(fn)
    present = (tpw + fnw) > 0
    f1 = (2 * tpw / np.maximum(2 * tpw + fpw + fnw, 1) * present).sum(2) / np.maximum(present.sum(2), 1)
    return f1


def run_online_baseline(name: str, P: np.ndarray, y: np.ndarray, window: int = 500,
                        recompute_every: int = 100, beta: float = 0.9) -> Dict[str, Any]:
    n, M, C = P.shape
    t0 = time.perf_counter()
    extra: Dict[str, Any] = {}
    if name == "B1":
        y_hat = _predict(P, np.full(M, 1.0 / M))
    elif name == "B3":
        from src.evaluation.metrics import macro_f1
        pred = model_predictions(P)
        scores = [macro_f1(y, pred[:, m]) for m in range(M)]
        best = int(np.argmax(scores))
        y_hat = pred[:, best]
        extra = {"best_model": best, "model_scores": scores}
    elif name == "B4":
        f1 = rolling_model_f1(P, y, window)
        pred = model_predictions(P)
        choice = f1.argmax(axis=1)
        y_hat = pred[np.arange(n), choice]
        y_hat[0] = _predict(P[:1], np.full(M, 1.0 / M))[0]
    elif name == "B5":
        f1 = rolling_model_f1(P, y, window)
        y_hat = np.empty(n, dtype=np.int64)
        w = np.full(M, 1.0 / M)
        for s in range(0, n, recompute_every):
            if s > 0:
                f = f1[s]
                w = f / f.sum() if f.sum() > 0 else np.full(M, 1.0 / M)
            y_hat[s:s + recompute_every] = _predict(P[s:s + recompute_every], w)
    elif name == "B6":
        pred = model_predictions(P)
        err = (pred != y[:, None]).astype(np.float64)
        cum = np.concatenate([np.zeros((1, M)), np.cumsum(err, axis=0)[:-1]])  # erros antes de t
        logw = cum * np.log(beta)
        logw -= logw.max(axis=1, keepdims=True)
        w = np.exp(logw)
        w /= w.sum(axis=1, keepdims=True)
        y_hat = np.einsum("nm,nmc->nc", w, P).argmax(axis=1)
    else:
        raise ValueError(name)
    return {"y_hat": y_hat, "replay_time": time.perf_counter() - t0, **extra}
