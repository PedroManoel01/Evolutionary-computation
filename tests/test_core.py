"""Testes unitários do núcleo: reparo no simplex, macro-F1 vs sklearn, operadores, estratégias e métricas."""

import numpy as np
import pytest
from sklearn.metrics import f1_score

from src.evaluation.metrics import detector_metrics, drift_reaction, macro_f1, windowed_macro_f1
from src.evaluation.objectives import macro_f1_batch, repair_simplex
from src.evaluation.replay import ReplayConfig, rolling_model_f1, run_online_baseline, run_triggered
from src.evolution.ga import DifferentialEvolution, GAConfig, GeneticAlgorithm, RandomSearch, random_simplex
from src.strategies.post_drift import MemoryArchive, PostDriftStrategy


def test_repair_simplex():
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, size=(100, 8))
    r = repair_simplex(x)
    assert np.allclose(r.sum(1), 1.0)
    assert (r >= 0).all()
    assert np.allclose(repair_simplex(np.zeros(8)), np.full(8, 1 / 8))


@pytest.mark.parametrize("C", [2, 3, 5])
def test_macro_f1_matches_sklearn(C):
    rng = np.random.default_rng(C)
    n, M = 400, 8
    P = rng.dirichlet(np.ones(C), size=(n, M))
    y = rng.integers(0, C, size=n)
    pop = random_simplex(20, M, rng)
    f = macro_f1_batch(pop, P, y)
    for i in range(20):
        yhat = np.einsum("m,nmc->nc", pop[i], P).argmax(1)
        ref = f1_score(y, yhat, average="macro", labels=np.unique(y), zero_division=0)
        assert f[i] == pytest.approx(ref, abs=1e-12)
        assert macro_f1(y, yhat) == pytest.approx(ref, abs=1e-12)


def test_ga_stays_in_simplex_and_improves():
    rng = np.random.default_rng(1)
    n, M, C = 500, 8, 3
    y = rng.integers(0, C, size=n)
    P = rng.dirichlet(np.ones(C), size=(n, M))
    P[:, 3, :] = 0.05
    P[np.arange(n), 3, y] = 0.9           # o modelo 3 é quase perfeito
    fit = lambda pop: macro_f1_batch(pop, P, y)  # noqa: E731
    for opt in [GeneticAlgorithm(GAConfig()), GeneticAlgorithm(GAConfig(survivor="mu+lambda")),
                DifferentialEvolution(), RandomSearch()]:
        res = opt.run(random_simplex(30, M, rng), fit, rng)
        assert np.allclose(res.pop.sum(1), 1.0) and (res.pop >= 0).all()
        assert res.best_fit >= res.init_best_fit
        assert res.n_evals <= 1500
        assert res.best_fit > 0.9


def test_memory_replaces_most_similar():
    mem = MemoryArchive(capacity=2)
    a, b = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
    mem.insert(a)
    mem.insert(b)
    mem.insert(np.array([0.9, 0.1, 0]))
    assert len(mem.items) == 2
    assert np.allclose(mem.items[0], [0.9, 0.1, 0]) and np.allclose(mem.items[1], b)


@pytest.mark.parametrize("s", ["S1", "S2", "S3", "S4", "S5", "S6"])
def test_strategies_population_size(s):
    rng = np.random.default_rng(2)
    st = PostDriftStrategy(s)
    prev = random_simplex(30, 8, rng)
    fit = rng.random(30)
    for _ in range(12):
        st.on_drift(random_simplex(1, 8, rng)[0])
    pop = st.initial_population(prev, fit, 8, rng)
    assert pop.shape == (30, 8)
    assert np.allclose(pop.sum(1), 1.0)
    if s == "S2":
        assert np.allclose(pop[0], prev[fit.argmax()])
    if s in {"S5", "S6"}:
        assert len(st.memory.items) == 10


def test_windowed_f1_and_reaction():
    y = np.array([0, 1] * 1000)
    yh = y.copy()
    yh[1000:1200] = 1 - yh[1000:1200]
    wf = windowed_macro_f1(y, yh, 2, window=100)
    assert wf[999] == pytest.approx(1.0)
    r = drift_reaction(wf, [1000])
    assert r["max_drop"] == pytest.approx(1.0)
    assert 1100 < 1000 + r["recovery_time"] < 1400


def test_detector_metrics():
    m = detector_metrics([1100, 1500, 5000], [1000, 3000], n=6000, tolerance=1000)
    assert m["detection_delay"] == 100 and m["false_alarms"] == 2 and m["missed_drifts"] == 1


def test_rolling_f1_is_strictly_past():
    rng = np.random.default_rng(3)
    P = rng.dirichlet(np.ones(2), size=(50, 3))
    y = rng.integers(0, 2, 50)
    f1a = rolling_model_f1(P, y, 10)
    y2 = y.copy()
    y2[30] = 1 - y2[30]
    f1b = rolling_model_f1(P, y2, 10)
    assert np.allclose(f1a[:31], f1b[:31])


def test_replay_runs_and_reacts():
    rng = np.random.default_rng(4)
    n, M, C = 6000, 4, 2
    y = rng.integers(0, C, n)
    P = np.full((n, M, C), 0.5)
    good = np.where(np.arange(n) < 3000, 0, 1)    # modelo 0 bom antes, modelo 1 bom depois
    for t in range(n):
        P[t, good[t], :] = 0.1
        P[t, good[t], y[t]] = 0.9
        P[t, 2, :] = rng.dirichlet([1, 1])
    out = run_triggered(P, y, [3000], ReplayConfig(strategy="S3", detector="oracle"), seed=0)
    assert len(out.events) == 2
    assert out.events[1]["t_apply"] == 3000 + 100
    assert np.argmax(out.weights_final) == 1
    for b in ["B1", "B3", "B4", "B5", "B6"]:
        assert len(run_online_baseline(b, P, y)["y_hat"]) == n
