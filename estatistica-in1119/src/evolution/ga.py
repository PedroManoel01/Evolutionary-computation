"""Algoritmo Genético de codificação real no simplex Δ^M (Seção 5 da proposta).

Representação: g ∈ [0,1]^M, reparado para g/Σg após cada operador.
Operadores padrão: torneio k=3, BLX-α (α=0,5; pc=0,9), mutação gaussiana por gene
(pm=1/M; σ=0,1) com clip em [0,1], substituição geracional com elitismo de 2.
Parada: G=50 gerações ou 10 gerações sem melhoria do melhor.

Também contém a Evolução Diferencial DE/rand/1/bin (baseline B8) e a busca aleatória (B7),
que compartilham a mesma interface ``run(pop0, fitness, rng) -> OptResult``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from src.evaluation.objectives import repair_simplex

FitnessFn = Callable[[np.ndarray], np.ndarray]


def random_simplex(n: int, m: int, rng: np.random.Generator) -> np.ndarray:
    """Amostragem uniforme no simplex: Dirichlet(1, ..., 1)."""
    return rng.dirichlet(np.ones(m), size=n)


def mean_pairwise_distance(pop: np.ndarray) -> float:
    """Diversidade: distância euclidiana média entre pares de indivíduos."""
    n = len(pop)
    if n < 2:
        return 0.0
    d = np.sqrt(((pop[:, None, :] - pop[None, :, :]) ** 2).sum(-1))
    return float(d[np.triu_indices(n, 1)].mean())


@dataclass
class OptResult:
    pop: np.ndarray            # população final (N, M)
    fit: np.ndarray            # fitness da população final (N,)
    best: np.ndarray           # melhor indivíduo (M,)
    best_fit: float
    history: List[float]       # melhor fitness por geração (gen 0 = população inicial)
    n_evals: int
    n_gens: int
    div_start: float
    div_end: float
    init_best_fit: float = field(default=0.0)


@dataclass
class GAConfig:
    pop_size: int = 30
    max_gens: int = 50
    stagnation: int = 10
    tournament_k: int = 3
    blx_alpha: float = 0.5
    pc: float = 0.9
    pm: Optional[float] = None       # None -> 1/M
    sigma: float = 0.1
    elitism: int = 2
    survivor: str = "generational"  # ou "mu+lambda"
    hyper_sigma: float = 0.5        # S4: σ durante a hipermutação
    hyper_gens: int = 10            # S4: nº de gerações com hipermutação


class GeneticAlgorithm:
    def __init__(self, cfg: GAConfig | None = None) -> None:
        self.cfg = cfg or GAConfig()

    # --- operadores -----------------------------------------------------------
    def _tournament(self, fit: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
        cand = rng.integers(0, len(fit), size=(n, self.cfg.tournament_k))
        winners = cand[np.arange(n), fit[cand].argmax(axis=1)]
        return winners

    def _blx(self, a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        d = hi - lo
        alpha = self.cfg.blx_alpha
        return rng.uniform(lo - alpha * d, hi + alpha * d)

    def _mutate(self, x: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
        m = x.shape[1]
        pm = self.cfg.pm if self.cfg.pm is not None else 1.0 / m
        mask = rng.random(x.shape) < pm
        x = x + mask * rng.normal(0.0, sigma, size=x.shape)
        return np.clip(x, 0.0, 1.0)

    def _offspring(self, pop: np.ndarray, fit: np.ndarray, n_off: int, sigma: float,
                   rng: np.random.Generator) -> np.ndarray:
        n_pairs = (n_off + 1) // 2
        pa = pop[self._tournament(fit, n_pairs, rng)]
        pb = pop[self._tournament(fit, n_pairs, rng)]
        do_cx = rng.random(n_pairs) < self.cfg.pc
        c1 = np.where(do_cx[:, None], self._blx(pa, pb, rng), pa)
        c2 = np.where(do_cx[:, None], self._blx(pa, pb, rng), pb)
        kids = np.concatenate([c1, c2])[:n_off]
        kids = np.clip(kids, 0.0, 1.0)
        kids = self._mutate(kids, sigma, rng)
        return repair_simplex(kids)

    # --- laço principal -----------------------------------------------------------
    def run(self, pop0: np.ndarray, fitness: FitnessFn, rng: np.random.Generator,
            hypermutation: bool = False) -> OptResult:
        cfg = self.cfg
        pop = repair_simplex(np.asarray(pop0, dtype=float))
        N = len(pop)
        fit = fitness(pop)
        n_evals = N
        best_hist = [float(fit.max())]
        div_start = mean_pairwise_distance(pop)
        no_improve = 0
        gen = 0
        for gen in range(1, cfg.max_gens):
            sigma = cfg.hyper_sigma if (hypermutation and gen <= cfg.hyper_gens) else cfg.sigma
            if cfg.survivor == "mu+lambda":
                kids = self._offspring(pop, fit, N, sigma, rng)
                kfit = fitness(kids)
                n_evals += N
                allp = np.concatenate([pop, kids])
                allf = np.concatenate([fit, kfit])
                idx = np.argsort(-allf, kind="stable")[:N]
                pop, fit = allp[idx], allf[idx]
            else:
                elite_idx = np.argsort(-fit, kind="stable")[: cfg.elitism]
                kids = self._offspring(pop, fit, N - cfg.elitism, sigma, rng)
                kfit = fitness(kids)
                n_evals += len(kids)
                pop = np.concatenate([pop[elite_idx], kids])
                fit = np.concatenate([fit[elite_idx], kfit])
            cur = float(fit.max())
            if cur > best_hist[-1] + 1e-12:
                no_improve = 0
            else:
                no_improve += 1
            best_hist.append(max(cur, best_hist[-1]))
            if no_improve >= cfg.stagnation:
                break
        b = int(fit.argmax())
        return OptResult(pop=pop, fit=fit, best=pop[b].copy(), best_fit=float(fit[b]),
                         history=best_hist, n_evals=n_evals, n_gens=gen,
                         div_start=div_start, div_end=mean_pairwise_distance(pop),
                         init_best_fit=best_hist[0])


class DifferentialEvolution:
    """DE/rand/1/bin (Storn & Price, 1997) com reparo no simplex — baseline B8."""

    def __init__(self, F: float = 0.5, CR: float = 0.9, max_gens: int = 50,
                 stagnation: int = 10) -> None:
        self.F, self.CR, self.max_gens, self.stagnation = F, CR, max_gens, stagnation

    def run(self, pop0: np.ndarray, fitness: FitnessFn, rng: np.random.Generator,
            hypermutation: bool = False) -> OptResult:  # noqa: ARG002
        pop = repair_simplex(np.asarray(pop0, dtype=float))
        N, M = pop.shape
        fit = fitness(pop)
        n_evals = N
        hist = [float(fit.max())]
        div_start = mean_pairwise_distance(pop)
        no_improve = 0
        gen = 0
        for gen in range(1, self.max_gens):
            idx = np.array([rng.choice(np.delete(np.arange(N), i), 3, replace=False) for i in range(N)])
            mutant = pop[idx[:, 0]] + self.F * (pop[idx[:, 1]] - pop[idx[:, 2]])
            cross = rng.random((N, M)) < self.CR
            cross[np.arange(N), rng.integers(0, M, N)] = True
            trial = repair_simplex(np.clip(np.where(cross, mutant, pop), 0.0, 1.0))
            tfit = fitness(trial)
            n_evals += N
            better = tfit >= fit
            pop = np.where(better[:, None], trial, pop)
            fit = np.where(better, tfit, fit)
            cur = float(fit.max())
            if cur > hist[-1] + 1e-12:
                no_improve = 0
            else:
                no_improve += 1
            hist.append(max(cur, hist[-1]))
            if no_improve >= self.stagnation:
                break
        b = int(fit.argmax())
        return OptResult(pop=pop, fit=fit, best=pop[b].copy(), best_fit=float(fit[b]),
                         history=hist, n_evals=n_evals, n_gens=gen,
                         div_start=div_start, div_end=mean_pairwise_distance(pop),
                         init_best_fit=hist[0])


class RandomSearch:
    """B7: mesmo gatilho, mesmo E e mesmo orçamento (1500 vetores Dirichlet); fica com o melhor."""

    def __init__(self, budget: int = 1500, keep: int = 30) -> None:
        self.budget, self.keep = budget, keep

    def run(self, pop0: np.ndarray, fitness: FitnessFn, rng: np.random.Generator,
            hypermutation: bool = False) -> OptResult:  # noqa: ARG002
        M = pop0.shape[1]
        cand = random_simplex(self.budget, M, rng)
        fit = fitness(cand)
        order = np.argsort(-fit, kind="stable")[: self.keep]
        hist = list(np.maximum.accumulate(fit.reshape(-1, 30).max(1))) if self.budget % 30 == 0 else [float(fit.max())]
        b = int(fit.argmax())
        return OptResult(pop=cand[order], fit=fit[order], best=cand[b].copy(), best_fit=float(fit[b]),
                         history=[float(h) for h in hist], n_evals=self.budget, n_gens=len(hist) - 1,
                         div_start=float("nan"), div_end=mean_pairwise_distance(cand[order]),
                         init_best_fit=float(hist[0]))
