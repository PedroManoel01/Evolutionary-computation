"""Algoritmo Genético de codificação real no simplex Δ^M (Seção 5 da proposta).

Representação: g ∈ [0,1]^M, reparado para g/Σg após cada operador.
Operadores padrão: torneio k=3, BLX-α (α=0,5; pc=0,9), mutação gaussiana por gene
(pm=1/M; σ=0,1) com clip em [0,1], substituição geracional com elitismo de 2.
Parada: G=50 gerações ou 10 gerações sem melhoria do melhor.

Também contém a busca aleatória (B7) e reexporta ``DifferentialEvolution`` (``de.py``)
por compatibilidade; todos compartilham a interface ``run(pop0, fitness, rng) -> OptResult``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.evaluation.objectives import repair_simplex
from src.evolution.base import FitnessFn, OptResult, mean_pairwise_distance, random_simplex
from src.evolution.de import DEConfig, DifferentialEvolution


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
