"""Evolução Diferencial (DE) no simplex Δ^M (Storn & Price, 1997).

Implementa DE como otimizador evolucionário de primeira classe com a mesma interface
``run(pop0, fitness, rng, hypermutation=False) -> OptResult`` do Algoritmo Genético,
suportando múltiplas estratégias de mutação/cruzamento, dither opcional de F e
hipermutação pós-drift (S4) compatível com populações warm-start convergidas.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.evaluation.objectives import repair_simplex
from src.evolution.base import FitnessFn, OptResult, mean_pairwise_distance

DE_MUTATIONS = {"rand/1", "best/1", "current-to-best/1", "rand/2"}
DE_CROSSOVERS = {"bin", "exp"}
DE_STRATEGIES = {f"{m}/{c}" for m in DE_MUTATIONS for c in DE_CROSSOVERS}


@dataclass
class DEConfig:
    pop_size: int = 30
    max_gens: int = 50
    stagnation: int = 10
    F: float = 0.5                  # fator de escala (mutação diferencial)
    CR: float = 0.9                 # taxa de cruzamento
    strategy: str = "rand/1/bin"    # rand/1/bin, best/1/bin, current-to-best/1/bin, rand/2/bin (ou /exp)
    dither: bool = False            # se True, amostra F ~ U(0.5, 1.0) a cada geração
    hyper_F: float = 0.9            # S4: fator F durante a hipermutação
    hyper_sigma: float = 0.25       # S4: perturbação gaussiana extra para escapar de colapso populacional
    hyper_gens: int = 10            # S4: nº de gerações com hipermutação

    def __post_init__(self) -> None:
        if self.strategy not in DE_STRATEGIES:
            raise ValueError(
                f"Estratégia DE desconhecida: {self.strategy!r}. "
                f"Opções válidas: {sorted(DE_STRATEGIES)}"
            )


class DifferentialEvolution:
    """Evolução Diferencial com reparo no simplex Δ^M."""

    def __init__(
        self,
        cfg: DEConfig | None = None,
        *,
        F: float | None = None,
        CR: float | None = None,
        max_gens: int | None = None,
        stagnation: int | None = None,
        strategy: str | None = None,
        dither: bool | None = None,
    ) -> None:
        base = cfg or DEConfig()
        self.cfg = DEConfig(
            pop_size=base.pop_size,
            max_gens=base.max_gens if max_gens is None else max_gens,
            stagnation=base.stagnation if stagnation is None else stagnation,
            F=base.F if F is None else F,
            CR=base.CR if CR is None else CR,
            strategy=base.strategy if strategy is None else strategy,
            dither=base.dither if dither is None else dither,
            hyper_F=base.hyper_F,
            hyper_sigma=base.hyper_sigma,
            hyper_gens=base.hyper_gens,
        )
        self._mut_name, self._cx_name = self.cfg.strategy.rsplit("/", 1)

    # --- operadores -----------------------------------------------------------
    @staticmethod
    def _sample_distinct(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
        """Para cada i ∈ [0, n), amostra k índices distintos diferentes de i."""
        if n <= k:
            raise ValueError(f"Tamanho da população ({n}) deve ser > {k} para esta estratégia DE.")
        pool = np.arange(n)
        return np.array([rng.choice(np.delete(pool, i), k, replace=False) for i in range(n)])

    def _mutate(
        self,
        pop: np.ndarray,
        fit: np.ndarray,
        F: float,
        sigma_extra: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        N = len(pop)
        best = pop[int(fit.argmax())]
        if self._mut_name == "rand/1":
            idx = self._sample_distinct(N, 3, rng)
            mutant = pop[idx[:, 0]] + F * (pop[idx[:, 1]] - pop[idx[:, 2]])
        elif self._mut_name == "best/1":
            idx = self._sample_distinct(N, 2, rng)
            mutant = best + F * (pop[idx[:, 0]] - pop[idx[:, 1]])
        elif self._mut_name == "current-to-best/1":
            idx = self._sample_distinct(N, 2, rng)
            mutant = pop + F * (best - pop) + F * (pop[idx[:, 0]] - pop[idx[:, 1]])
        elif self._mut_name == "rand/2":
            idx = self._sample_distinct(N, 5, rng)
            mutant = (
                pop[idx[:, 0]]
                + F * (pop[idx[:, 1]] - pop[idx[:, 2]])
                + F * (pop[idx[:, 3]] - pop[idx[:, 4]])
            )
        else:  # pragma: no cover
            raise ValueError(self._mut_name)

        if sigma_extra > 0.0:
            mutant = mutant + rng.normal(0.0, sigma_extra, size=mutant.shape)
        return mutant

    def _crossover(self, pop: np.ndarray, mutant: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        N, M = pop.shape
        CR = self.cfg.CR
        if self._cx_name == "bin":
            cross = rng.random((N, M)) < CR
            cross[np.arange(N), rng.integers(0, M, N)] = True
        else:  # exp
            cross = np.zeros((N, M), dtype=bool)
            starts = rng.integers(0, M, N)
            for i in range(N):
                j = int(starts[i])
                L = 0
                while L < M:
                    cross[i, (j + L) % M] = True
                    L += 1
                    if rng.random() >= CR:
                        break
        return np.where(cross, mutant, pop)

    # --- laço principal -------------------------------------------------------
    def run(
        self,
        pop0: np.ndarray,
        fitness: FitnessFn,
        rng: np.random.Generator,
        hypermutation: bool = False,
    ) -> OptResult:
        cfg = self.cfg
        pop = repair_simplex(np.asarray(pop0, dtype=float))
        N = len(pop)
        fit = fitness(pop)
        n_evals = N
        hist = [float(fit.max())]
        div_start = mean_pairwise_distance(pop)
        no_improve = 0
        gen = 0

        for gen in range(1, cfg.max_gens):
            in_hyper = hypermutation and gen <= cfg.hyper_gens
            if in_hyper:
                F_gen = cfg.hyper_F
                sigma_extra = cfg.hyper_sigma
            else:
                F_gen = float(rng.uniform(0.5, 1.0)) if cfg.dither else cfg.F
                sigma_extra = 0.0

            mutant = self._mutate(pop, fit, F_gen, sigma_extra, rng)
            trial = repair_simplex(np.clip(self._crossover(pop, mutant, rng), 0.0, 1.0))
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
            if no_improve >= cfg.stagnation:
                break

        b = int(fit.argmax())
        return OptResult(
            pop=pop,
            fit=fit,
            best=pop[b].copy(),
            best_fit=float(fit[b]),
            history=hist,
            n_evals=n_evals,
            n_gens=gen,
            div_start=div_start,
            div_end=mean_pairwise_distance(pop),
            init_best_fit=hist[0],
        )
