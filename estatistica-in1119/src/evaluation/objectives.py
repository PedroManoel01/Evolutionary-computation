"""Funções objetivo vetorizadas e reparo no simplex Δ^M.

Referência: Seções 5.1 e 5.3 de Proposta_Projeto_CE.pdf.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def repair_simplex(weights: np.ndarray) -> np.ndarray:
    """Projeta/normaliza um vetor (ou lote de vetores) para o simplex Δ^M.

    Valores negativos são truncados para 0; se a soma for 0, usa pesos uniformes.
    """
    arr = np.asarray(weights, dtype=float)
    single = arr.ndim == 1
    if single:
        arr = arr[np.newaxis, :]

    arr = np.clip(arr, 0.0, None)
    sums = arr.sum(axis=1, keepdims=True)
    n_models = arr.shape[1]
    uniform = np.full((1, n_models), 1.0 / n_models, dtype=float)
    repaired = np.where(sums > 0.0, arr / np.where(sums > 0.0, sums, 1.0), uniform)

    return repaired[0] if single else repaired


def macro_f1_batch(
    pop: np.ndarray,
    P: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Calcula o macro-F1 de uma população de vetores de pesos de forma vetorizada.

    Args:
        pop: Array (N, M) de pesos no simplex.
        P: Array (n, M, C) de probabilidades previstas por cada modelo.
        y: Array (n,) de rótulos inteiros (0 .. C-1).

    Returns:
        Array (N,) com o macro-F1 de cada indivíduo.
    """
    w = repair_simplex(pop)
    if w.ndim == 1:
        w = w[np.newaxis, :]

    scores = np.einsum("nm,imc->nic", w, P)  # (N, n, C)
    y_hat = scores.argmax(axis=2)  # (N, n)

    classes = np.unique(y)
    f1 = np.zeros(w.shape[0], dtype=float)
    for c in classes:
        tp = ((y_hat == c) & (y == c)).sum(axis=1)
        fp = ((y_hat == c) & (y != c)).sum(axis=1)
        fn = ((y_hat != c) & (y == c)).sum(axis=1)
        denom = np.maximum(2 * tp + fp + fn, 1)
        f1 += (2.0 * tp) / denom

    return f1 / len(classes)


def make_macro_f1_objective(
    P: np.ndarray,
    y: np.ndarray,
    minimize: bool = True,
) -> Callable[[np.ndarray], np.ndarray]:
    """Cria uma função objetivo caixa-preta f(W) sobre o conjunto de avaliação (P, y).

    Args:
        P: Matriz (n, M, C) de probabilidades do buffer.
        y: Vetor (n,) de rótulos reais do buffer.
        minimize: Se True, retorna `1.0 - macro_f1` (padrão para minimizadores).
    """

    def objective(weights: np.ndarray) -> np.ndarray:
        arr = np.asarray(weights, dtype=float)
        is_single = arr.ndim == 1
        scores = macro_f1_batch(arr, P, y)
        out = (1.0 - scores) if minimize else scores
        return out[0] if is_single else out

    return objective
