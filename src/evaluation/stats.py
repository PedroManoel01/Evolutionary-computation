"""Análise estatística (Seção 10.3): Friedman + Nemenyi (Demšar, 2006), Wilcoxon pareado
com correção de Holm e tamanho de efeito Â12 de Vargha-Delaney."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

# Valores críticos q_α (α = 0,05) do teste de Nemenyi (Demšar, 2006, Tabela 5a), k = 2..20.
_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102,
        10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391, 16: 3.426,
        17: 3.458, 18: 3.489, 19: 3.517, 20: 3.544}


def a12(x: Sequence[float], y: Sequence[float]) -> float:
    """Â12 de Vargha-Delaney: P(X > Y) + 0,5 P(X = Y). 0,5 = sem efeito."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    gt = (x[:, None] > y[None, :]).sum()
    eq = (x[:, None] == y[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(x) * len(y)))


def a12_label(v: float) -> str:
    d = abs(v - 0.5)
    if d < 0.06:
        return "desprezível"
    if d < 0.14:
        return "pequeno"
    if d < 0.21:
        return "médio"
    return "grande"


def holm(pvals: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for i, k in enumerate(order):
        run = max(run, (m - i) * p[k])
        adj[k] = min(run, 1.0)
    return adj


def wilcoxon_p(x: Sequence[float], y: Sequence[float]) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if np.allclose(x, y):
        return 1.0
    return float(stats.wilcoxon(x, y, zero_method="zsplit").pvalue)


def friedman_nemenyi(table: pd.DataFrame, higher_is_better: bool = True) -> Dict[str, object]:
    """``table``: linhas = blocos (bases), colunas = métodos. Retorna ranks médios, p-valor e CD."""
    vals = table.values
    ranks = pd.DataFrame(table).rank(axis=1, ascending=not higher_is_better).mean(axis=0)
    N, k = vals.shape
    p = float(stats.friedmanchisquare(*[vals[:, j] for j in range(k)]).pvalue) if k >= 3 else float("nan")
    cd = _Q05.get(k, np.nan) * np.sqrt(k * (k + 1) / (6.0 * N))
    return {"avg_ranks": ranks.sort_values(), "p_value": p, "cd": float(cd), "N": N, "k": k}


def cd_diagram(avg_ranks: pd.Series, cd: float, title: str, path: str,
               p_value: Optional[float] = None) -> None:
    """Diagrama de diferença crítica no estilo de Demšar (2006)."""
    import matplotlib.pyplot as plt

    r = avg_ranks.sort_values()
    names, vals = list(r.index), list(r.values)
    k = len(names)
    lo, hi = 1, max(k, int(np.ceil(max(vals))))
    half = (k + 1) // 2
    fig_h = 1.6 + 0.28 * half
    fig, ax = plt.subplots(figsize=(9, fig_h))
    ax.set_xlim(lo - 0.5, hi + 0.5)
    ax.set_ylim(0, fig_h)
    ax.axis("off")
    top = fig_h - 1.0
    ax.hlines(top, lo, hi, color="k", lw=1)
    for i in range(lo, hi + 1):
        ax.vlines(i, top, top + 0.1, color="k", lw=1)
        ax.text(i, top + 0.15, str(i), ha="center", va="bottom", fontsize=9)
    # barra do CD
    ax.hlines(top + 0.6, lo, min(lo + cd, hi), color="tab:red", lw=2)
    ax.text(lo + min(cd, hi - lo) / 2, top + 0.65, f"CD = {cd:.2f}", ha="center", va="bottom",
            fontsize=9, color="tab:red")
    for i, (nm, v) in enumerate(zip(names, vals)):
        if i < half:
            y = top - 0.35 - 0.28 * i
            ax.plot([v, v, lo - 0.3], [top, y, y], color="k", lw=0.8)
            ax.text(lo - 0.35, y, f"{nm} ({v:.2f})", ha="right", va="center", fontsize=9)
        else:
            j = k - 1 - i
            y = top - 0.35 - 0.28 * j
            ax.plot([v, v, hi + 0.3], [top, y, y], color="k", lw=0.8)
            ax.text(hi + 0.35, y, f"({v:.2f}) {nm}", ha="left", va="center", fontsize=9)
    # grupos não significativamente diferentes
    groups: List[tuple] = []
    for i in range(k):
        j = i
        while j + 1 < k and vals[j + 1] - vals[i] <= cd:
            j += 1
        if j > i and not any(g[0] <= i and j <= g[1] for g in groups):
            groups.append((i, j))
    for gi, (i, j) in enumerate(groups):
        y = top - 0.12 - 0.07 * gi
        ax.hlines(y, vals[i] - 0.03, vals[j] + 0.03, color="tab:blue", lw=3)
    t = title + (f"  (Friedman p = {p_value:.2g})" if p_value is not None else "")
    ax.set_title(t, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
