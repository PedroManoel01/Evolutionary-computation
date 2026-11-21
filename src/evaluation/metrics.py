"""Métricas prequenciais, de reação ao drift e do detector (Seção 10.2 da proposta)."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np


def macro_f1(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Macro-F1 sobre as classes presentes nos rótulos verdadeiros (mesma definição da fitness)."""
    classes = np.unique(y)
    f1 = 0.0
    for c in classes:
        tp = np.sum((y_hat == c) & (y == c))
        fp = np.sum((y_hat == c) & (y != c))
        fn = np.sum((y_hat != c) & (y == c))
        f1 += 2 * tp / max(2 * tp + fp + fn, 1)
    return float(f1 / len(classes))


def kappa(y: np.ndarray, y_hat: np.ndarray, n_classes: int) -> float:
    n = len(y)
    po = float(np.mean(y == y_hat))
    yh = np.clip(y_hat, 0, n_classes - 1)
    pe = float(np.sum(np.bincount(y, minlength=n_classes) * np.bincount(yh, minlength=n_classes)) / n**2)
    return (po - pe) / (1 - pe) if pe < 1 else 0.0


def windowed_macro_f1(y: np.ndarray, y_hat: np.ndarray, n_classes: int, window: int = 1000) -> np.ndarray:
    """Macro-F1 em janela deslizante [t-window+1, t] para todo t (vetorizado por somas acumuladas)."""
    n = len(y)
    out = np.zeros(n)
    yc = (y[:, None] == np.arange(n_classes)[None, :])
    pc = (y_hat[:, None] == np.arange(n_classes)[None, :])
    tp = np.cumsum(yc & pc, axis=0)
    fp = np.cumsum(~yc & pc, axis=0)
    fn = np.cumsum(yc & ~pc, axis=0)

    def lag(a: np.ndarray) -> np.ndarray:
        b = np.zeros_like(a)
        b[window:] = a[:-window]
        return b

    tpw, fpw, fnw = tp - lag(tp), fp - lag(fp), fn - lag(fn)
    present = (tpw + fnw) > 0
    f1c = 2 * tpw / np.maximum(2 * tpw + fpw + fnw, 1)
    out = (f1c * present).sum(1) / np.maximum(present.sum(1), 1)
    return out


def prequential_summary(y: np.ndarray, y_hat: np.ndarray, n_classes: int, window: int = 1000) -> Dict[str, float]:
    wf1 = windowed_macro_f1(y, y_hat, n_classes, window)
    return {
        "macro_f1": macro_f1(y, y_hat),
        "accuracy": float(np.mean(y == y_hat)),
        "kappa": kappa(y, y_hat, n_classes),
        "wf1_mean": float(wf1[window - 1:].mean()) if len(wf1) >= window else float(wf1.mean()),
    }


def drift_reaction(wf1: np.ndarray, drift_positions: Sequence[int], threshold: float = 0.95) -> Dict[str, float]:
    """Tempo de recuperação e queda máxima após cada drift real.

    Nível pré-drift = macro-F1 em janela imediatamente antes do drift. A queda máxima é
    ``pre - min`` até o próximo drift. O tempo de recuperação é o nº de instâncias desde o
    drift até, depois do ponto mínimo, a janela voltar a ``threshold * pre`` (censurado no
    próximo drift se não recuperar).
    """
    n = len(wf1)
    rec, drops = [], []
    pos = [p for p in drift_positions if 0 < p < n]
    for i, d in enumerate(pos):
        end = pos[i + 1] if i + 1 < len(pos) else n
        pre = wf1[d - 1]
        seg = wf1[d:end]
        if len(seg) == 0:
            continue
        j_min = int(seg.argmin())
        drops.append(float(pre - seg[j_min]))
        if seg[j_min] >= threshold * pre:
            rec.append(0.0)
            continue
        after = np.nonzero(seg[j_min:] >= threshold * pre)[0]
        rec.append(float(j_min + after[0]) if len(after) else float(len(seg)))
    return {
        "recovery_time": float(np.mean(rec)) if rec else float("nan"),
        "max_drop": float(np.mean(drops)) if drops else float("nan"),
    }


def detector_metrics(detections: Sequence[int], drift_positions: Sequence[int], n: int,
                     tolerance: int = 3000, early: int = 0) -> Dict[str, float]:
    """Atraso médio, falsos alarmes e drifts perdidos.

    Uma detecção casa com o drift d se cair em [d - early, min(d + tolerance, próximo drift)).
    Só a primeira detecção casada conta; as demais detecções são falsos alarmes.
    """
    det = sorted(detections)
    pos = sorted(p for p in drift_positions if 0 < p < n)
    used = set()
    delays: List[float] = []
    missed = 0
    for i, d in enumerate(pos):
        hi = min(d + tolerance, pos[i + 1] - early if i + 1 < len(pos) else n)
        match: Optional[int] = None
        for k, t in enumerate(det):
            if k in used:
                continue
            if d - early <= t < hi:
                match = k
                break
        if match is None:
            missed += 1
        else:
            used.add(match)
            delays.append(det[match] - d)
    return {
        "n_detections": float(len(det)),
        "detection_delay": float(np.mean(delays)) if delays else float("nan"),
        "false_alarms": float(len(det) - len(used)),
        "missed_drifts": float(missed),
    }
