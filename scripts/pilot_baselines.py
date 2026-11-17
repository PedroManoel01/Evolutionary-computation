"""Script piloto para validar DETO e DASE (com detector HCDD original) em streams sintéticas.

Uso:
    python scripts/pilot_baselines.py --dataset SEA-Rec --detector adwin_pair --max-instances 15000
"""

from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Tuple

import numpy as np

# Garante que a raiz do projeto esteja no sys.path ao rodar como script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.dase_adapter import DASEAdapter
from src.baselines.deto_adapter import DETOAdapter
from src.datasets import build_dataset
from src.detectors import WarningBuffer, build_detector
from src.evaluation.objectives import make_macro_f1_objective
from src.models import WeightedEnsemble
from src.utils import WandbLogger


def probs_dict_to_matrix(
    pool_probs: Dict[str, Dict[Any, float]],
    model_names: List[str],
    n_classes: int,
) -> np.ndarray:
    """Converte `{modelo: {classe: prob}}` em matriz `(M, C)`."""
    mat = np.zeros((len(model_names), n_classes), dtype=float)
    for m_idx, name in enumerate(model_names):
        probs = pool_probs.get(name, {})
        for c in range(n_classes):
            mat[m_idx, c] = float(probs.get(c, 0.0))
    return mat


def run_pilot(
    dataset_name: str = "SEA-Rec",
    detector_name: str = "adwin_pair",
    max_instances: int = 15000,
    dase_chunk_interval: int = 1000,
    seed: int = 42,
    use_wandb: bool = False,
) -> None:
    dataset = build_dataset(dataset_name, seed=seed)
    ensemble = WeightedEnsemble(seed=seed)
    n_models = ensemble.n_models

    detector_kwargs: Dict[str, Any] = {}
    if detector_name.lower() == "oracle":
        detector_kwargs["drift_positions"] = dataset.drift_positions
        detector_kwargs["warning_window"] = 200

    detector = build_detector(detector_name, **detector_kwargs)
    buffer = WarningBuffer(max_size=2000, min_eval_size=100)

    deto = DETOAdapter(n_models=n_models, n_init=12, n_iter=10, seed=seed)
    dase = DASEAdapter(n_models=n_models, chunk_size=50, pop_size=40, max_iter=20, seed=seed)
    dase_window: Deque[Tuple[np.ndarray, int]] = deque(maxlen=500)

    logger = WandbLogger(
        project="dyn-ea-ensemble-drift",
        name=f"pilot-{dataset_name}-{detector_name}",
        config={
            "dataset": dataset_name,
            "detector": detector_name,
            "max_instances": max_instances,
            "seed": seed,
        },
        enabled=use_wandb,
    )

    correct = 0
    total = 0
    deto_drifts = 0
    dase_drifts = 0

    try:
        for t, (x, y) in enumerate(dataset):
            if t >= max_instances:
                break

            y_int = int(y)
            pool_probs = ensemble.predict_proba_pool(x)
            p_t = probs_dict_to_matrix(pool_probs, ensemble.model_names, dataset.n_classes)
            dase_window.append((p_t, y_int))

            y_hat = ensemble.predict_one(x)
            err = 1 if y_hat != y_int else 0
            correct += 1 - err
            total += 1

            detector.update(err)

            if buffer.awaiting_post_drift:
                buffer.add_post_drift_sample(p_t, y_int)
            elif detector.drift_detected:
                buffer.on_drift(p_t, y_int)
            elif detector.warning_detected:
                buffer.add_warning_sample(p_t, y_int)
            else:
                buffer.on_false_warning()

            # 1) DETO acionado pelo detector + WarningBuffer
            if buffer.is_ready():
                deto_drifts += 1
                P_eval, y_eval = buffer.get_evaluation_set()
                obj_deto = make_macro_f1_objective(P_eval, y_eval, minimize=True)

                w_deto, f1_deto = deto.optimize(obj_deto)
                ensemble.set_weights(w_deto)

                print(
                    f"[t={t:5d} | DETO drift #{deto_drifts} | |E|={len(y_eval):4d}] "
                    f"DETO macro-F1={f1_deto:.4f}"
                )
                logger.log(
                    {
                        "deto_drifts": deto_drifts,
                        "deto_eval_size": len(y_eval),
                        "deto_buffer_macro_f1": f1_deto,
                        "prequential_acc": correct / total,
                    },
                    step=t,
                )

                buffer.clear()
                detector.reset()

            # 2) DASE rodando de forma autônoma com seu detector HCDD original
            if (t + 1) % dase_chunk_interval == 0 and len(dase_window) >= 100:
                P_dase = np.stack([item[0] for item in dase_window], axis=0)
                y_dase = np.asarray([item[1] for item in dase_window])
                obj_dase = make_macro_f1_objective(P_dase, y_dase, minimize=True)

                _, f1_dase, hcdd_drift = dase.step(obj_dase)
                if hcdd_drift:
                    dase_drifts += 1

                print(
                    f"[t={t:5d} | DASE step | HCDD drift={hcdd_drift}] "
                    f"DASE macro-F1={f1_dase:.4f}"
                )
                logger.log(
                    {
                        "dase_drifts_hcdd": dase_drifts,
                        "dase_window_macro_f1": f1_dase,
                        "dase_hcdd_triggered": int(hcdd_drift),
                    },
                    step=t,
                )

            ensemble.learn_one(x, y_int)

        acc = correct / max(total, 1)
        print(
            f"\nPiloto finalizado ({dataset_name}, {total} instâncias): "
            f"acurácia prequencial={acc:.4f} | drifts DETO={deto_drifts} | drifts HCDD(DASE)={dase_drifts}"
        )
        logger.log_summary(
            {
                "prequential_acc": acc,
                "deto_drifts": deto_drifts,
                "dase_drifts_hcdd": dase_drifts,
            }
        )
    finally:
        logger.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Piloto de validação de DETO e DASE (HCDD original) em streams sintéticas."
    )
    parser.add_argument("--dataset", type=str, default="SEA-Rec")
    parser.add_argument(
        "--detector",
        type=str,
        default="adwin_pair",
        choices=["adwin_pair", "ddm", "oracle"],
    )
    parser.add_argument("--max-instances", type=int, default=15000)
    parser.add_argument("--dase-chunk-interval", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pilot(
        dataset_name=args.dataset,
        detector_name=args.detector,
        max_instances=args.max_instances,
        dase_chunk_interval=args.dase_chunk_interval,
        seed=args.seed,
        use_wandb=args.wandb,
    )
