"""Wrapper utilitário para integração com Weights & Biases (wandb).

Permite habilitar/desabilitar logging de forma transparente sem acoplar
os experimentos diretamente à API do wandb.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


class WandbLogger:
    """Gerenciador de execuções e métricas no Weights & Biases."""

    def __init__(
        self,
        project: str = "dyn-ea-ensemble-drift",
        entity: Optional[str] = None,
        name: Optional[str] = None,
        group: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        mode: str = "online",
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled and mode != "disabled"
        self.run = None

        if not self.enabled:
            return

        import wandb

        self._wandb = wandb
        self.run = wandb.init(
            project=project,
            entity=entity,
            name=name,
            group=group,
            tags=list(tags) if tags is not None else None,
            config=config,
            mode=mode,
        )

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Registra um dicionário de métricas escalares."""
        if not self.enabled or self.run is None:
            return
        self.run.log(metrics, step=step)

    def log_summary(self, summary: Dict[str, Any]) -> None:
        """Registra métricas finais no resumo da execução."""
        if not self.enabled or self.run is None:
            return
        for key, value in summary.items():
            self.run.summary[key] = value

    def log_table(
        self,
        key: str,
        columns: List[str],
        rows: Iterable[Sequence[Any]],
    ) -> None:
        """Registra uma tabela estruturada (ex.: resultados por base/semente)."""
        if not self.enabled or self.run is None:
            return
        table = self._wandb.Table(columns=columns, data=[list(r) for r in rows])
        self.run.log({key: table})

    def log_artifact(
        self,
        file_path: str | Path,
        name: str,
        artifact_type: str = "dataset",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Salva um arquivo ou diretório (ex.: CSV, .npy, figura) como Artifact."""
        if not self.enabled or self.run is None:
            return
        artifact = self._wandb.Artifact(
            name=name,
            type=artifact_type,
            metadata=metadata or {},
        )
        path = Path(file_path)
        if path.is_dir():
            artifact.add_dir(str(path))
        else:
            artifact.add_file(str(path))
        self.run.log_artifact(artifact)

    def finish(self) -> None:
        """Finaliza a execução atual no wandb."""
        if not self.enabled or self.run is None:
            return
        self.run.finish()
        self.run = None

    def __enter__(self) -> "WandbLogger":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.finish()
