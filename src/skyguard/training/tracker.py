"""
Experiment tracking for SkyGuard training.

Backends supported:
  * mlflow     (default, local-first)
  * wandb      (cloud, requires API key)
  * tensorboard
  * none       (disable tracking)

MLflow is the default because it:
  * works fully offline (local filesystem backend)
  * has a built-in UI (`mlflow ui`)
  * is easy to self-host (no cloud dependency)
  * integrates natively with Ultralytics

Usage:
    tracker = ExperimentTracker({"backend": "mlflow", "experiment_name": "skyguard"})
    tracker.start_run(run_name="exp-001", params={"epochs": 100})
    tracker.log_metrics({"mAP50": 0.87})
    tracker.end_run()
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from skyguard.core.logger import get_logger

log = get_logger(__name__)


class ExperimentTracker:
    """Lightweight wrapper around MLflow / W&B / TensorBoard."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.backend = str(config.get("backend", "none")).lower()
        self.experiment_name = str(config.get("experiment_name", "skyguard"))
        self.run_name = str(config.get("run_name", ""))
        self.tags: Dict[str, str] = {k: str(v) for k, v in config.get("tags", {}).items()}
        self._run_id: Optional[str] = None
        self._client: Any = None

        if self.backend == "mlflow":
            self._init_mlflow()
        elif self.backend == "wandb":
            self._init_wandb()
        elif self.backend == "tensorboard":
            self._init_tensorboard()
        else:
            log.info("Experiment tracking disabled (backend='{}')", self.backend)

    # ------------------------------------------------------------------
    # Backend init
    # ------------------------------------------------------------------

    def _init_mlflow(self) -> None:
        try:
            import mlflow
            # Use local filesystem tracking by default
            tracking_dir = Path("./mlruns").resolve()
            tracking_dir.mkdir(parents=True, exist_ok=True)
            mlflow.set_tracking_uri(f"file://{tracking_dir}")
            mlflow.set_experiment(self.experiment_name)
            self._client = mlflow
            log.info(
                "MLflow tracking initialized: experiment='{}' uri='{}'",
                self.experiment_name,
                tracking_dir,
            )
        except ImportError:
            log.warning("mlflow not installed. Tracking disabled.")
            self.backend = "none"

    def _init_wandb(self) -> None:
        try:
            import wandb
            self._client = wandb
            log.info("W&B tracking initialized")
        except ImportError:
            log.warning("wandb not installed. Tracking disabled.")
            self.backend = "none"

    def _init_tensorboard(self) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
            log_dir = Path("./runs") / self.experiment_name / datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir.mkdir(parents=True, exist_ok=True)
            self._client = SummaryWriter(log_dir=str(log_dir))
            log.info("TensorBoard logging to {}", log_dir)
        except ImportError:
            log.warning("tensorboard not installed. Tracking disabled.")
            self.backend = "none"

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    def start_run(
        self,
        run_name: str = "",
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self.backend == "none" or self._client is None:
            return

        name = run_name or self.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")

        if self.backend == "mlflow":
            self._client.start_run(run_name=name)
            self._client.set_tags(self.tags)
            if params:
                # Filter non-serializable values
                clean = {k: v for k, v in params.items() if isinstance(v, (int, float, str, bool))}
                self._client.log_params(clean)
            self._run_id = self._client.active_run().info.run_id
            log.info("MLflow run started: {}", self._run_id)

        elif self.backend == "wandb":
            self._run_id = self._client.init(
                project=self.experiment_name,
                name=name,
                config=params,
                tags=list(self.tags.values()),
            ).id
            log.info("W&B run started: {}", self._run_id)

        elif self.backend == "tensorboard":
            if params:
                for k, v in params.items():
                    if isinstance(v, (int, float)):
                        self._client.add_scalar(f"hyperparams/{k}", v, 0)
            log.info("TensorBoard run started")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if self.backend == "none" or self._client is None:
            return

        # Filter NaN / None values
        clean = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}

        if self.backend == "mlflow":
            self._client.log_metrics(clean, step=step)

        elif self.backend == "wandb":
            self._client.log(clean, step=step)

        elif self.backend == "tensorboard":
            for k, v in clean.items():
                self._client.add_scalar(k, v, step or 0)

    def log_artifact(self, path: Path) -> None:
        if self.backend == "none" or self._client is None:
            return

        if not path.exists():
            log.warning("Artifact not found: {}", path)
            return

        if self.backend == "mlflow":
            self._client.log_artifact(str(path))

        elif self.backend == "wandb":
            artifact = self._client.Artifact("model", type="model")
            artifact.add_file(str(path))
            self._client.log_artifact(artifact)

        elif self.backend == "tensorboard":
            # TensorBoard doesn't support arbitrary artifacts
            pass

    def end_run(self, status: str = "FINISHED") -> None:
        if self.backend == "none" or self._client is None:
            return

        if self.backend == "mlflow":
            self._client.set_tag("status", status)
            self._client.end_run()
            log.info("MLflow run ended: status={}", status)

        elif self.backend == "wandb":
            self._client.finish()
            log.info("W&B run ended: status={}", status)

        elif self.backend == "tensorboard":
            self._client.close()
            log.info("TensorBoard writer closed")
