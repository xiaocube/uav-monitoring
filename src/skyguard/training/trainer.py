"""
SkyGuard YOLO training pipeline.

Wraps Ultralytics YOLO.train() with:
  * Config-driven hyperparameters
  * Experiment tracking (MLflow / W&B)
  * Automatic model export after training
  * Reproducible seed management

Usage:
    from skyguard.training import YOLOTrainer
    trainer = YOLOTrainer.from_config("config/training/default.yaml")
    results = trainer.train()
    trainer.export()  # ONNX + TensorRT
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from skyguard.core.exceptions import ConfigurationError, ModelError
from skyguard.core.logger import get_logger
from skyguard.utils.device import resolve_torch_device
from skyguard.training.export import ModelExporter
from skyguard.training.tracker import ExperimentTracker

log = get_logger(__name__)


@dataclass
class TrainingConfig:
    """Typed training hyperparameters."""

    enabled: bool = True
    base_model: str = "yolov8n.pt"
    data: str = "data/skyguard-v1/processed/data.yaml"
    project: str = "models/trained"
    name: str = "skyguard-v1"
    imgsz: int = 640
    epochs: int = 100
    batch: int = 16
    optimizer: str = "AdamW"
    lr0: float = 0.001
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    patience: int = 20
    save: bool = True
    save_period: int = 10
    device: str = "auto"
    workers: int = 4
    exist_ok: bool = False
    pretrained: bool = True
    seed: int = 42
    deterministic: bool = True
    single_cls: bool = False
    rect: bool = False
    cos_lr: bool = False
    close_mosaic: int = 10
    amp: bool = True
    fraction: float = 1.0
    dropout: float = 0.0
    val: bool = True
    iou: float = 0.6
    max_det: int = 300
    plots: bool = True
    # Augmentation
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 5.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 2.0
    perspective: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5
    mosaic: float = 1.0
    mixup: float = 0.1
    copy_paste: float = 0.0
    # Tracking
    tracking: Dict[str, Any] = field(default_factory=dict)
    # Export
    export: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "TrainingConfig":
        p = Path(path)
        if not p.exists():
            raise ConfigurationError(f"Training config not found: {p}")
        with p.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data.get("training", {}))

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainingConfig":
        # Only keep keys that match dataclass fields
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)

    def to_ultralytics_kwargs(self) -> Dict[str, Any]:
        """Return a dict suitable for YOLO().train(**kwargs)."""
        return {
            "data": self.data,
            "epochs": self.epochs,
            "imgsz": self.imgsz,
            "batch": self.batch,
            "optimizer": self.optimizer,
            "lr0": self.lr0,
            "lrf": self.lrf,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "patience": self.patience,
            "save": self.save,
            "save_period": self.save_period,
            "device": resolve_torch_device(self.device),
            "workers": self.workers,
            "exist_ok": self.exist_ok,
            "pretrained": self.pretrained,
            "seed": self.seed,
            "deterministic": self.deterministic,
            "single_cls": self.single_cls,
            "rect": self.rect,
            "cos_lr": self.cos_lr,
            "close_mosaic": self.close_mosaic,
            "amp": self.amp,
            "fraction": self.fraction,
            "dropout": self.dropout,
            "val": self.val,
            "iou": self.iou,
            "max_det": self.max_det,
            "plots": self.plots,
            "hsv_h": self.hsv_h,
            "hsv_s": self.hsv_s,
            "hsv_v": self.hsv_v,
            "degrees": self.degrees,
            "translate": self.translate,
            "scale": self.scale,
            "shear": self.shear,
            "perspective": self.perspective,
            "flipud": self.flipud,
            "fliplr": self.fliplr,
            "mosaic": self.mosaic,
            "mixup": self.mixup,
            "copy_paste": self.copy_paste,
            "project": self.project,
            "name": self.name,
        }


class YOLOTrainer:
    """High-level trainer for SkyGuard YOLO models."""

    def __init__(self, config: TrainingConfig) -> None:
        self.cfg = config
        self.tracker = ExperimentTracker(config.tracking)
        self.exporter = ModelExporter(config.export)
        self._results: Any = None

    @classmethod
    def from_yaml(cls, path: Path | str) -> "YOLOTrainer":
        return cls(TrainingConfig.from_yaml(path))

    @classmethod
    def from_config_dict(cls, d: Dict[str, Any]) -> "YOLOTrainer":
        return cls(TrainingConfig.from_dict(d))

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self) -> Any:
        """Run the full training loop.

        Returns the Ultralytics results object (contains metrics, plots, etc.).
        """
        from ultralytics import YOLO

        log.info(
            "Starting training: model={} data={} epochs={} imgsz={}",
            self.cfg.base_model,
            self.cfg.data,
            self.cfg.epochs,
            self.cfg.imgsz,
        )

        # Ensure dataset exists
        data_path = Path(self.cfg.data)
        if not data_path.exists():
            raise ConfigurationError(
                f"Dataset config not found: {data_path}. "
                "Run `python scripts/dataset/build_dataset.py` first."
            )

        # Seed for reproducibility
        if self.cfg.seed >= 0:
            import torch
            import numpy as np
            import random
            random.seed(self.cfg.seed)
            np.random.seed(self.cfg.seed)
            torch.manual_seed(self.cfg.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.cfg.seed)
            log.info("Random seed set to {}", self.cfg.seed)

        # Initialize tracker before training
        self.tracker.start_run(
            run_name=self.cfg.tracking.get("run_name", ""),
            params=self.cfg.to_ultralytics_kwargs(),
        )

        try:
            model = YOLO(self.cfg.base_model)
            kwargs = self.cfg.to_ultralytics_kwargs()
            self._results = model.train(**kwargs)
        except Exception as e:
            self.tracker.end_run(status="FAILED")
            raise ModelError(f"Training failed: {e}") from e

        # Log final metrics
        if self._results is not None:
            metrics = self._get_best_metrics()
            self.tracker.log_metrics(metrics)
            log.info("Training complete. Best mAP50: {:.4f}", metrics.get("metrics/mAP50(B)", 0))

        self.tracker.end_run(status="FINISHED")
        return self._results

    def _get_best_metrics(self) -> Dict[str, float]:
        """Extract best metrics from training results."""
        if self._results is None:
            return {}
        # Ultralytics results.results_dict contains the final metrics
        return getattr(self._results, "results_dict", {}) or {}

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export(self, model_path: Optional[Path] = None) -> List[Path]:
        """Export the trained model to configured formats.

        If model_path is not provided, looks for best.pt in the training output.
        """
        if model_path is None:
            model_path = self._find_best_model()

        if model_path is None or not model_path.exists():
            raise ModelError(
                "No trained model found. Train first or provide a model path."
            )

        log.info("Exporting model: {}", model_path)
        exported = self.exporter.export(model_path)
        log.info("Exported {} artifact(s)", len(exported))
        for p in exported:
            log.info("  -> {}", p)
        return exported

    def _find_best_model(self) -> Optional[Path]:
        """Locate best.pt from the most recent training run."""
        run_dir = Path(self.cfg.project) / self.cfg.name
        if not run_dir.exists():
            # Ultralytics adds a numeric suffix (e.g., skyguard-v12)
            candidates = sorted(
                [d for d in Path(self.cfg.project).glob(f"{self.cfg.name}*") if d.is_dir()],
                key=lambda p: p.stat().st_mtime,
            )
            if candidates:
                run_dir = candidates[-1]
        best = run_dir / "weights" / "best.pt"
        return best if best.exists() else None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def best_model_path(self) -> Optional[Path]:
        return self._find_best_model()

    def cleanup_checkpoints(self, keep_best: bool = True) -> None:
        """Remove intermediate checkpoints to save disk space."""
        run_dir = self.best_model_path
        if run_dir is None:
            return
        weights_dir = run_dir.parent
        for ckpt in weights_dir.glob("epoch_*.pt"):
            ckpt.unlink()
            log.info("Removed intermediate checkpoint: {}", ckpt)
        if not keep_best:
            best = weights_dir / "best.pt"
            if best.exists():
                best.unlink()
                log.info("Removed best checkpoint: {}", best)
