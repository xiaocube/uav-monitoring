"""Tests for SkyGuard training pipeline."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from skyguard.core.exceptions import ConfigurationError
from skyguard.training.tracker import ExperimentTracker
from skyguard.training.trainer import TrainingConfig, YOLOTrainer
from skyguard.training.export import ModelExporter, SUPPORTED_FORMATS


def test_training_config_from_dict():
    cfg = TrainingConfig.from_dict({
        "base_model": "yolov8s.pt",
        "epochs": 50,
        "batch": 8,
        "lr0": 0.01,
        "tracking": {"backend": "mlflow"},
    })
    assert cfg.base_model == "yolov8s.pt"
    assert cfg.epochs == 50
    assert cfg.batch == 8
    assert cfg.lr0 == 0.01
    assert cfg.tracking["backend"] == "mlflow"


def test_training_config_default_values():
    cfg = TrainingConfig()
    assert cfg.base_model == "yolov8n.pt"
    assert cfg.epochs == 100
    assert cfg.imgsz == 640
    assert cfg.optimizer == "AdamW"


def test_training_config_to_ultralytics_kwargs():
    cfg = TrainingConfig(epochs=10, batch=4)
    kwargs = cfg.to_ultralytics_kwargs()
    assert kwargs["epochs"] == 10
    assert kwargs["batch"] == 4
    assert kwargs["data"] == cfg.data
    assert "project" in kwargs
    assert "name" in kwargs


def test_training_config_from_yaml(tmp_path):
    config = {"training": {"epochs": 20, "batch": 2, "base_model": "yolov8n.pt"}}
    path = tmp_path / "train.yaml"
    with path.open("w") as f:
        yaml.dump(config, f)

    cfg = TrainingConfig.from_yaml(path)
    assert cfg.epochs == 20
    assert cfg.batch == 2


def test_training_config_missing_yaml_raises():
    with pytest.raises(ConfigurationError):
        TrainingConfig.from_yaml("/does/not/exist.yaml")


def test_experiment_tracker_none_backend():
    tracker = ExperimentTracker({"backend": "none"})
    assert tracker.backend == "none"
    # These should not raise
    tracker.start_run(run_name="test")
    tracker.log_metrics({"acc": 0.9})
    tracker.end_run()


def test_model_exporter_supported_formats():
    assert "onnx" in SUPPORTED_FORMATS
    assert "engine" in SUPPORTED_FORMATS


def test_model_exporter_config_defaults():
    exporter = ModelExporter({})
    assert exporter.imgsz == 640
    assert exporter.half is True
    assert exporter.simplify is True


def test_model_exporter_with_formats():
    exporter = ModelExporter({"formats": ["onnx", "torchscript"], "imgsz": 320, "half": False})
    assert exporter.formats == ["onnx", "torchscript"]
    assert exporter.imgsz == 320
    assert exporter.half is False


def test_yolo_trainer_from_yaml(tmp_path):
    config = {"training": {"epochs": 5, "batch": 2, "base_model": "yolov8n.pt"}}
    path = tmp_path / "train.yaml"
    with path.open("w") as f:
        yaml.dump(config, f)

    trainer = YOLOTrainer.from_yaml(path)
    assert trainer.cfg.epochs == 5
    assert trainer.cfg.batch == 2


def test_yolo_trainer_from_config_dict():
    trainer = YOLOTrainer.from_config_dict({"epochs": 3, "name": "test-run"})
    assert trainer.cfg.epochs == 3
    assert trainer.cfg.name == "test-run"
