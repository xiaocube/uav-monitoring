"""
SkyGuard YOLO training script.

Usage:
  # Train with default config
  python scripts/train/train_yolo.py

  # Train with custom config
  python scripts/train/train_yolo.py --config config/training/default.yaml

  # Override specific parameters
  python scripts/train/train_yolo.py \
      --config config/training/default.yaml \
      --epochs 50 \
      --batch 32 \
      --model yolov8s.pt

  # Train + export
  python scripts/train/train_yolo.py --export

  # Resume from checkpoint
  python scripts/train/train_yolo.py --resume models/trained/skyguard-v1/weights/last.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for p in [str(SRC), str(ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from skyguard.core.logger import setup_logging
from skyguard.training.trainer import TrainingConfig, YOLOTrainer


def _override_config(config: TrainingConfig, args: argparse.Namespace) -> TrainingConfig:
    """Apply CLI overrides to the config object."""
    if args.model:
        config.base_model = args.model
    if args.data:
        config.data = args.data
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch is not None:
        config.batch = args.batch
    if args.imgsz is not None:
        config.imgsz = args.imgsz
    if args.device:
        config.device = args.device
    if args.project:
        config.project = args.project
    if args.name:
        config.name = args.name
    if args.seed is not None:
        config.seed = args.seed
    if args.lr is not None:
        config.lr0 = args.lr
    if args.patience is not None:
        config.patience = args.patience
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description="Train SkyGuard YOLO model")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "training" / "default.yaml")
    parser.add_argument("--model", type=str, default=None, help="Base model (e.g., yolov8n.pt)")
    parser.add_argument("--data", type=str, default=None, help="Path to YOLO data.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, help="cpu | cuda | mps | auto")
    parser.add_argument("--project", type=str, default=None, help="Output directory")
    parser.add_argument("--name", type=str, default=None, help="Experiment name")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None, help="Initial learning rate")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience")
    parser.add_argument("--export", action="store_true", help="Export model after training")
    parser.add_argument("--resume", type=Path, default=None, help="Resume from checkpoint")
    parser.add_argument("--no-tracking", action="store_true", help="Disable experiment tracking")
    args = parser.parse_args()

    setup_logging()

    # Load config
    config = TrainingConfig.from_yaml(args.config)
    config = _override_config(config, args)

    if args.no_tracking:
        config.tracking["backend"] = "none"

    trainer = YOLOTrainer(config)

    # Resume or train from scratch
    if args.resume:
        print(f"Resuming from: {args.resume}")
        # Ultralytics handles resume internally via the model path
        # We just need to pass the checkpoint as the base model
        config.base_model = str(args.resume)
        # Set exist_ok to avoid creating a new run directory
        config.exist_ok = True

    # Train
    results = trainer.train()

    # Export
    if args.export:
        trainer.export()

    print("\nTraining complete!")
    best = trainer.best_model_path
    if best:
        print(f"Best model: {best}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
