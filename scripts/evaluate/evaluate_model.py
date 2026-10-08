"""
Evaluate a trained SkyGuard YOLO model.

Usage:
  python scripts/evaluate/evaluate_model.py \
      --model models/trained/skyguard-v1/weights/best.pt \
      --data data/skyguard-v1/processed/data.yaml \
      --split val \
      --output eval_report.json

  # Evaluate with custom confidence threshold
  python scripts/evaluate/evaluate_model.py \
      --model models/trained/skyguard-v1/weights/best.pt \
      --conf 0.5 \
      --iou 0.65
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for p in [str(SRC), str(ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from skyguard.core.logger import setup_logging
from skyguard.training.evaluate import ModelEvaluator


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate SkyGuard YOLO model")
    parser.add_argument("--model", type=Path, required=True, help="Path to trained model (.pt)")
    parser.add_argument("--data", type=str, required=True, help="Path to YOLO data.yaml")
    parser.add_argument("--split", type=str, default="val", help="Dataset split: train | val | test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output", type=Path, default=None, help="JSON output path")
    args = parser.parse_args()

    setup_logging()

    evaluator = ModelEvaluator(
        model_path=args.model,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
    )

    report = evaluator.evaluate(data_yaml=args.data, split=args.split)
    report.print_summary()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            f.write(report.to_json())
        print(f"Report saved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
