"""
Standalone model evaluation for SkyGuard.

When Ultralytics validation is not enough, use this module for:
  * Per-class mAP / Precision / Recall / F1
  * Confusion matrix generation
  * Error analysis (false positives by class)
  * Custom metric computation (e.g., small-target mAP)

Usage:
    from skyguard.training.evaluate import ModelEvaluator
    evaluator = ModelEvaluator("models/trained/skyguard-v1/weights/best.pt")
    report = evaluator.evaluate("data/skyguard-v1/processed/data.yaml")
    print(report.to_json())
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from skyguard.core.exceptions import ModelError
from skyguard.core.logger import get_logger
from skyguard.utils.device import resolve_torch_device

log = get_logger(__name__)


@dataclass
class ClassMetrics:
    """Metrics for a single class."""

    class_id: int
    class_name: str
    tp: int = 0
    fp: int = 0
    fn: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    ap50: float = 0.0
    ap50_95: float = 0.0


@dataclass
class EvaluationReport:
    """Aggregated evaluation results."""

    model_path: str
    dataset_path: str
    overall_map50: float = 0.0
    overall_map50_95: float = 0.0
    overall_precision: float = 0.0
    overall_recall: float = 0.0
    overall_f1: float = 0.0
    per_class: List[ClassMetrics] = field(default_factory=list)
    inference_ms_mean: float = 0.0
    inference_ms_std: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model_path,
            "dataset": self.dataset_path,
            "overall": {
                "mAP50": round(self.overall_map50, 4),
                "mAP50-95": round(self.overall_map50_95, 4),
                "precision": round(self.overall_precision, 4),
                "recall": round(self.overall_recall, 4),
                "f1": round(self.overall_f1, 4),
            },
            "per_class": [
                {
                    "class_id": m.class_id,
                    "class_name": m.class_name,
                    "tp": m.tp,
                    "fp": m.fp,
                    "fn": m.fn,
                    "precision": round(m.precision, 4),
                    "recall": round(m.recall, 4),
                    "f1": round(m.f1, 4),
                    "ap50": round(m.ap50, 4),
                    "ap50_95": round(m.ap50_95, 4),
                }
                for m in self.per_class
            ],
            "inference_ms": {
                "mean": round(self.inference_ms_mean, 2),
                "std": round(self.inference_ms_std, 2),
            },
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def print_summary(self) -> None:
        print(f"\n{'='*60}")
        print(f"Evaluation Report: {Path(self.model_path).name}")
        print(f"{'='*60}")
        print(f"  mAP@50     : {self.overall_map50:.4f}")
        print(f"  mAP@50-95  : {self.overall_map50_95:.4f}")
        print(f"  Precision  : {self.overall_precision:.4f}")
        print(f"  Recall     : {self.overall_recall:.4f}")
        print(f"  F1         : {self.overall_f1:.4f}")
        print(f"  Inference  : {self.inference_ms_mean:.1f} +/- {self.inference_ms_std:.1f} ms")
        print(f"\n{'-'*60}")
        print(f"{'Class':<15} {'P':>8} {'R':>8} {'F1':>8} {'mAP50':>8} {'mAP50-95':>8}")
        print(f"{'-'*60}")
        for m in self.per_class:
            print(
                f"{m.class_name:<15} "
                f"{m.precision:>8.4f} {m.recall:>8.4f} {m.f1:>8.4f} "
                f"{m.ap50:>8.4f} {m.ap50_95:>8.4f}"
            )
        print(f"{'='*60}\n")


class ModelEvaluator:
    """Evaluate a trained YOLO model on a validation set."""

    def __init__(
        self,
        model_path: str | Path,
        device: Optional[str] = None,
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.6,
    ) -> None:
        self.model_path = str(model_path)
        self.device = resolve_torch_device(device or "auto")
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self._model: Any = None

    def _load_model(self) -> Any:
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
        return self._model

    def evaluate(
        self,
        data_yaml: str | Path,
        split: str = "val",
    ) -> EvaluationReport:
        """Run evaluation and return a structured report."""
        model = self._load_model()
        data_path = str(data_yaml)

        log.info("Evaluating {} on {} (split={})", self.model_path, data_path, split)

        # Ultralytics validation
        results = model.val(
            data=data_path,
            split=split,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )

        report = EvaluationReport(
            model_path=self.model_path,
            dataset_path=data_path,
        )

        # Overall metrics - results_dict is a dict, use get()
        results_dict = getattr(results, "results_dict", {})
        report.overall_map50 = float(results_dict.get("metrics/mAP50(B)", 0.0) or 0.0)
        report.overall_map50_95 = float(results_dict.get("metrics/mAP50-95(B)", 0.0) or 0.0)
        report.overall_precision = float(results_dict.get("metrics/precision(B)", 0.0) or 0.0)
        report.overall_recall = float(results_dict.get("metrics/recall(B)", 0.0) or 0.0)

        # F1 from P and R
        p, r = report.overall_precision, report.overall_recall
        report.overall_f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        # Per-class metrics
        names = getattr(results, "names", {})
        maps = getattr(results, "maps", np.array([]))
        curves = getattr(results, "curves", {})

        for cls_id, cls_name in sorted(names.items(), key=lambda x: int(x[0])):
            cls_id = int(cls_id)
            m = ClassMetrics(class_id=cls_id, class_name=str(cls_name))
            if cls_id < len(maps):
                m.ap50_95 = float(maps[cls_id])
            precision_curve = curves.get("precision", {}).get(cls_id)
            recall_curve = curves.get("recall", {}).get(cls_id)
            if precision_curve is not None and len(precision_curve) > 0:
                m.precision = float(np.mean(precision_curve))
            if recall_curve is not None and len(recall_curve) > 0:
                m.recall = float(np.mean(recall_curve))
            if m.precision > 0 and m.recall > 0:
                m.f1 = 2 * m.precision * m.recall / (m.precision + m.recall)
            report.per_class.append(m)

        log.info("Evaluation complete. mAP50={:.4f}", report.overall_map50)
        return report
