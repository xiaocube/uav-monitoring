"""
Model export utilities for SkyGuard.

Supports:
  * PyTorch (.pt)          -- native format, already produced by training
  * ONNX (.onnx)           -- cross-platform, good for CPU inference
  * TensorRT (.engine)     -- NVIDIA GPU optimized (Sprint 5)
  * TorchScript (.torchscript) -- mobile / edge deployment

Usage:
    exporter = ModelExporter({"formats": ["onnx"], "imgsz": 640, "half": True})
    paths = exporter.export(Path("models/trained/skyguard-v1/weights/best.pt"))
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from skyguard.core.exceptions import ModelError
from skyguard.core.logger import get_logger
from skyguard.utils.device import resolve_torch_device

log = get_logger(__name__)


SUPPORTED_FORMATS = {"onnx", "engine", "torchscript", "openvino", "coreml"}


class ModelExporter:
    """Export a trained YOLO model to production formats."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.formats: List[str] = list(config.get("formats", ["onnx"]))
        self.imgsz: int = int(config.get("imgsz", 640))
        self.half: bool = bool(config.get("half", True))
        self.int8: bool = bool(config.get("int8", False))
        self.simplify: bool = bool(config.get("simplify", True))
        self.opset: int = int(config.get("opset", 12))
        self.workspace: int = int(config.get("workspace", 4))

    def export(self, model_path: Path) -> List[Path]:
        """Export the model to all configured formats.

        Returns a list of exported file paths.
        """
        from ultralytics import YOLO

        if not model_path.exists():
            raise ModelError(f"Model file not found: {model_path}")

        log.info(
            "Exporting {} to formats: {}",
            model_path.name,
            ", ".join(self.formats),
        )

        model = YOLO(str(model_path))
        exported: List[Path] = []

        for fmt in self.formats:
            fmt = fmt.lower()
            if fmt == "pt":
                exported.append(model_path)  # Already in PT format
                continue
            if fmt not in SUPPORTED_FORMATS:
                log.warning("Unsupported export format: {}. Skipping.", fmt)
                continue

            try:
                out = model.export(
                    format=fmt,
                    imgsz=self.imgsz,
                    half=self.half,
                    int8=self.int8,
                    simplify=self.simplify,
                    opset=self.opset,
                    workspace=self.workspace,
                    device=resolve_torch_device("auto"),
                )
                if out:
                    out_path = Path(out)
                    if out_path.exists():
                        exported.append(out_path)
                        log.info("  Exported {} -> {}", fmt, out_path)
                    else:
                        log.warning("Export reported success but file not found: {}", out)
            except Exception as e:
                log.error("Failed to export {}: {}", fmt, e)
                if fmt == "engine":
                    log.info(
                        "TensorRT export requires NVIDIA GPU + TensorRT. "
                        "Run on Jetson or CUDA machine, or skip engine format."
                    )

        return exported
