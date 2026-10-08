"""
YOLOv8-based object detector abstraction.

Design goals:
  * Wrap Ultralytics YOLO so the rest of the system is NOT tied to Ultralytics API.
  * Output a stable, serializable Detection data model (Pydantic v2) regardless
    of whether the underlying model is YOLOv8 / YOLO11 / ONNX / TensorRT.
  * One-time device resolution per process (no repeated cuda/mps queries).
  * Optional class filtering and confidence / NMS thresholding.
  * Image and batch inference paths.

TODO in later sprints:
  * Replace `COCO_TO_SKYGUARD` mapping with native class output once we train
    our own model in Sprint 4.
  * Add TensorRT / ONNX runtime backends in Sprint 5.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field
from ultralytics import YOLO

from skyguard.core.config import get_settings
from skyguard.core.exceptions import ModelError
from skyguard.core.logger import get_logger
from skyguard.utils.device import resolve_torch_device
from skyguard.vision.classes import COCO_TO_SKYGUARD, SkyGuardClass

log = get_logger(__name__)


class Detection(BaseModel):
    """A single detection, normalized for downstream use (tracking / DB / web)."""

    # Bounding box in pixel coordinates: [x1, y1, x2, y2].
    xyxy: Tuple[int, int, int, int] = Field(..., description="Bounding box [x1, y1, x2, y2] in pixels")

    # Confidence score.
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")

    # Class ID returned by the model. For pretrained YOLOv8 this is COCO ID.
    # For our own model this will be SkyGuardClass value.
    class_id: int = Field(..., description="Model class ID")

    # Human-readable class name.
    class_name: str = Field(..., description="Class name")

    # SkyGuard taxonomy class (may be None if COCO class has no mapping).
    skyguard_class: Optional[SkyGuardClass] = Field(None, description="SkyGuard taxonomy class")

    # Center point (useful for PTZ / map projection later).
    cx: int = Field(..., description="Center x")
    cy: int = Field(..., description="Center y")

    class Config:
        frozen = True


class DetectionResult(BaseModel):
    """Collection of detections plus frame-level metadata."""

    detections: List[Detection] = Field(default_factory=list)
    frame_id: int = 0
    timestamp_ms: Optional[float] = None
    model_name: str = ""
    inference_ms: Optional[float] = None
    img_width: int = 0
    img_height: int = 0


class YOLODetector:
    """Production detector wrapper around a YOLO-family model."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        imgsz: Optional[int] = None,
        classes: Optional[List[int]] = None,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the detector.

        Parameters
        ----------
        model_path : str | Path | None
            Path to .pt / .engine / .onnx. If None, read from config.
        device : str | None
            "cpu", "cuda", "mps", or "auto". If None, read from config.
        conf, iou, imgsz : optional
            Override config defaults.
        classes : list[int] | None
            Restrict to these class IDs. For COCO-pretrained model this means
            COCO IDs (e.g. [5, 14, 38] for airplane/bird/kite). None = all.
        verbose : bool
            Enable Ultralytics verbose output (default False for production).
        """
        cfg = get_settings().detection
        self.model_path = str(model_path or cfg.model_name)
        self.device = str(device or get_settings().compute.device)
        self.conf = float(conf if conf is not None else cfg.conf)
        self.iou = float(iou if iou is not None else cfg.iou)
        self.imgsz = int(imgsz if imgsz is not None else cfg.imgsz)
        self.classes = classes if classes is not None else cfg.classes
        self.verbose = verbose

        self._torch_device = resolve_torch_device(self.device)
        self._model: Optional[YOLO] = None

        log.info(
            "YOLODetector configured: model={} device={} imgsz={} conf={} iou={} classes={}",
            self.model_path,
            self._torch_device,
            self.imgsz,
            self.conf,
            self.iou,
            self.classes,
        )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> YOLO:
        """Lazy-load the model. Ultralytics automatically downloads pretrained .pt."""
        if self._model is None:
            try:
                self._model = YOLO(self.model_path)
            except Exception as e:
                raise ModelError(f"Failed to load YOLO model from {self.model_path}: {e}") from e
            log.info("Model loaded: {}", self.model_path)
        return self._model

    # ------------------------------------------------------------------
    # Public inference API
    # ------------------------------------------------------------------

    def predict(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        classes: Optional[List[int]] = None,
    ) -> DetectionResult:
        """
        Run detection on a single image.

        Returns a DetectionResult ready for annotation, tracking, or HTTP API.
        """
        model = self._load_model()
        _conf = conf if conf is not None else self.conf
        _iou = iou if iou is not None else self.iou
        _classes = classes if classes is not None else self.classes

        try:
            results = model.predict(
                source=image,
                device=self._torch_device,
                conf=_conf,
                iou=_iou,
                imgsz=self.imgsz,
                classes=_classes if _classes else None,
                verbose=self.verbose,
            )
        except Exception as e:
            raise ModelError(f"Inference failed: {e}") from e

        if not results:
            return DetectionResult(model_name=self.model_path)

        r = results[0]
        img_h, img_w = r.orig_shape[:2]
        inference_ms = getattr(r, "speed", {}).get("inference", None)

        detections: List[Detection] = []
        boxes = r.boxes
        if boxes is not None and len(boxes) > 0:
            xyxyn = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(int)

            for xyxy, score, cls_id in zip(xyxyn, confs, cls_ids):
                x1, y1, x2, y2 = map(int, xyxy)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                sg_class = COCO_TO_SKYGUARD.get(cls_id)

                detections.append(
                    Detection(
                        xyxy=(x1, y1, x2, y2),
                        confidence=float(score),
                        class_id=int(cls_id),
                        class_name=self._class_name(cls_id),
                        skyguard_class=sg_class,
                        cx=cx,
                        cy=cy,
                    )
                )

        return DetectionResult(
            detections=detections,
            model_name=self.model_path,
            inference_ms=inference_ms,
            img_width=img_w,
            img_height=img_h,
        )

    def predict_batch(
        self,
        images: List[Union[str, Path, np.ndarray, Image.Image]],
    ) -> List[DetectionResult]:
        """Run detection on a batch of images."""
        model = self._load_model()
        try:
            results = model(
                images,
                device=self._torch_device,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.imgsz,
                classes=self.classes if self.classes else None,
                verbose=self.verbose,
            )
        except Exception as e:
            raise ModelError(f"Batch inference failed: {e}") from e

        out: List[DetectionResult] = []
        for r in results:
            # Reuse single-image conversion by feeding a tiny wrapper.
            img_h, img_w = r.orig_shape[:2]
            inference_ms = getattr(r, "speed", {}).get("inference", None)
            detections: List[Detection] = []
            boxes = r.boxes
            if boxes is not None and len(boxes) > 0:
                for xyxy, score, cls_id in zip(
                    boxes.xyxy.cpu().numpy(),
                    boxes.conf.cpu().numpy(),
                    boxes.cls.cpu().numpy().astype(int),
                ):
                    x1, y1, x2, y2 = map(int, xyxy)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    detections.append(
                        Detection(
                            xyxy=(x1, y1, x2, y2),
                            confidence=float(score),
                            class_id=int(cls_id),
                            class_name=self._class_name(int(cls_id)),
                            skyguard_class=COCO_TO_SKYGUARD.get(int(cls_id)),
                            cx=cx,
                            cy=cy,
                        )
                    )
            out.append(
                DetectionResult(
                    detections=detections,
                    model_name=self.model_path,
                    inference_ms=inference_ms,
                    img_width=img_w,
                    img_height=img_h,
                )
            )
        return out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _class_name(cls_id: int) -> str:
        """Return a human-readable name for a COCO class ID (best effort)."""
        sg_class = COCO_TO_SKYGUARD.get(cls_id)
        if sg_class is not None:
            return sg_class.name.lower().replace("_", " ")
        # Fallback to a generic label.
        return f"coco_{cls_id}"

    def get_class_filter(self) -> List[int]:
        """Return the current class filter as a list of model class IDs."""
        return list(self.classes) if self.classes else []
