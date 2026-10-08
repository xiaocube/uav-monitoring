"""SkyGuard vision pipeline: detection, classification, annotation, streaming."""

from skyguard.vision.classes import SKYGUARD_COCO_IDS, SKYGUARD_COCO_NAMES, SkyGuardClass
from skyguard.vision.detector import Detection, DetectionResult, YOLODetector

__all__ = [
    "Detection",
    "DetectionResult",
    "YOLODetector",
    "SkyGuardClass",
    "SKYGUARD_COCO_IDS",
    "SKYGUARD_COCO_NAMES",
]
