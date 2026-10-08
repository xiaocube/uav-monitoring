"""
SkyGuard target taxonomy and the mapping to COCO classes used by pretrained YOLOv8.

For Sprint 2 we still rely on the COCO-pretrained YOLOv8 model. In Sprint 4 we will
switch to a custom-trained model trained on our own dataset with the exact
SkyGuard class IDs (drone, bird, airplane, helicopter, balloon, kite).

Design notes:
  * Each SkyGuardClass carries a stable internal ID and a COCO fallback ID.
  * This module is the SINGLE source of truth for class names / colors / IDs.
  * Future model retraining only requires changing the `model_id` and `name`
    fields; the rest of the pipeline uses these constants.
"""
from __future__ import annotations

from enum import IntEnum
from typing import Dict, List, Tuple


class SkyGuardClass(IntEnum):
    """Stable class IDs for the SkyGuard product taxonomy."""

    DRONE = 0
    BIRD = 1
    AIRPLANE = 2
    HELICOPTER = 3
    BALLOON = 4
    KITE = 5


# Friendly names (displayed in UI / reports).
SKYGUARD_NAMES: Dict[SkyGuardClass, str] = {
    SkyGuardClass.DRONE: "drone",
    SkyGuardClass.BIRD: "bird",
    SkyGuardClass.AIRPLANE: "airplane",
    SkyGuardClass.HELICOPTER: "helicopter",
    SkyGuardClass.BALLOON: "balloon",
    SkyGuardClass.KITE: "kite",
}

# Chinese labels for on-site demos and reports.
SKYGUARD_LABELS_CN: Dict[SkyGuardClass, str] = {
    SkyGuardClass.DRONE: "无人机",
    SkyGuardClass.BIRD: "鸟类",
    SkyGuardClass.AIRPLANE: "固定翼飞机",
    SkyGuardClass.HELICOPTER: "直升机",
    SkyGuardClass.BALLOON: "气球",
    SkyGuardClass.KITE: "风筝",
}

# Stable BGR color palette for annotation (OpenCV uses BGR).
SKYGUARD_COLORS: Dict[SkyGuardClass, Tuple[int, int, int]] = {
    SkyGuardClass.DRONE: (0, 140, 255),      # orange
    SkyGuardClass.BIRD: (0, 255, 255),       # yellow
    SkyGuardClass.AIRPLANE: (255, 0, 0),     # blue
    SkyGuardClass.HELICOPTER: (0, 255, 0),   # green
    SkyGuardClass.BALLOON: (255, 0, 255),    # magenta
    SkyGuardClass.KITE: (255, 255, 0),       # cyan
}

# COCO fallback IDs used by YOLOv8 pretrained on COCO 2017.
# airplane -> COCO 5, bird -> COCO 14, kite -> COCO 38.
# drone/helicopter/balloon do NOT exist in COCO, so they map to None.
SKYGUARD_COCO_IDS: Dict[SkyGuardClass, int | None] = {
    SkyGuardClass.DRONE: None,
    SkyGuardClass.BIRD: 14,
    SkyGuardClass.AIRPLANE: 5,
    SkyGuardClass.HELICOPTER: None,
    SkyGuardClass.BALLOON: None,
    SkyGuardClass.KITE: 38,
}

# Reverse lookup: COCO ID -> SkyGuardClass (only classes that exist in COCO).
COCO_TO_SKYGUARD: Dict[int, SkyGuardClass] = {
    coco_id: sg_class
    for sg_class, coco_id in SKYGUARD_COCO_IDS.items()
    if coco_id is not None
}

# Convenience list of all COCO IDs we are currently interested in.
SKYGUARD_COCO_NAMES: List[str] = [
    "airplane",
    "bird",
    "kite",
]


def get_skyguard_class_from_coco(coco_id: int) -> SkyGuardClass | None:
    """Map a COCO class ID to the corresponding SkyGuard class (if any)."""
    return COCO_TO_SKYGUARD.get(coco_id)


def get_name(sg_class: SkyGuardClass | int) -> str:
    """Return the English class name."""
    return SKYGUARD_NAMES[SkyGuardClass(sg_class)]


def get_label_cn(sg_class: SkyGuardClass | int) -> str:
    """Return the Chinese class label."""
    return SKYGUARD_LABELS_CN[SkyGuardClass(sg_class)]


def get_color(sg_class: SkyGuardClass | int) -> Tuple[int, int, int]:
    """Return the BGR color assigned to a class."""
    return SKYGUARD_COLORS[SkyGuardClass(sg_class)]
