"""
Annotation format conversion between YOLO (txt) and COCO (JSON).

YOLO format (per image):
  <class_id> <x_center_norm> <y_center_norm> <w_norm> <h_norm>

COCO format (single JSON):
  {
    "images": [...],
    "annotations": [...],
    "categories": [...]
  }

Design:
  * Both formats use the same internal BoundingBox representation.
  * Conversion is lossless for axis-aligned boxes.
  * Category IDs are stable and map to SkyGuardClass.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from skyguard.core.exceptions import ConfigurationError
from skyguard.core.logger import get_logger
from skyguard.vision.classes import SKYGUARD_NAMES, SkyGuardClass

log = get_logger(__name__)


@dataclass
class BoundingBox:
    """Normalized or pixel bounding box."""

    x: float
    y: float
    w: float
    h: float
    image_width: int = 0
    image_height: int = 0
    normalized: bool = True

    def to_xyxy(self) -> Tuple[int, int, int, int]:
        """Return pixel xyxy coordinates."""
        if self.normalized:
            if self.image_width == 0 or self.image_height == 0:
                raise ValueError("Cannot convert normalized box to xyxy without image size")
            x1 = int(self.x * self.image_width)
            y1 = int(self.y * self.image_height)
            x2 = int((self.x + self.w) * self.image_width)
            y2 = int((self.y + self.h) * self.image_height)
        else:
            x1 = int(self.x)
            y1 = int(self.y)
            x2 = int(self.x + self.w)
            y2 = int(self.y + self.h)
        return x1, y1, x2, y2

    @classmethod
    def from_xyxy(
        cls,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        image_width: int,
        image_height: int,
        normalized: bool = False,
    ) -> "BoundingBox":
        return cls(
            x=x1,
            y=y1,
            w=x2 - x1,
            h=y2 - y1,
            image_width=image_width,
            image_height=image_height,
            normalized=normalized,
        )

    def to_yolo_line(self) -> str:
        """Return a single YOLO-format line (normalized)."""
        if not self.normalized:
            if self.image_width == 0 or self.image_height == 0:
                raise ValueError("Cannot normalize box without image size")
            xc = (self.x + self.w / 2) / self.image_width
            yc = (self.y + self.h / 2) / self.image_height
            wn = self.w / self.image_width
            hn = self.h / self.image_height
        else:
            xc = self.x + self.w / 2
            yc = self.y + self.h / 2
            wn = self.w
            hn = self.h
        return f"{xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}"


@dataclass
class Annotation:
    """A single object annotation."""

    class_id: int
    bbox: BoundingBox


class YOLODataset:
    """Reader/writer for YOLO-format dataset layout.

    Expected layout:
      root/
        images/
          train/
          val/
          test/
        labels/
          train/
          val/
          test/
        data.yaml
    """

    def __init__(self, root: Path, classes: Optional[List[str]] = None) -> None:
        self.root = Path(root)
        self.classes = classes or list(SKYGUARD_NAMES.values())

    def write_yaml(self, path: Optional[Path] = None) -> Path:
        path = path or self.root / "data.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        content = {
            "path": str(self.root.resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": len(self.classes),
            "names": {i: name for i, name in enumerate(self.classes)},
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(content, f, indent=2, ensure_ascii=False)
        log.info("YOLO data.yaml written: {}", path)
        return path

    def read_labels(self, image_stem: str, split: str) -> List[Annotation]:
        label_path = self.root / "labels" / split / f"{image_stem}.txt"
        if not label_path.exists():
            return []
        annotations: List[Annotation] = []
        with label_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    log.warning("Malformed YOLO label line in {}: {}", label_path, line)
                    continue
                cls_id = int(parts[0])
                xc, yc, w, h = map(float, parts[1:])
                # Convert YOLO center-format to top-left-format for internal use.
                bbox = BoundingBox(
                    x=xc - w / 2,
                    y=yc - h / 2,
                    w=w,
                    h=h,
                    normalized=True,
                )
                annotations.append(Annotation(class_id=cls_id, bbox=bbox))
        return annotations

    def write_labels(
        self,
        image_stem: str,
        split: str,
        annotations: List[Annotation],
    ) -> Path:
        label_dir = self.root / "labels" / split
        label_dir.mkdir(parents=True, exist_ok=True)
        label_path = label_dir / f"{image_stem}.txt"
        with label_path.open("w", encoding="utf-8") as f:
            for ann in annotations:
                f.write(f"{ann.class_id} {ann.bbox.to_yolo_line()}\n")
        return label_path


class COCODataset:
    """Reader/writer for COCO-format JSON."""

    def __init__(self, classes: Optional[List[str]] = None) -> None:
        self.classes = classes or list(SKYGUARD_NAMES.values())

    def to_coco(
        self,
        image_entries: List[Dict],
        annotations: List[Tuple[int, int, BoundingBox]],
    ) -> Dict:
        """Build a COCO JSON dict.

        image_entries: list of dicts with keys id, file_name, width, height
        annotations: list of (image_id, class_id, bbox) where bbox uses pixel coords
        """
        cats = [
            {"id": i, "name": name, "supercategory": "object"}
            for i, name in enumerate(self.classes)
        ]
        anns: List[Dict] = []
        for ann_id, (image_id, cls_id, bbox) in enumerate(annotations):
            x1, y1, x2, y2 = bbox.to_xyxy()
            anns.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": cls_id,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": (x2 - x1) * (y2 - y1),
                    "iscrowd": 0,
                }
            )
        return {
            "images": image_entries,
            "annotations": anns,
            "categories": cats,
        }

    def save(self, path: Path, image_entries: List[Dict], annotations: List[Tuple[int, int, BoundingBox]]) -> None:
        coco = self.to_coco(image_entries, annotations)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(coco, f, indent=2, ensure_ascii=False)
        log.info("COCO JSON saved: {}", path)

    def load(self, path: Path) -> Dict:
        if not path.exists():
            raise ConfigurationError(f"COCO file not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
