"""
Offline data augmentation for SkyGuard dataset.

We use albumentations for geometric / color transforms because it is
image-annotation aware (bbox transforms stay synchronized).

Policy design:
  * Keep augmentation conservative for drone/bird detection.
  * Simulate real-world deployment variations: lighting, weather, haze, motion blur.
  * Do NOT augment the original images in-place; write to processed/aug/.
  * Maintain YOLO annotation alignment.

Required package (already in requirements-dev.txt; runtime optional):
  pip install albumentations
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from skyguard.core.logger import get_logger
from skyguard.data.formats import Annotation, BoundingBox

log = get_logger(__name__)


class AugmentationPolicy:
    """Default augmentation policy for low-altitude targets."""

    def __init__(self) -> None:
        try:
            import albumentations as A
        except ImportError as e:
            raise ImportError(
                "albumentations is required for augmentation. "
                "Install with: pip install albumentations"
            ) from e

        self.transform = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=0.3,
                ),
                A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),
                A.MotionBlur(blur_limit=3, p=0.2),
                A.RandomFog(fog_coef_lower=0.05, fog_coef_upper=0.15, p=0.1),
                A.RandomRain(
                    slant_lower=-5,
                    slant_upper=5,
                    drop_length=10,
                    drop_width=1,
                    p=0.1,
                ),
            ],
            bbox_params=A.BboxParams(
                format="yolo",
                label_fields=["class_ids"],
                min_visibility=0.3,
            ),
        )

    def apply(
        self,
        image: np.ndarray,
        annotations: List[Annotation],
    ) -> Tuple[np.ndarray, List[Annotation]]:
        """Apply augmentation and return new image + aligned annotations."""
        bboxes = []
        class_ids = []
        for ann in annotations:
            # Convert internal top-left normalized format to albumentations yolo center format.
            xc = ann.bbox.x + ann.bbox.w / 2
            yc = ann.bbox.y + ann.bbox.h / 2
            bboxes.append([xc, yc, ann.bbox.w, ann.bbox.h])
            class_ids.append(ann.class_id)

        transformed = self.transform(image=image, bboxes=bboxes, class_ids=class_ids)

        new_annotations: List[Annotation] = []
        for bbox, cls_id in zip(transformed["bboxes"], transformed["class_ids"]):
            xc, yc, w, h = bbox
            new_annotations.append(
                Annotation(
                    class_id=int(cls_id),
                    bbox=BoundingBox(
                        x=xc - w / 2,
                        y=yc - h / 2,
                        w=w,
                        h=h,
                        normalized=True,
                    ),
                )
            )
        return transformed["image"], new_annotations


def augment_image_file(
    image_path: Path,
    label_path: Path,
    output_image_path: Path,
    output_label_path: Path,
    policy: AugmentationPolicy | None = None,
) -> bool:
    """Offline augment a single image + YOLO labels, write results to disk."""
    policy = policy or AugmentationPolicy()

    img = cv2.imread(str(image_path))
    if img is None:
        log.warning("Cannot read image: {}", image_path)
        return False

    annotations = _read_yolo_label_file(label_path)
    new_img, new_anns = policy.apply(img, annotations)

    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    output_label_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_image_path), new_img)

    ds_out = YOLODataset(root=output_label_path.parent.parent.parent)
    ds_out.write_labels(image_stem=output_image_path.stem, split=output_label_path.parent.name, annotations=new_anns)
    return True


def _read_yolo_label_file(label_path: Path) -> List[Annotation]:
    annotations: List[Annotation] = []
    if not label_path.exists():
        return annotations
    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            cls_id = int(parts[0])
            xc, yc, w, h = map(float, parts[1:])
            annotations.append(
                Annotation(
                    class_id=cls_id,
                    bbox=BoundingBox(
                        x=xc - w / 2,
                        y=yc - h / 2,
                        w=w,
                        h=h,
                        normalized=True,
                    ),
                )
            )
    return annotations
