"""
Convert between YOLO (txt) and COCO (JSON) annotation formats.

Usage:
  # YOLO -> COCO
  python scripts/dataset/convert_format.py yolo2coco \
      --images data/skyguard-v1/processed/images/train \
      --labels data/skyguard-v1/processed/labels/train \
      --output data/skyguard-v1/annotations/instances_train.json

  # COCO -> YOLO
  python scripts/dataset/convert_format.py coco2yolo \
      --json data/skyguard-v1/annotations/instances_train.json \
      --output data/skyguard-v1/processed/labels/train
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from skyguard.core.logger import setup_logging
from skyguard.data.formats import (
    Annotation,
    BoundingBox,
    COCODataset,
    YOLODataset,
)


def _yolo2coco(args: argparse.Namespace) -> int:
    image_dir = Path(args.images)
    label_dir = Path(args.labels)
    output_path = Path(args.output)

    ds = YOLODataset(root=image_dir.parent.parent)
    coco = COCODataset(classes=ds.classes)

    image_entries = []
    annotations: list[tuple[int, int, BoundingBox]] = []

    for img_path in sorted(image_dir.glob("*")):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        image_id = len(image_entries)
        image_entries.append(
            {"id": image_id, "file_name": img_path.name, "width": w, "height": h}
        )

        split = image_dir.name  # train/val/test
        anns = ds.read_labels(img_path.stem, split)
        for ann in anns:
            pixel_bbox = BoundingBox(
                x=ann.bbox.x,
                y=ann.bbox.y,
                w=ann.bbox.w,
                h=ann.bbox.h,
                image_width=w,
                image_height=h,
                normalized=True,
            )
            annotations.append((image_id, ann.class_id, pixel_bbox))

    coco.save(output_path, image_entries, annotations)
    print(f"Saved COCO JSON: {output_path}")
    return 0


def _coco2yolo(args: argparse.Namespace) -> int:
    json_path = Path(args.json)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    coco = COCODataset()
    data = coco.load(json_path)

    images = {img["id"]: img for img in data["images"]}
    categories = {cat["id"]: cat["name"] for cat in data["categories"]}

    # Group annotations by image.
    grouped: dict[int, list] = {}
    for ann in data["annotations"]:
        grouped.setdefault(ann["image_id"], []).append(ann)

    for image_id, anns in grouped.items():
        img_info = images[image_id]
        w, h = img_info["width"], img_info["height"]
        annotations: list[Annotation] = []
        for ann in anns:
            x, y, bw, bh = ann["bbox"]
            annotations.append(
                Annotation(
                    class_id=ann["category_id"],
                    bbox=BoundingBox.from_xyxy(
                        int(x), int(y), int(x + bw), int(y + bh), w, h, normalized=False
                    ),
                )
            )
        label_path = output_dir / f"{Path(img_info['file_name']).stem}.txt"
        ds = YOLODataset(root=output_dir.parent.parent)
        ds.classes = list(categories.values())
        ds.write_labels(Path(img_info["file_name"]).stem, split=output_dir.name, annotations=annotations)

    print(f"Saved YOLO labels to: {output_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert annotation formats")
    sub = parser.add_subparsers(dest="command", required=True)

    y2c = sub.add_parser("yolo2coco")
    y2c.add_argument("--images", type=str, required=True)
    y2c.add_argument("--labels", type=str, required=True)
    y2c.add_argument("--output", type=str, required=True)

    c2y = sub.add_parser("coco2yolo")
    c2y.add_argument("--json", type=str, required=True)
    c2y.add_argument("--output", type=str, required=True)

    args = parser.parse_args()
    setup_logging()

    if args.command == "yolo2coco":
        return _yolo2coco(args)
    if args.command == "coco2yolo":
        return _coco2yolo(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
