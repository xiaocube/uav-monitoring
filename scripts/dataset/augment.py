"""
Apply offline augmentation to a YOLO-format labeled dataset.

Usage:
  python scripts/dataset/augment.py \
      --images data/skyguard-v1/processed/images/train \
      --labels data/skyguard-v1/processed/labels/train \
      --output-images data/skyguard-v1/processed/images/train_aug \
      --output-labels data/skyguard-v1/processed/labels/train_aug \
      --count 3

This creates 3 augmented variants of every input image/label pair.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from skyguard.core.logger import setup_logging
from skyguard.data.augmentation import AugmentationPolicy, augment_image_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline augmentation for YOLO dataset")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-images", type=Path, required=True)
    parser.add_argument("--output-labels", type=Path, required=True)
    parser.add_argument("--count", type=int, default=3, help="Augmented copies per image")
    args = parser.parse_args()

    setup_logging()

    policy = AugmentationPolicy()

    image_paths = sorted(args.images.glob("*.jpg")) + sorted(args.images.glob("*.png"))
    total = 0

    for img_path in image_paths:
        label_path = args.labels / f"{img_path.stem}.txt"
        for i in range(args.count):
            out_img = args.output_images / f"{img_path.stem}_aug{i}{img_path.suffix}"
            out_lbl = args.output_labels / f"{img_path.stem}_aug{i}.txt"
            if augment_image_file(img_path, label_path, out_img, out_lbl, policy=policy):
                total += 1

    print(f"Augmented images written: {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
