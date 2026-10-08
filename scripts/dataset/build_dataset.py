"""
One-click SkyGuard dataset builder.

This script orchestrates the full pipeline:
  1) Extract frames from raw videos
  2) Clean images (blur / exposure / duplicates)
  3) Copy cleaned images into processed/images/{train,val,test}
  4) Split into train/val/test (default 80/10/10)
  5) Generate manifest.json and YOLO data.yaml

After this script, you are ready to annotate in Label Studio / CVAT / Roboflow.
Once annotated, run `scripts/dataset/convert_format.py` to generate COCO JSON.

Usage:
  python scripts/dataset/build_dataset.py \
      --root data/skyguard-v1 \
      --video-dir data/skyguard-v1/raw/videos \
      --scene sunny \
      --location solar \
      --train-ratio 0.8 \
      --val-ratio 0.1
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for p in [str(SRC), str(ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from skyguard.core.logger import setup_logging
from skyguard.data.cleaning import QualityFilters
from skyguard.data.dataset import DatasetManifest, DatasetVersion
from skyguard.data.formats import YOLODataset
from skyguard.vision.classes import SkyGuardClass


def _split_file(path: Path, train_r: float, val_r: float) -> str:
    x = random.random()
    if x < train_r:
        return "train"
    if x < train_r + val_r:
        return "val"
    return "test"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build SkyGuard dataset from raw videos")
    parser.add_argument("--root", type=Path, default=ROOT / "data" / "skyguard-v1")
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--scene", type=str, default="")
    parser.add_argument("--location", type=str, default="")
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--blur", type=float, default=100.0)
    parser.add_argument("--duplicate", type=float, default=0.98)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging()
    random.seed(args.seed)

    ds = DatasetVersion(root=args.root, version="1.0.0")
    ds.ensure_dirs()

    interim_frames = ds.interim_dir / "frames"
    interim_cleaned = ds.interim_dir / "cleaned"

    # Step 1: extract frames
    print("==> Step 1/5: Extract frames from videos")
    from scripts.dataset.extract_frames import extract

    filters = QualityFilters(blur_threshold=args.blur)
    total_extracted = 0
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".webm"}
    for video in sorted(args.video_dir.glob("*")):
        if not video.is_file() or video.suffix.lower() not in video_extensions:
            continue
        total_extracted += extract(
            video,
            interim_frames,
            interval=args.interval,
            filters=filters,
            scene=args.scene,
            location=args.location,
        )
    print(f"    Extracted: {total_extracted}")

    # Step 2: clean
    print("==> Step 2/5: Clean images")
    clean_filters = QualityFilters(
        blur_threshold=args.blur,
        duplicate_threshold=args.duplicate,
    )
    kept = 0
    last_kept_image = None
    for img_path in sorted(interim_frames.rglob("*.jpg")):
        keep, reason = clean_filters.check(img_path, reference=last_kept_image)
        if keep:
            rel = img_path.relative_to(interim_frames)
            out_path = interim_cleaned / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, out_path)
            last_kept_image = cv2.imread(str(img_path))
            kept += 1
        else:
            print(f"    drop {img_path.name}: {reason}")
    print(f"    Kept: {kept}")

    # Step 3/4: split and copy to processed
    print("==> Step 3/5: Split train/val/test")
    manifest = DatasetManifest(
        name="skyguard-v1",
        version="1.0.0",
        description=f"scene={args.scene}, location={args.location}",
        classes=[c.name.lower() for c in SkyGuardClass],
    )

    for img_path in sorted(interim_cleaned.rglob("*.jpg")):
        split = _split_file(img_path, args.train_ratio, args.val_ratio)
        out_img_dir = ds.processed_dir / "images" / split
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_img_dir / img_path.name
        shutil.copy2(img_path, out_path)

        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        manifest.add_image(
            image_path=out_path.relative_to(ds.root),
            split=split,
            width=w,
            height=h,
            source=img_path.stem.split("_f")[0],
            scene=args.scene,
            location=args.location,
        )

    manifest.save(ds.manifest_path)
    print(f"    Manifest: {ds.manifest_path}")

    # Step 5: YOLO data.yaml
    print("==> Step 5/5: Generate YOLO data.yaml")
    yolo = YOLODataset(root=ds.processed_dir, classes=manifest.classes)
    yolo.write_yaml(ds.processed_dir / "data.yaml")
    print("    Done. Next: annotate images in processed/images/* and write labels to processed/labels/*")
    return 0


if __name__ == "__main__":
    sys.exit(main())
