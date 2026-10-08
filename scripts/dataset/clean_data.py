"""
Clean an image folder by removing blurry / overexposed / underexposed / duplicate images.

Usage:
  python scripts/dataset/clean_data.py \
      --input data/skyguard-v1/interim/frames \
      --output data/skyguard-v1/interim/cleaned \
      --blur 100 \
      --duplicate 0.98

The script preserves the directory structure and writes a clean report CSV.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from skyguard.core.logger import setup_logging
from skyguard.data.cleaning import QualityFilters


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean image dataset")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blur", type=float, default=100.0)
    parser.add_argument("--overexposure", type=float, default=250.0)
    parser.add_argument("--underexposure", type=float, default=10.0)
    parser.add_argument("--duplicate", type=float, default=0.98)
    args = parser.parse_args()

    setup_logging()

    filters = QualityFilters(
        blur_threshold=args.blur,
        overexposure_threshold=args.overexposure,
        underexposure_threshold=args.underexposure,
        duplicate_threshold=args.duplicate,
    )

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "clean_report.csv"
    kept = 0
    removed = 0
    last_kept_image: cv2.Mat | None = None

    images = sorted(input_dir.rglob("*.jpg")) + sorted(input_dir.rglob("*.png"))

    with report_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "status", "reason"])

        for img_path in images:
            rel = img_path.relative_to(input_dir)
            keep, reason = filters.check(img_path, reference=last_kept_image)
            if keep:
                out_path = output_dir / rel
                out_path.parent.mkdir(parents=True, exist_ok=True)
                img = cv2.imread(str(img_path))
                cv2.imwrite(str(out_path), img)
                last_kept_image = img
                kept += 1
                writer.writerow([str(rel), "kept", reason])
            else:
                removed += 1
                writer.writerow([str(rel), "removed", reason])

    print(f"Kept: {kept}, Removed: {removed}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
