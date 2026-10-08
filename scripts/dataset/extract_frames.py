"""
Extract frames from videos for dataset creation.

Usage:
  python scripts/dataset/extract_frames.py \
      --input data/skyguard-v1/raw/videos \
      --output data/skyguard-v1/interim/frames \
      --interval 1.0 \
      --scene sunny \
      --location solar

Rules:
  * Skip consecutive frames that are too similar (histogram correlation).
  * Apply quality filters (blur / exposure).
  * Preserve source metadata in the output filename.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from skyguard.core.logger import setup_logging
from skyguard.data.cleaning import QualityFilters
from skyguard.vision.stream import VideoStream


def _similar_to_last(img: cv2.Mat, last: cv2.Mat | None, threshold: float = 0.98) -> bool:
    if last is None:
        return False
    size = (64, 64)
    g1 = cv2.cvtColor(cv2.resize(img, size), cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(cv2.resize(last, size), cv2.COLOR_BGR2GRAY)
    corr = cv2.compareHist(
        cv2.calcHist([g1], [0], None, [256], [0, 256]),
        cv2.calcHist([g2], [0], None, [256], [0, 256]),
        cv2.HISTCMP_CORREL,
    )
    return corr > threshold


def extract(
    video_path: Path,
    output_dir: Path,
    interval: float = 1.0,
    filters: QualityFilters | None = None,
    scene: str = "",
    location: str = "",
) -> int:
    filters = filters or QualityFilters()
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    last_frame = None
    next_time = 0.0

    with VideoStream(str(video_path)).open() as stream:
        for frame in stream:
            ms = frame.frame_id / stream.fps * 1000.0
            if ms < next_time:
                continue
            next_time = ms + interval * 1000.0

            if _similar_to_last(frame.image, last_frame):
                continue
            last_frame = frame.image.copy()

            keep, reason = filters.check(frame.image)
            if not keep:
                print(f"  skip frame {frame.frame_id}: {reason}")
                continue

            stem = f"{video_path.stem}_f{frame.frame_id:06d}_s{scene}_l{location}"
            out_path = output_dir / f"{stem}.jpg"
            cv2.imwrite(str(out_path), frame.image)
            saved += 1

    return saved


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract frames from videos")
    parser.add_argument("--input", type=Path, required=True, help="Video file or directory")
    parser.add_argument("--output", type=Path, required=True, help="Output frames directory")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between frames")
    parser.add_argument("--scene", type=str, default="", help="Scene tag (sunny/cloudy/...)")
    parser.add_argument("--location", type=str, default="", help="Location tag")
    parser.add_argument("--blur", type=float, default=100.0, help="Minimum Laplacian variance")
    args = parser.parse_args()

    setup_logging()

    filters = QualityFilters(blur_threshold=args.blur)

    input_path = Path(args.input)
    videos = [input_path] if input_path.is_file() else sorted(input_path.glob("*"))

    total = 0
    for video in videos:
        if not video.is_file():
            continue
        print(f"Extracting from {video.name} ...")
        total += extract(
            video,
            args.output,
            interval=args.interval,
            filters=filters,
            scene=args.scene,
            location=args.location,
        )

    print(f"Total frames saved: {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
