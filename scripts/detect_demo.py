"""
Quick demo: download a sample image and run YOLO detection.

Usage:
  python scripts/detect_demo.py

What it does:
  1) Downloads a small CC0 sample image from a public CDN.
  2) Saves it to data/samples/demo_airplane.jpg.
  3) Runs YOLOv8n on it.
  4) Writes annotated result to data/samples/demo_airplane_detected.jpg.
  5) Prints detected objects.

You can replace the URL with any local file:
  python scripts/detect_demo.py --source /path/to/your.jpg
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from skyguard.core.config import get_settings
from skyguard.core.logger import setup_logging
from skyguard.vision.annotate import annotate
from skyguard.vision.detector import YOLODetector
from skyguard.vision.stream import VideoStream

SAMPLE_URL = (
    "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg"
)


def _download_sample(url: str, dest: Path) -> None:
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)


def main() -> int:
    parser = argparse.ArgumentParser(description="SkyGuard YOLO detection demo")
    parser.add_argument(
        "--source",
        type=str,
        default="",
        help="Image/video path or RTSP URL. If empty, downloads a sample image.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output annotated path. Default: <source>_detected.<ext>",
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    setup_logging()
    get_settings()  # ensure config loads

    source = args.source
    if not source:
        sample_path = ROOT / "data" / "samples" / "demo_airplane.jpg"
        print(f"Downloading sample image to {sample_path} ...")
        _download_sample(SAMPLE_URL, sample_path)
        source = str(sample_path)

    detector = YOLODetector(
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        verbose=False,
    )

    src_path = Path(source)
    output = Path(args.output) if args.output else None
    is_stream = (
        source.lower().startswith(("rtsp://", "http://", "https://"))
        or source.isdigit()
    )
    is_image = src_path.is_file() and src_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    if is_stream:
        print(f"Running detection on stream: {source}")
        for frame in VideoStream.frames(source, limit=args.max_frames or None):
            result = detector.predict(frame.image)
            if frame.frame_id % 10 == 0:
                print(
                    f"  frame {frame.frame_id}: {len(result.detections)} object(s), "
                    f"{result.inference_ms or 0:.1f} ms"
                )
        return 0

    if is_image:
        print(f"Running detection on image: {source}")
        result = detector.predict(source)
        print(f"Found {len(result.detections)} object(s)")
        for d in result.detections:
            print(f"  - {d.class_name}: {d.confidence:.2f} @ {d.xyxy}")

        if is_image:
            import cv2

            img = cv2.imread(source)
            annotated = annotate(
                img,
                boxes=[d.xyxy for d in result.detections],
                class_ids=[d.class_id for d in result.detections],
                scores=[d.confidence for d in result.detections],
            )
            out_path = output or (src_path.parent / f"{src_path.stem}_detected{src_path.suffix}")
            cv2.imwrite(str(out_path), annotated)
            print(f"Annotated image saved to {out_path}")
    else:
        print(f"[warn] Unknown source type or file not found: {source}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
