"""
Real-time stream detection script for SkyGuard.

Single stream:
  python scripts/detect_stream.py --source 0 --model models/trained/drone-v1-2/weights/best.pt
  python scripts/detect_stream.py --source rtsp://admin:pass@192.168.1.100:554/stream --backend onnx

Multi stream:
  python scripts/detect_stream.py \
      --source cam1:rtsp://192.168.1.100 \
      --source cam2:rtsp://192.168.1.101 \
      --source usb:0 \
      --model models/trained/drone-v1-2/weights/best.onnx \
      --show

Flags:
  --show          Display with OpenCV (Mac GUI session only)
  --save-dir      Save annotated frames to directory
  --conf          Confidence threshold (default 0.25)
  --frame-skip    Process 1 frame per (skip+1) received (default 0)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in [str(SRC), str(ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import cv2

from skyguard.core.logger import setup_logging
from skyguard.vision.annotate import annotate
from skyguard.vision.async_detector import AsyncDetector


def parse_sources(sources: list[str]) -> list[tuple[str, str]]:
    """Parse --source arguments as stream_id:source pairs."""
    result = []
    for s in sources:
        if ":" in s:
            sid, src = s.split(":", 1)
        else:
            # Auto-generate stream_id
            sid = f"stream_{len(result)}"
            src = s
        result.append((sid, src))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="SkyGuard real-time stream detection")
    parser.add_argument("--source", action="append", required=True, help="stream_id:source or just source (repeatable)")
    parser.add_argument("--model", type=str, required=True, help="Model path (.pt / .onnx)")
    parser.add_argument("--backend", type=str, default="auto", help="auto | pytorch | onnx | tensorrt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--frame-skip", type=int, default=0, help="Process 1 in every N+1 frames")
    parser.add_argument("--max-fps", type=float, default=None, help="Cap capture FPS")
    parser.add_argument("--show", action="store_true", help="Display with OpenCV")
    parser.add_argument("--save-dir", type=Path, default=None, help="Save annotated frames")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0=unlimited)")
    args = parser.parse_args()

    setup_logging()

    sources = parse_sources(args.source)
    print(f"Sources: {sources}")
    print(f"Model: {args.model}")
    print(f"Backend: {args.backend}")
    print(f"Frame skip: {args.frame_skip}")

    # Create save directory
    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)

    detector = AsyncDetector(
        model_path=args.model,
        backend=args.backend,
        input_size=(args.imgsz, args.imgsz),
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        frame_skip=args.frame_skip,
    )

    for sid, src in sources:
        detector.add_stream(src, stream_id=sid, max_fps=args.max_fps)

    print("Starting detection... Press Ctrl+C to stop")

    try:
        with detector:
            total_frames = 0
            for result in detector.results_iter():
                total_frames += 1
                if args.max_frames and total_frames >= args.max_frames:
                    break

                # Print stats
                labels = [f"{d.class_name}:{d.confidence:.2f}" for d in result.detections]
                print(
                    f"[{result.stream_id}] frame={result.frame_id} "
                    f"detections={len(result.detections)} "
                    f"inference={result.inference_ms:.1f}ms "
                    f"objects={labels}"
                )

                # Annotate
                if result.detections:
                    annotated = annotate(
                        result.image,
                        boxes=[d.bbox for d in result.detections],
                        class_ids=[d.class_id for d in result.detections],
                        scores=[d.confidence for d in result.detections],
                    )
                else:
                    annotated = result.image

                # Save
                if args.save_dir:
                    path = args.save_dir / f"{result.stream_id}_{result.frame_id:06d}.jpg"
                    cv2.imwrite(str(path), annotated)

                # Show
                if args.show:
                    cv2.imshow(f"SkyGuard - {result.stream_id}", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        if args.show:
            cv2.destroyAllWindows()

    print(f"Total frames processed: {total_frames}")
    return 0


if __name__ == "__main__":
    sys.exit(main())