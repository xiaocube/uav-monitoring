"""
Multi-object tracking script for SkyGuard.

Combines detection + ByteTrack tracking on a video stream or image directory.

Usage:
  python scripts/track_video.py --source video.mp4 --model models/trained/best.pt
  python scripts/track_video.py --source 0 --model models/trained/best.onnx --show
  python scripts/track_video.py --source rtsp://... --model models/trained/best.pt --save-dir output/
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
from skyguard.vision.stream import VideoStream
from skyguard.vision.tracking_pipeline import TrackingPipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="SkyGuard multi-object tracking")
    parser.add_argument("--source", type=str, required=True, help="Video file, camera index, or RTSP URL")
    parser.add_argument("--model", type=str, required=True, help="Model path (.pt / .onnx)")
    parser.add_argument("--backend", type=str, default="auto")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence")
    parser.add_argument("--track-thresh", type=float, default=0.5, help="Tracker high-conf threshold")
    parser.add_argument("--match-thresh", type=float, default=0.8, help="IoU match threshold")
    parser.add_argument("--track-buffer", type=int, default=30, help="Track alive buffer (frames)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--show", action="store_true", help="Display with OpenCV")
    parser.add_argument("--save-dir", type=Path, default=None, help="Save annotated frames")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0=unlimited)")
    args = parser.parse_args()

    setup_logging()

    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)

    pipeline = TrackingPipeline(
        model_path=args.model,
        backend=args.backend,
        track_thresh=args.track_thresh,
        match_thresh=args.match_thresh,
        conf_threshold=args.conf,
        track_buffer=args.track_buffer,
    )

    # Determine source type
    src = args.source
    is_camera = src.isdigit()
    source = int(src) if is_camera else src

    print(f"Tracking on: {source}")
    print(f"Model: {args.model} (backend={args.backend})")
    print("Press Ctrl+C or 'q' to stop\n")

    total_frames = 0
    try:
        with VideoStream(source).open() as stream:
            for frame in stream:
                total_frames += 1
                if args.max_frames and total_frames > args.max_frames:
                    break

                result = pipeline.process_frame(frame, stream_id="main")

                # Print tracks
                if result.tracks:
                    track_strs = [
                        f"#{t.track_id} {t.class_name}({t.confidence:.2f})"
                        for t in result.tracks
                    ]
                    print(
                        f"frame {frame.frame_id}: {len(result.tracks)} tracks | "
                        f"inf={result.inference_ms:.1f}ms track={result.tracking_ms:.1f}ms | "
                        + " ".join(track_strs)
                    )

                # Annotate
                if result.tracks:
                    annotated = annotate(
                        frame.image,
                        boxes=[t.bbox for t in result.tracks],
                        class_ids=[t.class_id for t in result.tracks],
                        scores=[t.confidence for t in result.tracks],
                        track_ids=[t.track_id for t in result.tracks],
                    )
                else:
                    annotated = frame.image

                if args.save_dir:
                    path = args.save_dir / f"frame_{frame.frame_id:06d}.jpg"
                    cv2.imwrite(str(path), annotated)

                if args.show:
                    cv2.imshow("SkyGuard Tracking", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                if total_frames % 30 == 0:
                    log.info(
                        "Processed {} frames, current tracks: {}",
                        total_frames,
                        len(result.tracks),
                    )

    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        if args.show:
            cv2.destroyAllWindows()

    print(f"\nTotal frames processed: {total_frames}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
