"""
SkyGuard CLI entry point.

Sprint 2 adds the `detect` subcommand for image / video / stream inference.
Later sprints will add:
  * skyguard track   --source rtsp://...
  * skyguard serve   --config production.yaml
  * skyguard train   --data data/skyguard-v1
  * ...
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from skyguard import __version__
from skyguard.core.config import describe, get_settings
from skyguard.core.logger import get_logger, setup_logging
from skyguard.utils.environment import verify_cli
from skyguard.utils.device import benchmark_cli

cli = typer.Typer(
    name="skyguard",
    add_completion=False,
    no_args_is_help=True,
    help="SkyGuard - AI Low-Altitude Intelligent Monitoring Platform",
)

from skyguard.db.cli import db_cli
cli.add_typer(db_cli, name="db")

from skyguard.edge.cli import edge_cli
cli.add_typer(edge_cli, name="edge")
console = Console()
log = get_logger("skyguard.main")


@cli.command()
def info() -> None:
    """Show resolved configuration and runtime environment."""
    setup_logging()
    s = get_settings()
    console.print(
        Panel.fit(
            describe(),
            title=f"[bold cyan]SkyGuard {__version__}[/bold cyan]",
            border_style="cyan",
        )
    )


@cli.command()
def verify() -> None:
    """Run the full environment verification suite."""
    raise SystemExit(verify_cli())


@cli.command()
def bench() -> None:
    """Print available compute backends."""
    raise SystemExit(benchmark_cli())


@cli.command()
def train(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Training config YAML."),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Base model (e.g., yolov8n.pt)."),
    data: Optional[str] = typer.Option(None, "--data", "-d", help="YOLO data.yaml path."),
    epochs: int = typer.Option(100, "--epochs", "-e", help="Training epochs."),
    batch: int = typer.Option(16, "--batch", "-b", help="Batch size."),
    imgsz: int = typer.Option(640, "--imgsz", help="Input image size."),
    device: Optional[str] = typer.Option(None, "--device", help="cpu | cuda | mps | auto."),
    project: str = typer.Option("models/trained", "--project", help="Output directory."),
    name: str = typer.Option("skyguard-v1", "--name", "-n", help="Experiment name."),
    export_model: bool = typer.Option(False, "--export", help="Export to ONNX after training."),
    no_tracking: bool = typer.Option(False, "--no-tracking", help="Disable experiment tracking."),
) -> None:
    """Train a custom YOLO model on the SkyGuard dataset."""
    from skyguard.training.trainer import TrainingConfig, YOLOTrainer

    setup_logging()

    if config and config.exists():
        cfg = TrainingConfig.from_yaml(config)
    else:
        default_cfg = Path("config/training/default.yaml")
        cfg = TrainingConfig.from_yaml(default_cfg) if default_cfg.exists() else TrainingConfig()

    # CLI overrides
    if model:
        cfg.base_model = model
    if data:
        cfg.data = data
    cfg.epochs = epochs
    cfg.batch = batch
    cfg.imgsz = imgsz
    if device:
        cfg.device = device
    cfg.project = project
    cfg.name = name
    if no_tracking:
        cfg.tracking["backend"] = "none"

    trainer = YOLOTrainer(cfg)
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task("Training...", total=None)
        results = trainer.train()

    console.print("[green]Training complete![/green]")
    best = trainer.best_model_path
    if best:
        console.print(f"Best model: {best}")

    if export_model:
        console.print("Exporting model...")
        exported = trainer.export()
        for p in exported:
            console.print(f"  Exported: {p}")


@cli.command()
def detect(
    source: str = typer.Argument(..., help="Image, video file, RTSP URL, or camera index."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output image/video path."),
    conf: float = typer.Option(0.25, "--conf", "-c", help="Confidence threshold."),
    iou: float = typer.Option(0.45, "--iou", "-i", help="NMS IoU threshold."),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model path (default from config)."),
    device: Optional[str] = typer.Option(None, "--device", "-d", help="cpu | cuda | mps | auto."),
    save_annotated: bool = typer.Option(True, "--save/--no-save", help="Write annotated result."),
    show: bool = typer.Option(False, "--show", help="Display with OpenCV (Mac: only GUI sessions)."),
    max_frames: int = typer.Option(0, "--max-frames", "-n", help="Stop after N frames (0 = unlimited)."),
) -> None:
    """Run YOLO detection on an image or video stream."""
    # Lazy imports: detection is heavy (PyTorch/Ultralytics/OpenCV). Keep `info`/`verify` fast.
    import cv2

    from skyguard.vision.annotate import annotate
    from skyguard.vision.detector import YOLODetector
    from skyguard.vision.stream import VideoStream

    setup_logging()
    detector = YOLODetector(
        model_path=model,
        device=device,
        conf=conf,
        iou=iou,
        verbose=False,
    )

    src_path = Path(source)
    is_stream = (
        source.lower().startswith(("rtsp://", "http://", "https://"))
        or source.isdigit()
    )
    is_image = src_path.is_file() and src_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    if is_image:
        _detect_image(detector, src_path, output, save_annotated, show)
    else:
        _detect_stream(detector, source, output, save_annotated, show, max_frames)


@cli.command(name="track")
def track(
    source: str = typer.Argument(..., help="Video file, camera index, or RTSP URL."),
    model: str = typer.Option(..., "--model", "-m", help="Model path (.pt / .onnx)"),
    backend: str = typer.Option("auto", "--backend", "-b"),
    conf: float = typer.Option(0.25, "--conf", "-c"),
    track_thresh: float = typer.Option(0.5, "--track-thresh"),
    match_thresh: float = typer.Option(0.8, "--match-thresh"),
    track_buffer: int = typer.Option(30, "--track-buffer"),
    show: bool = typer.Option(False, "--show"),
    save_dir: Optional[Path] = typer.Option(None, "--save-dir"),
    max_frames: int = typer.Option(0, "--max-frames", "-n"),
) -> None:
    """Multi-object tracking (detection + ByteTrack)."""
    import cv2

    from skyguard.vision.annotate import annotate
    from skyguard.vision.stream import VideoStream
    from skyguard.vision.tracking_pipeline import TrackingPipeline

    setup_logging()

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    pipeline = TrackingPipeline(
        model_path=model,
        backend=backend,
        track_thresh=track_thresh,
        match_thresh=match_thresh,
        conf_threshold=conf,
        track_buffer=track_buffer,
    )

    is_camera = source.isdigit()
    src = int(source) if is_camera else source

    console.print(f"[cyan]Tracking: {src}[/cyan] (model={model})")
    console.print("[dim]Press Ctrl+C or 'q' to stop[/dim]")

    total = 0
    try:
        with VideoStream(src).open() as stream:
            for frame in stream:
                total += 1
                if max_frames and total > max_frames:
                    break

                result = pipeline.process_frame(frame, stream_id="main")
                if result.tracks:
                    console.print(
                        f"frame {frame.frame_id}: {len(result.tracks)} tracks | "
                        f"inf={result.inference_ms:.1f}ms track={result.tracking_ms:.1f}ms"
                    )
                    for t in result.tracks:
                        console.print(
                            f"  #{t.track_id} {t.class_name}({t.confidence:.2f}) "
                            f"age={t.age}f hits={t.hits} bbox={t.bbox}"
                        )

                if result.tracks and (show or save_dir):
                    annotated = annotate(
                        frame.image,
                        boxes=[t.bbox for t in result.tracks],
                        class_ids=[t.class_id for t in result.tracks],
                        scores=[t.confidence for t in result.tracks],
                        track_ids=[t.track_id for t in result.tracks],
                    )
                else:
                    annotated = frame.image

                if save_dir:
                    path = save_dir / f"frame_{frame.frame_id:06d}.jpg"
                    cv2.imwrite(str(path), annotated)

                if show:
                    cv2.imshow("SkyGuard Tracking", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped by user[/yellow]")
    finally:
        if show:
            cv2.destroyAllWindows()

    console.print(f"[green]Total frames: {total}[/green]")


@cli.command(name="serve")
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-H", help="Bind address."),
    port: int = typer.Option(8000, "--port", "-p", help="Listen port."),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code changes."),
    workers: int = typer.Option(1, "--workers", "-w", help="Number of worker processes."),
) -> None:
    """Start the SkyGuard API server."""
    from skyguard.api.server import start
    
    setup_logging()
    console.print(f"[cyan]Starting SkyGuard API server on {host}:{port}[/cyan]")
    console.print(f"[dim]Docs: http://{host}:{port}/docs[/dim]")
    console.print("[dim]Press Ctrl+C to stop[/dim]")
    
    try:
        start(host=host, port=port, reload=reload, workers=workers)
    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped[/yellow]")


@cli.command(name="detect-stream")
def detect_stream(
    source: List[str] = typer.Option([], "--source", "-s", help="stream_id:source (repeatable)"),
    model: str = typer.Option(..., "--model", "-m", help="Model path (.pt / .onnx)"),
    backend: str = typer.Option("auto", "--backend", "-b", help="auto | pytorch | onnx | tensorrt"),
    conf: float = typer.Option(0.25, "--conf", "-c"),
    iou: float = typer.Option(0.45, "--iou", "-i"),
    imgsz: int = typer.Option(640, "--imgsz"),
    frame_skip: int = typer.Option(0, "--frame-skip", help="Process 1 in every N+1 frames"),
    max_fps: Optional[float] = typer.Option(None, "--max-fps"),
    show: bool = typer.Option(False, "--show", help="Display with OpenCV"),
    save_dir: Optional[Path] = typer.Option(None, "--save-dir"),
) -> None:
    """Real-time async detection on one or more video streams."""
    import cv2

    from skyguard.vision.annotate import annotate
    from skyguard.vision.async_detector import AsyncDetector

    setup_logging()

    if not source:
        console.print("[red]Error: At least one --source required[/red]")
        raise typer.Exit(code=1)

    detector = AsyncDetector(
        model_path=model,
        backend=backend,
        input_size=(imgsz, imgsz),
        conf_threshold=conf,
        iou_threshold=iou,
        frame_skip=frame_skip,
    )

    for s in source:
        if ":" in s:
            sid, src = s.split(":", 1)
        else:
            sid = f"stream_{len(detector._streams)}"
            src = s
        detector.add_stream(src, stream_id=sid, max_fps=max_fps)

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[cyan]Starting async detection on {len(source)} stream(s)...[/cyan]")
    console.print("[dim]Press Ctrl+C to stop[/dim]")

    try:
        with detector:
            for result in detector.results_iter():
                labels = [f"{d.class_name}:{d.confidence:.2f}" for d in result.detections]
                console.print(
                    f"[{result.stream_id}] frame={result.frame_id} "
                    f"detections={len(result.detections)} "
                    f"inference={result.inference_ms:.1f}ms"
                )

                if result.detections:
                    annotated = annotate(
                        result.image,
                        boxes=[d.bbox for d in result.detections],
                        class_ids=[d.class_id for d in result.detections],
                        scores=[d.confidence for d in result.detections],
                    )
                else:
                    annotated = result.image

                if save_dir:
                    path = save_dir / f"{result.stream_id}_{result.frame_id:06d}.jpg"
                    cv2.imwrite(str(path), annotated)

                if show:
                    cv2.imshow(f"SkyGuard - {result.stream_id}", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped by user[/yellow]")
    finally:
        if show:
            cv2.destroyAllWindows()


def _detect_image(
    detector: YOLODetector,
    source: Path,
    output: Optional[Path],
    save_annotated: bool,
    show: bool,
) -> None:
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task("Detecting...", total=None)
        result = detector.predict(str(source))

    console.print(f"Found {len(result.detections)} object(s) in {source}")
    for d in result.detections:
        console.print(f"  - {d.class_name}: {d.confidence:.2f} @ {d.xyxy}")

    if save_annotated or show:
        img = cv2.imread(str(source))
        if img is None:
            log.error("Cannot read image: {}", source)
            raise typer.Exit(code=1)
        annotated = annotate(
            img,
            boxes=[d.xyxy for d in result.detections],
            class_ids=[d.class_id for d in result.detections],
            scores=[d.confidence for d in result.detections],
        )
        out_path = output or Path(f"{source.stem}_detected{source.suffix}")
        if save_annotated:
            cv2.imwrite(str(out_path), annotated)
            console.print(f"Saved annotated image to {out_path}")
        if show:
            cv2.imshow("SkyGuard Detection", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def _detect_stream(
    detector: YOLODetector,
    source: str,
    output: Optional[Path],
    save_annotated: bool,
    show: bool,
    max_frames: int,
) -> None:
    console.print(f"Detecting on stream: {source}")
    writer: Optional[cv2.VideoWriter] = None
    writer_path: Optional[Path] = None

    try:
        with VideoStream(source).open() as stream:
            for frame in stream:
                if max_frames and frame.frame_id > max_frames:
                    break

                result = detector.predict(frame.image)
                annotated = annotate(
                    frame.image,
                    boxes=[d.xyxy for d in result.detections],
                    class_ids=[d.class_id for d in result.detections],
                    scores=[d.confidence for d in result.detections],
                )

                if save_annotated and writer is None and output:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer_path = output
                    writer = cv2.VideoWriter(
                        str(writer_path),
                        fourcc,
                        stream.fps,
                        (stream.width, stream.height),
                    )

                if writer:
                    writer.write(annotated)

                if show:
                    cv2.imshow("SkyGuard Detection", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                if frame.frame_id % 30 == 0:
                    console.print(
                        f"  frame {frame.frame_id}: {len(result.detections)} object(s), "
                        f"inference {result.inference_ms or 0:.1f} ms"
                    )
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped by user[/yellow]")
    finally:
        if writer:
            writer.release()
            console.print(f"Saved annotated video to {writer_path}")
        if show:
            cv2.destroyAllWindows()


@cli.callback()
def _root(
    version: bool = typer.Option(False, "--version", help="Show version and exit."),
) -> None:
    if version:
        typer.echo(f"skyguard {__version__}")
        raise typer.Exit()


def main() -> int:
    """Used by `python -m skyguard` and the `skyguard` console script."""
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
