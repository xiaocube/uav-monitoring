"""
Asynchronous real-time detection pipeline for SkyGuard.

Architecture:
  +----------------+     +------------------+     +------------------+
  | Video Source 1 |---->| Frame Queue 1    |---->|                  |
  +----------------+     +------------------+     |   Inference      |
  +----------------+     +------------------+     |   Worker Pool    |----> Results
  | Video Source 2 |---->| Frame Queue 2    |---->|                  |
  +----------------+     +------------------+     +------------------+

Design:
  * Each source runs a dedicated capture thread (Producer).
  * A shared inference worker pool pulls frames and runs detection (Consumer).
  * Frame queues are bounded to prevent memory explosion on slow inference.
  * Frame skip strategy: always process the latest frame, drop stale ones.
  * Graceful shutdown with event flags.

Usage (single stream):
    detector = AsyncDetector(model_path="models/trained/drone-v1-2/weights/best.pt")
    detector.add_stream("rtsp://...", stream_id="cam_01")
    for result in detector.results_iter():
        print(result)

Usage (multi stream):
    detector.add_stream("rtsp://cam1", stream_id="cam_01")
    detector.add_stream("rtsp://cam2", stream_id="cam_02")
    detector.add_stream(0, stream_id="usb_01")  # local webcam
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from skyguard.core.exceptions import VideoError
from skyguard.core.logger import get_logger
from skyguard.inference.backend import Detection, InferenceBackend
from skyguard.vision.stream import Frame, VideoStream

log = get_logger(__name__)


@dataclass
class DetectionFrame:
    """Frame with detection results."""

    stream_id: str
    frame_id: int
    timestamp_ms: float
    image: np.ndarray
    detections: List[Detection] = field(default_factory=list)
    inference_ms: float = 0.0


class StreamCaptureThread(threading.Thread):
    """Producer: continuously capture frames from a video source."""

    def __init__(
        self,
        stream_id: str,
        source: str,
        output_queue: "Queue[Frame]",
        stop_event: threading.Event,
        reconnect_interval: float = 5.0,
        max_fps: Optional[float] = None,
    ) -> None:
        super().__init__(name=f"capture-{stream_id}", daemon=True)
        self.stream_id = stream_id
        self.source = source
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.reconnect_interval = reconnect_interval
        self.max_fps = max_fps
        self.min_interval = 1000.0 / max_fps if max_fps else None

        self._stream: Optional[VideoStream] = None
        self._last_frame_time = 0.0
        self._frame_count = 0
        self._is_connected = False

    def run(self) -> None:
        log.info("[{}] Capture thread started: {}", self.stream_id, self.source)
        while not self.stop_event.is_set():
            try:
                if not self._is_connected:
                    self._connect()

                assert self._stream is not None
                frame = self._stream.read()
                if frame is None:
                    # EOF or stream dropped
                    log.warning("[{}] Stream EOF / disconnected", self.stream_id)
                    self._disconnect()
                    time.sleep(self.reconnect_interval)
                    continue

                # FPS throttle
                if self.min_interval:
                    now = time.monotonic() * 1000.0
                    elapsed = now - self._last_frame_time
                    if elapsed < self.min_interval:
                        time.sleep((self.min_interval - elapsed) / 1000.0)
                    self._last_frame_time = now

                self._frame_count += 1
                self._put_frame(frame)

            except VideoError as e:
                log.error("[{}] Video error: {}", self.stream_id, e)
                self._disconnect()
                time.sleep(self.reconnect_interval)
            except Exception:
                log.exception("[{}] Unexpected error in capture thread", self.stream_id)
                self._disconnect()
                time.sleep(self.reconnect_interval)

        self._disconnect()
        log.info("[{}] Capture thread stopped", self.stream_id)

    def _connect(self) -> None:
        self._stream = VideoStream(self.source).open()
        self._is_connected = True
        log.info(
            "[{}] Connected: {}x{} @ {:.1f} fps",
            self.stream_id,
            self._stream.width,
            self._stream.height,
            self._stream.fps,
        )

    def _disconnect(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None
        self._is_connected = False

    def _put_frame(self, frame: Frame) -> None:
        """Put frame into queue, dropping oldest if full."""
        if self.output_queue.full():
            try:
                self.output_queue.get_nowait()
            except Empty:
                pass
        self.output_queue.put_nowait(frame)


class InferenceWorker(threading.Thread):
    """Consumer: pull frames from queue and run detection."""

    def __init__(
        self,
        input_queue: "Queue[Tuple[str, Frame]]",
        output_queue: "Queue[DetectionFrame]",
        backend,
        stop_event: threading.Event,
        frame_skip: int = 0,
    ) -> None:
        super().__init__(name="inference-worker", daemon=True)
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.backend = backend
        self.stop_event = stop_event
        self.frame_skip = frame_skip  # process 1 in every (skip+1) frames
        self._skip_counter: Dict[str, int] = {}

    def run(self) -> None:
        log.info("Inference worker started")
        while not self.stop_event.is_set():
            try:
                stream_id, frame = self.input_queue.get(timeout=0.5)
            except Empty:
                continue

            # Frame skip strategy
            self._skip_counter[stream_id] = self._skip_counter.get(stream_id, 0) + 1
            if self._skip_counter[stream_id] <= self.frame_skip:
                continue
            self._skip_counter[stream_id] = 0

            try:
                t0 = time.perf_counter()
                result = self.backend.infer(frame.image)
                detections = self.backend.postprocess(
                    result.output,
                    orig_shape=(
                        frame.image.shape[0],
                        frame.image.shape[1],
                        1.0,
                        (0, 0),
                    ),
                )
                inference_ms = (time.perf_counter() - t0) * 1000.0

                det_frame = DetectionFrame(
                    stream_id=stream_id,
                    frame_id=frame.frame_id,
                    timestamp_ms=frame.timestamp_ms,
                    image=frame.image,
                    detections=detections,
                    inference_ms=inference_ms,
                )

                # Drop oldest if output queue is full
                if self.output_queue.full():
                    try:
                        self.output_queue.get_nowait()
                    except Empty:
                        pass
                self.output_queue.put_nowait(det_frame)

            except Exception:
                log.exception("Inference error")

        log.info("Inference worker stopped")


class AsyncDetector:
    """High-level async detector managing multiple streams."""

    def __init__(
        self,
        model_path: str,
        backend: str = "auto",
        input_size: Tuple[int, int] = (640, 640),
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_queue_size: int = 5,
        frame_skip: int = 0,
    ) -> None:
        """
        Parameters
        ----------
        model_path : str
            Path to model (.pt / .onnx / .engine).
        backend : str
            "auto" | "pytorch" | "onnx" | "tensorrt".
        input_size : tuple
            (H, W) for model input.
        conf_threshold, iou_threshold : float
            Detection thresholds.
        max_queue_size : int
            Max frames queued per stream (bounded to prevent memory growth).
        frame_skip : int
            Process 1 frame for every (frame_skip + 1) frames received.
        """
        self.model_path = model_path
        self.backend = backend
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_queue_size = max_queue_size
        self.frame_skip = frame_skip

        self._stop_event = threading.Event()
        self._inference_backend = None
        self._streams: Dict[str, StreamCaptureThread] = {}
        self._raw_queues: Dict[str, "Queue[Frame]"] = {}
        self._inference_queue: "Queue[Tuple[str, Frame]]" = Queue(maxsize=max_queue_size * 2)
        self._result_queue: "Queue[DetectionFrame]" = Queue(maxsize=max_queue_size * 4)
        self._inference_worker: Optional[InferenceWorker] = None

    def _init_backend(self) -> InferenceBackend:
        if self._inference_backend is None:
            if self.backend == "auto":
                self._inference_backend = InferenceBackend.from_path(
                    self.model_path,
                    input_size=self.input_size,
                    conf_threshold=self.conf_threshold,
                    iou_threshold=self.iou_threshold,
                )
            else:
                self._inference_backend = InferenceBackend.create(
                    self.backend,
                    self.model_path,
                    input_size=self.input_size,
                    conf_threshold=self.conf_threshold,
                    iou_threshold=self.iou_threshold,
                )
        return self._inference_backend

    def add_stream(
        self,
        source: str,
        stream_id: str,
        max_fps: Optional[float] = None,
    ) -> None:
        """Add a video source to the detector.

        Parameters
        ----------
        source : str
            File path, RTSP URL, or camera index.
        stream_id : str
            Unique identifier for this stream.
        max_fps : float | None
            Cap capture FPS (None = no limit).
        """
        if stream_id in self._streams:
            raise ValueError(f"Stream {stream_id} already exists")

        raw_queue: "Queue[Frame]" = Queue(maxsize=self.max_queue_size)
        self._raw_queues[stream_id] = raw_queue

        capture = StreamCaptureThread(
            stream_id=stream_id,
            source=source,
            output_queue=raw_queue,
            stop_event=self._stop_event,
            max_fps=max_fps,
        )
        self._streams[stream_id] = capture
        capture.start()

        log.info("Added stream {}: {}", stream_id, source)

    def start(self) -> None:
        """Start the inference worker and mux frames from all streams."""
        backend = self._init_backend()

        # Start inference worker
        self._inference_worker = InferenceWorker(
            input_queue=self._inference_queue,
            output_queue=self._result_queue,
            backend=backend,
            stop_event=self._stop_event,
            frame_skip=self.frame_skip,
        )
        self._inference_worker.start()

        # Start mux thread: pull frames from all raw queues into inference queue
        self._mux_thread = threading.Thread(target=self._mux_loop, daemon=True)
        self._mux_thread.start()

        log.info("AsyncDetector started with {} stream(s)", len(self._streams))

    def _mux_loop(self) -> None:
        """Multiplex frames from multiple stream queues into the inference queue."""
        while not self._stop_event.is_set():
            for stream_id, raw_queue in list(self._raw_queues.items()):
                try:
                    frame = raw_queue.get_nowait()
                    if self._inference_queue.full():
                        try:
                            self._inference_queue.get_nowait()
                        except Empty:
                            pass
                    self._inference_queue.put_nowait((stream_id, frame))
                except Empty:
                    continue
            time.sleep(0.001)  # 1ms polling interval

    def get_result(self, timeout: float = 0.5) -> Optional[DetectionFrame]:
        """Get a detection result (non-blocking)."""
        try:
            return self._result_queue.get(timeout=timeout)
        except Empty:
            return None

    def results_iter(self, timeout: float = 1.0):
        """Iterate over detection results (blocking generator)."""
        while not self._stop_event.is_set():
            result = self.get_result(timeout=timeout)
            if result is not None:
                yield result

    def stop(self) -> None:
        """Stop all threads gracefully."""
        log.info("Stopping AsyncDetector...")
        self._stop_event.set()

        for stream_id, capture in self._streams.items():
            capture.join(timeout=5.0)
            if capture.is_alive():
                log.warning("[{}] Capture thread did not stop in time", stream_id)

        if self._inference_worker:
            self._inference_worker.join(timeout=5.0)

        if self._mux_thread:
            self._mux_thread.join(timeout=2.0)

        log.info("AsyncDetector stopped")

    def __enter__(self) -> "AsyncDetector":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):  # type: ignore[no-untyped-def]
        self.stop()
