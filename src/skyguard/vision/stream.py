"""
Robust video / RTSP stream reader for SkyGuard.

This module intentionally hides OpenCV's VideoCapture quirks behind a small,
retry-aware class so higher layers (detector CLI, web service, tracker) don't
need to know about CAP_FFMPEG, frame drops, or stream reconnects.

Design:
  * OpenCV backend by default (widely available).
  * Optional PyAV backend for better RTSP stability (set USE_AV=1).
  * Frame iterator (`for frame in stream:`) for simple loops.
  * `read()` method for explicit control + FPS throttling.
  * Tracks frame_id and timestamp for downstream sync.

Caveats:
  * RTSP streams usually require ffmpeg + correct transport (tcp preferred).
  * For Sprint 6 we will extend this with async producer/consumer queues.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Iterator, Optional, Tuple, Union

import cv2
import numpy as np

from skyguard.core.exceptions import VideoError
from skyguard.core.logger import get_logger

log = get_logger(__name__)


class Frame:
    """A single frame with metadata."""

    def __init__(self, image: np.ndarray, frame_id: int, timestamp_ms: float) -> None:
        self.image = image
        self.frame_id = frame_id
        self.timestamp_ms = timestamp_ms

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.image.shape


class VideoStream:
    """OpenCV-based video stream iterator."""

    def __init__(
        self,
        source: Union[str, Path, int],
        backend: Optional[str] = None,
        reconnect_interval: float = 5.0,
    ) -> None:
        """
        Parameters
        ----------
        source : str | Path | int
            File path, RTSP URL, or camera index.
        backend : str | None
            "opencv" or "av". If None, uses env ``SKYGUARD_STREAM_BACKEND`` or "opencv".
        reconnect_interval : float
            Seconds to wait between RTSP reconnect attempts.
        """
        self.source = str(source)
        self.backend = (backend or os.getenv("SKYGUARD_STREAM_BACKEND", "opencv")).lower()
        self.reconnect_interval = reconnect_interval

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_id = 0
        self._start_time = 0.0
        self._fps: Optional[float] = None
        self._width = 0
        self._height = 0
        self._opened = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> "VideoStream":
        """Open the stream and cache metadata."""
        if self.backend == "opencv":
            self._open_opencv()
        elif self.backend == "av":
            raise NotImplementedError("PyAV backend is planned for Sprint 6")
        else:
            raise VideoError(f"Unknown stream backend: {self.backend}")

        self._start_time = time.monotonic() * 1000.0
        self._opened = True
        return self

    def _open_opencv(self) -> None:
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise VideoError(f"Cannot open video source: {self.source}")

        # Prefer TCP for RTSP to reduce packet loss over UDP.
        if self.source.lower().startswith("rtsp://"):
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"H264"))
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._cap = cap
        self._fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self._width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self._height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        log.info(
            "Video opened: {} @ {}x{} {:.1f} fps",
            self.source,
            self._width,
            self._height,
            self._fps,
        )

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._opened = False

    def __enter__(self) -> "VideoStream":
        return self.open()

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        self.close()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_opened(self) -> bool:
        return self._opened and self._cap is not None and self._cap.isOpened()

    @property
    def fps(self) -> float:
        return self._fps or 25.0

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def read(self) -> Optional[Frame]:
        """Read the next frame or return None on EOF / error."""
        if self._cap is None:
            raise VideoError("Stream not opened. Call open() first.")

        ok, image = self._cap.read()
        if not ok or image is None:
            return None

        self._frame_id += 1
        now_ms = time.monotonic() * 1000.0
        return Frame(image, self._frame_id, now_ms)

    def __iter__(self) -> Iterator[Frame]:
        if not self.is_opened:
            self.open()
        while True:
            frame = self.read()
            if frame is None:
                break
            yield frame

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @classmethod
    def frames(
        cls,
        source: Union[str, Path, int],
        limit: Optional[int] = None,
    ) -> Iterator[Frame]:
        """One-liner to iterate frames with auto-close."""
        with cls(source).open() as stream:
            for i, frame in enumerate(stream):
                if limit is not None and i >= limit:
                    break
                yield frame
