"""
Tracking pipeline for SkyGuard.

Combines detection + tracking into a unified processing pipeline.
Handles per-stream tracker state and produces enriched output (track_id, age, etc.)

Usage:
    pipeline = TrackingPipeline(model_path="models/trained/best.onnx")
    for frame in stream:
        result = pipeline.process_frame(frame, stream_id="cam_01")
        for t in result.tracks:
            print(t.track_id, t.class_name, t.bbox, t.confidence)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from skyguard.core.logger import get_logger
from skyguard.inference.backend import Detection
from skyguard.vision.stream import Frame
from skyguard.vision.tracker import ByteTracker, Detection as TrackerDetection, Track

log = get_logger(__name__)


@dataclass
class TrackingResult:
    """Result of processing one frame."""

    stream_id: str
    frame_id: int
    timestamp_ms: float
    tracks: List[Track] = field(default_factory=list)
    inference_ms: float = 0.0
    tracking_ms: float = 0.0
    total_ms: float = 0.0


class TrackingPipeline:
    """Per-stream detection + tracking pipeline.

    One pipeline instance handles a single stream's tracker state.
    For multiple streams, create one pipeline per stream.

    Parameters
    ----------
    model_path : str
        Path to detection model.
    backend : str
        "auto" | "pytorch" | "onnx".
    track_thresh : float
        Tracker high-confidence threshold.
    match_thresh : float
        IoU threshold for association.
    conf_threshold : float
        Detection confidence threshold (low-conf used by tracker too).
    """

    def __init__(
        self,
        model_path: str,
        backend: str = "auto",
        track_thresh: float = 0.5,
        match_thresh: float = 0.8,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        track_buffer: int = 30,
        min_box_area: int = 100,
    ) -> None:
        from skyguard.inference.backend import InferenceBackend

        self._backend = InferenceBackend.from_path(
            model_path,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        ) if backend == "auto" else InferenceBackend.create(
            backend, model_path,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )
        self._tracker = ByteTracker(
            track_thresh=track_thresh,
            match_thresh=match_thresh,
            track_buffer=track_buffer,
            min_box_area=min_box_area,
        )

    def process_frame(
        self,
        frame: Frame,
        stream_id: str = "default",
    ) -> TrackingResult:
        """Process a single frame: detect + track."""
        t0 = time.perf_counter()

        # Detection
        infer_result = self._backend.infer(frame.image)
        detections = self._backend.postprocess(
            infer_result.output,
            orig_shape=(
                frame.image.shape[0],
                frame.image.shape[1],
                1.0,
                (0, 0),
            ),
        )
        inference_ms = infer_result.inference_ms

        # Convert to tracker input
        tracker_dets = [
            TrackerDetection(
                bbox=d.bbox,
                confidence=d.confidence,
                class_id=d.class_id,
                class_name=d.class_name,
            )
            for d in detections
        ]

        # Tracking
        t1 = time.perf_counter()
        tracks = self._tracker.update(tracker_dets)
        tracking_ms = (time.perf_counter() - t1) * 1000.0
        total_ms = (time.perf_counter() - t0) * 1000.0

        return TrackingResult(
            stream_id=stream_id,
            frame_id=frame.frame_id,
            timestamp_ms=frame.timestamp_ms,
            tracks=tracks,
            inference_ms=inference_ms,
            tracking_ms=tracking_ms,
            total_ms=total_ms,
        )

    def reset_tracker(self) -> None:
        self._tracker.reset()
