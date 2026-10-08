"""
ByteTrack multi-object tracker for SkyGuard.

ByteTrack is a SOTA multi-object tracking algorithm published in ECCV 2022.
It performs data association in two stages:
  1. First associate high-confidence detections (conf >= high_thresh) with existing tracks
  2. Second, associate the LOW confidence detections (low_thresh <= conf < high_thresh)
     with remaining unmatched tracks (this recovers objects in occlusion / motion blur)

Key ideas:
  * Use ALL detection boxes, not just high-confidence ones
  * IoU-based association with Kalman-filtered motion prediction
  * Per-class tracking (separated by class_id)

Reference:
  Zhang, Y. et al. (2022). ByteTrack: Multi-Object Tracking by Associating Every
  Detection Box. ECCV 2022. arXiv:2110.06864

Usage:
    tracker = ByteTracker(track_thresh=0.5, match_thresh=0.8, frame_rate=30)
    for frame_id, detections in enumerate(detection_stream):
        online_targets = tracker.update(detections)
        for t in online_targets:
            print(t.track_id, t.class_name, t.bbox)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from filterpy.kalman import KalmanFilter

from skyguard.core.logger import get_logger

log = get_logger(__name__)


@dataclass
class Detection:
    """Input detection to the tracker."""

    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str = ""


@dataclass
class Track:
    """A single tracked object with its state."""

    track_id: int
    class_id: int
    class_name: str
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    age: int = 0
    hits: int = 1
    miss: int = 0
    history: List[Tuple[int, int, int, int]] = field(default_factory=list)

    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]


def bbox_iou(b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]) -> float:
    """Compute IoU between two boxes in xyxy format."""
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def xyxy_to_xywh(b: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    """xyxy -> xywh (top-left + width/height)."""
    return (b[0], b[1], b[2] - b[0], b[3] - b[1])


def xywh_to_xyxy(b: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    """xywh -> xyxy."""
    return (b[0], b[1], b[0] + b[2], b[1] + b[3])


def iou_matrix(
    boxes_a: np.ndarray, boxes_b: np.ndarray
) -> np.ndarray:
    """Compute IoU matrix between two sets of boxes (xyxy)."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    a = np.asarray(boxes_a, dtype=np.float32)
    b = np.asarray(boxes_b, dtype=np.float32)

    # Top-left and bottom-right corners
    a_tl = a[:, None, :2]
    a_br = a[:, None, 2:]
    b_tl = b[None, :, :2]
    b_br = b[None, :, 2:]

    inter_tl = np.maximum(a_tl, b_tl)
    inter_br = np.minimum(a_br, b_br)
    inter_wh = np.maximum(0, inter_br - inter_tl)
    inter = inter_wh[..., 0] * inter_wh[..., 1]

    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter

    return inter / np.maximum(union, 1e-6)


def linear_assignment(cost: np.ndarray, threshold: float) -> List[Tuple[int, int]]:
    """Greedy assignment by minimum cost below threshold.

    `cost` is a dissimilarity matrix (higher = worse). Pairs are picked
    greedily in order of lowest cost. Stops when best remaining cost
    exceeds the threshold. Returns list of (row, col) index pairs.

    ByteTrack uses this simple greedy matcher instead of Hungarian —
    faster and works well in practice.
    """
    cost = cost.copy()
    matches: List[Tuple[int, int]] = []
    if cost.size == 0:
        return matches

    while True:
        idx = np.unravel_index(np.argmin(cost), cost.shape)
        i, j = int(idx[0]), int(idx[1])
        if cost[i, j] > threshold:
            break
        matches.append((i, j))
        cost[i, :] = np.inf
        cost[:, j] = np.inf
    return matches


class STrack:
    """Internal track state with a Kalman filter for motion prediction.

    State vector (8D): [cx, cy, w, h, vx, vy, vw, vh]
    Measurement (4D): [cx, cy, w, h]
    """

    def __init__(self, detection: Detection, track_id: int, frame_id: int) -> None:
        self.track_id = track_id
        self.class_id = detection.class_id
        self.class_name = detection.class_name
        self.confidence = detection.confidence
        self.bbox: Tuple[int, int, int, int] = detection.bbox
        self.hits = 1
        self.miss = 0
        self.last_observation_frame = frame_id
        self.start_frame = frame_id
        self.history: List[Tuple[int, int, int, int]] = [detection.bbox]

        # Initialize Kalman filter
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        # State transition (constant velocity model)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ], dtype=np.float32)
        # Measurement matrix
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ], dtype=np.float32)
        # Measurement noise
        self.kf.R[2:, 2:] *= 10.0
        # State covariance
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        # Process noise
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        # Initialize state with detection
        x, y, w, h = xyxy_to_xywh(detection.bbox)
        self.kf.x[:4] = np.array([[x], [y], [w], [h]], dtype=np.float32)

    def predict(self) -> None:
        """Advance state by one time step. Returns predicted bbox."""
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        if (self.kf.x[7] + self.kf.x[3]) <= 0:
            self.kf.x[7] *= 0.0
        self.kf.predict()
        x, y, w, h = self.kf.x[:4].flatten()
        x, y, w, h = int(x), int(y), int(w), int(h)
        self.bbox = xywh_to_xyxy((x, y, max(0, w), max(0, h)))

    def update(self, detection: Detection, frame_id: int) -> None:
        """Update state with new detection."""
        x, y, w, h = xyxy_to_xywh(detection.bbox)
        measurement = np.array([[x], [y], [w], [h]], dtype=np.float32)
        self.kf.update(measurement)
        self.bbox = detection.bbox
        self.confidence = detection.confidence
        self.miss = 0
        self.hits += 1
        self.last_observation_frame = frame_id
        self.history.append(detection.bbox)

    def mark_missed(self) -> None:
        self.miss += 1

    @property
    def is_confirmed(self) -> bool:
        return self.hits >= 3  # Need at least 3 hits to confirm

    def to_track(self) -> Track:
        return Track(
            track_id=self.track_id,
            class_id=self.class_id,
            class_name=self.class_name,
            bbox=self.bbox,
            confidence=self.confidence,
            age=0,
            hits=self.hits,
            miss=self.miss,
            history=list(self.history),
        )


class ByteTracker:
    """ByteTrack multi-object tracker.

    Parameters
    ----------
    track_thresh : float
        High-confidence threshold. Detections with conf >= this value
        are used in the first matching pass.
    match_thresh : float
        IoU threshold for matching. Lower = stricter, higher = more permissive.
    track_buffer : int
        Number of frames to keep a track alive without observation.
    min_box_area : int
        Filter out detections with area smaller than this (pixels).
    frame_rate : int
        Processing frame rate (informational).
    """

    def __init__(
        self,
        track_thresh: float = 0.5,
        match_thresh: float = 0.8,
        track_buffer: int = 30,
        min_box_area: int = 100,
        frame_rate: int = 30,
    ) -> None:
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.min_box_area = min_box_area
        self.frame_rate = frame_rate

        # Tracks maintained per class
        self._tracks: Dict[int, List[STrack]] = {}
        self._next_id = 1
        self._frame_id = 0

    def _next_track_id(self) -> int:
        tid = self._next_id
        self._next_id += 1
        return tid

    def update(self, detections: List[Detection]) -> List[Track]:
        """Update tracker with new detections and return active tracks.

        This implements the ByteTrack algorithm:
          1. Separate detections into high-conf and low-conf
          2. Predict existing tracks
          3. First association: high-conf with ALL tracks (including unconfirmed)
          4. Second association: low-conf with remaining unmatched tracks
          5. Initialize new tracks from unmatched high-conf detections
          6. Mark unmatched tracks as missed; remove old tracks
        """
        self._frame_id += 1

        # Filter by min area
        dets = [d for d in detections if self._box_area(d.bbox) >= self.min_box_area]
        # Split into high / low confidence
        high_dets = [d for d in dets if d.confidence >= self.track_thresh]
        low_dets = [d for d in dets if d.confidence < self.track_thresh]

        # Group existing tracks by class (ALL tracks, not just confirmed)
        all_tracks: List[STrack] = []
        for cls_tracks in self._tracks.values():
            all_tracks.extend(cls_tracks)

        # Predict step
        for track in all_tracks:
            track.predict()

        # Step 1: Associate high-confidence detections with ALL tracks
        # (Including unconfirmed ones, so they can accumulate hits)
        unmatched_tracks = list(all_tracks)
        unmatched_dets = list(high_dets)

        if unmatched_tracks and unmatched_dets:
            iou = self._iou_cost_matrix(unmatched_tracks, unmatched_dets)
            # Cost = 1 - IoU, threshold = 1 - match_thresh
            cost = 1.0 - iou
            matches = linear_assignment(cost, threshold=1.0 - self.match_thresh)

            matched_tracks = set()
            matched_dets = set()
            for track_idx, det_idx in matches:
                unmatched_tracks[track_idx].update(unmatched_dets[det_idx], self._frame_id)
                matched_tracks.add(track_idx)
                matched_dets.add(det_idx)
            unmatched_tracks = [t for i, t in enumerate(unmatched_tracks) if i not in matched_tracks]
            unmatched_dets = [d for i, d in enumerate(unmatched_dets) if i not in matched_dets]

        # Step 2: Associate low-confidence detections with remaining unmatched tracks
        if unmatched_tracks and low_dets:
            iou = self._iou_cost_matrix(unmatched_tracks, low_dets)
            cost = 1.0 - iou
            matches = linear_assignment(cost, threshold=0.5)  # Stricter for low-conf

            matched_tracks = set()
            matched_dets = set()
            for track_idx, det_idx in matches:
                unmatched_tracks[track_idx].update(low_dets[det_idx], self._frame_id)
                matched_tracks.add(track_idx)
                matched_dets.add(det_idx)
            unmatched_tracks = [t for i, t in enumerate(unmatched_tracks) if i not in matched_tracks]

        # Step 3: Mark unmatched tracks as missed
        for track in unmatched_tracks:
            track.mark_missed()

        # Step 4: Initialize new tracks from unmatched high-conf detections
        for det in unmatched_dets:
            new_track = STrack(det, self._next_track_id(), self._frame_id)
            self._tracks.setdefault(det.class_id, []).append(new_track)

        # Step 5: Remove dead tracks (missed for too many frames)
        for class_id in list(self._tracks.keys()):
            self._tracks[class_id] = [
                t for t in self._tracks[class_id]
                if t.miss <= self.track_buffer
            ]
            if not self._tracks[class_id]:
                del self._tracks[class_id]

        # Return confirmed tracks
        output: List[Track] = []
        for class_id, tracks in self._tracks.items():
            for t in tracks:
                if t.is_confirmed:
                    track = t.to_track()
                    track.age = self._frame_id - t.start_frame
                    output.append(track)

        return output

    @staticmethod
    def _box_area(b: Tuple[int, int, int, int]) -> int:
        return max(0, b[2] - b[0]) * max(0, b[3] - b[1])

    @staticmethod
    def _iou_cost_matrix(tracks: List[STrack], dets: List[Detection]) -> np.ndarray:
        """Compute IoU cost matrix between tracks and detections."""
        if not tracks or not dets:
            return np.zeros((len(tracks), len(dets)), dtype=np.float32)
        track_boxes = np.array([t.bbox for t in tracks], dtype=np.float32)
        det_boxes = np.array([d.bbox for d in dets], dtype=np.float32)
        return iou_matrix(track_boxes, det_boxes)

    def reset(self) -> None:
        """Reset tracker state."""
        self._tracks.clear()
        self._next_id = 1
        self._frame_id = 0
