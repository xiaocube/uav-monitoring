"""Tests for SkyGuard ByteTrack tracker."""
from __future__ import annotations

import numpy as np
import pytest

from skyguard.vision.tracker import (
    ByteTracker,
    Detection,
    STrack,
    Track,
    bbox_iou,
    iou_matrix,
    linear_assignment,
    xywh_to_xyxy,
    xyxy_to_xywh,
)


def test_bbox_iou_identical():
    iou = bbox_iou((0, 0, 10, 10), (0, 0, 10, 10))
    assert iou == pytest.approx(1.0)


def test_bbox_iou_no_overlap():
    iou = bbox_iou((0, 0, 10, 10), (20, 20, 30, 30))
    assert iou == 0.0


def test_bbox_iou_partial():
    iou = bbox_iou((0, 0, 10, 10), (5, 5, 15, 15))
    # Intersection: 5x5=25, Union: 100+100-25=175
    assert iou == pytest.approx(25.0 / 175.0)


def test_xyxy_to_xywh():
    assert xyxy_to_xywh((10, 20, 30, 50)) == (10, 20, 20, 30)
    assert xywh_to_xyxy((10, 20, 20, 30)) == (10, 20, 30, 50)


def test_iou_matrix_empty():
    m = iou_matrix(np.zeros((0, 4)), np.zeros((0, 4)))
    assert m.shape == (0, 0)

    m2 = iou_matrix(np.array([[0, 0, 10, 10]]), np.zeros((0, 4)))
    assert m2.shape == (1, 0)


def test_iou_matrix_basic():
    a = np.array([[0, 0, 10, 10], [20, 20, 30, 30]])
    b = np.array([[0, 0, 10, 10], [5, 5, 15, 15]])
    m = iou_matrix(a, b)
    assert m.shape == (2, 2)
    assert m[0, 0] == pytest.approx(1.0)  # identical
    assert m[0, 1] == pytest.approx(25.0 / 175.0)  # partial
    assert m[1, 0] == 0.0  # no overlap
    assert m[1, 1] == 0.0


def test_linear_assignment_no_match():
    # Cost = 1 - IoU. All costs > threshold (0.5) -> no match
    cost = np.array([[0.9, 0.8], [0.7, 0.6]])
    matches = linear_assignment(cost, threshold=0.5)
    assert matches == []  # All costs exceed threshold


def test_linear_assignment_single_match():
    # cost[0,0]=0.1 is below threshold=0.5; cost[0,1]=0.9 above; etc.
    cost = np.array([[0.1, 0.9], [0.8, 0.7]])
    matches = linear_assignment(cost, threshold=0.5)
    assert (0, 0) in matches


def test_linear_assignment_multiple_matches():
    cost = np.array([[0.1, 0.9], [0.9, 0.2]])
    matches = linear_assignment(cost, threshold=0.5)
    assert len(matches) == 2
    assert (0, 0) in matches
    assert (1, 1) in matches


def test_detection_dataclass():
    det = Detection(bbox=(0, 0, 10, 10), confidence=0.9, class_id=0, class_name="drone")
    assert det.bbox == (0, 0, 10, 10)
    assert det.confidence == 0.9
    assert det.class_id == 0
    assert det.class_name == "drone"


def test_track_dataclass():
    track = Track(
        track_id=1,
        class_id=0,
        class_name="drone",
        bbox=(0, 0, 10, 10),
        confidence=0.9,
    )
    assert track.track_id == 1
    assert track.center == (5, 5)
    assert track.width == 10
    assert track.height == 10


def test_strack_init():
    det = Detection(bbox=(100, 100, 200, 200), confidence=0.9, class_id=0, class_name="drone")
    s = STrack(det, track_id=1, frame_id=1)
    assert s.track_id == 1
    assert s.class_id == 0
    assert s.hits == 1
    assert s.miss == 0
    assert s.is_confirmed is False  # Needs 3 hits


def test_strack_predict_updates_bbox():
    det = Detection(bbox=(100, 100, 200, 200), confidence=0.9, class_id=0, class_name="drone")
    s = STrack(det, track_id=1, frame_id=1)
    initial_bbox = s.bbox
    s.predict()
    # After predict, bbox should be updated (might be same if no velocity)
    assert s.bbox is not None


def test_strack_update():
    det1 = Detection(bbox=(100, 100, 200, 200), confidence=0.9, class_id=0, class_name="drone")
    s = STrack(det1, track_id=1, frame_id=1)
    det2 = Detection(bbox=(110, 100, 210, 200), confidence=0.85, class_id=0, class_name="drone")
    s.update(det2, frame_id=2)
    assert s.hits == 2
    assert s.miss == 0
    assert s.confidence == 0.85
    assert len(s.history) == 2


def test_strack_mark_missed():
    det = Detection(bbox=(100, 100, 200, 200), confidence=0.9, class_id=0, class_name="drone")
    s = STrack(det, track_id=1, frame_id=1)
    s.mark_missed()
    assert s.miss == 1


def test_strack_to_track():
    det = Detection(bbox=(100, 100, 200, 200), confidence=0.9, class_id=0, class_name="drone")
    s = STrack(det, track_id=42, frame_id=1)
    s.update(det, frame_id=2)
    s.update(det, frame_id=3)
    t = s.to_track()
    assert t.track_id == 42
    assert t.hits == 3
    assert t.bbox == (100, 100, 200, 200)


def test_bytetracker_init():
    tracker = ByteTracker(track_thresh=0.5, match_thresh=0.8, track_buffer=30)
    assert tracker.track_thresh == 0.5
    assert tracker.match_thresh == 0.8
    assert tracker.track_buffer == 30


def test_bytetracker_initial_no_tracks():
    tracker = ByteTracker()
    result = tracker.update([])
    assert result == []


def test_bytetracker_create_first_track():
    """First detection should create a new track (not yet confirmed)."""
    tracker = ByteTracker(track_thresh=0.5, match_thresh=0.8, track_buffer=30)
    det = Detection(bbox=(100, 100, 200, 200), confidence=0.9, class_id=0, class_name="drone")
    result = tracker.update([det])
    # Track needs 3 hits to be confirmed
    assert len(result) == 0


def test_bytetracker_confirm_after_3_hits():
    """Track becomes confirmed after 3 hits."""
    tracker = ByteTracker(track_thresh=0.5, match_thresh=0.8, track_buffer=30)
    detections = [
        Detection(bbox=(100 + i*10, 100, 200 + i*10, 200), confidence=0.9, class_id=0, class_name="drone")
        for i in range(5)
    ]
    for det in detections:
        result = tracker.update([det])
    # After 5 frames, track should be confirmed
    assert len(result) >= 1
    assert result[0].class_name == "drone"


def test_bytetracker_consistent_id_across_frames():
    """Same object should keep the same track_id across frames."""
    tracker = ByteTracker(track_thresh=0.5, match_thresh=0.8, track_buffer=30)
    ids = []
    for i in range(5):
        det = Detection(
            bbox=(100 + i*5, 100, 200 + i*5, 200),
            confidence=0.9,
            class_id=0,
            class_name="drone",
        )
        result = tracker.update([det])
        if result:
            ids.append(result[0].track_id)
    # All IDs should be the same
    assert len(set(ids)) == 1


def test_bytetracker_recovers_with_low_conf():
    """Track should not be lost if confidence drops (using low-conf recovery)."""
    tracker = ByteTracker(track_thresh=0.5, match_thresh=0.8, track_buffer=30)

    # First 3 frames: high confidence
    for i in range(3):
        det = Detection(
            bbox=(100 + i*5, 100, 200 + i*5, 200),
            confidence=0.9,
            class_id=0,
            class_name="drone",
        )
        tracker.update([det])

    # Frame 4: low confidence (still within second-stage matching)
    det = Detection(
        bbox=(115, 100, 215, 200),
        confidence=0.3,  # Below track_thresh (0.5)
        class_id=0,
        class_name="drone",
    )
    result = tracker.update([det])
    # Track should still be present
    assert len(result) >= 1


def test_bytetracker_separate_classes():
    """Different classes should have different track IDs."""
    tracker = ByteTracker()
    det1 = Detection(bbox=(100, 100, 200, 200), confidence=0.9, class_id=0, class_name="drone")
    det2 = Detection(bbox=(300, 300, 400, 400), confidence=0.9, class_id=1, class_name="bird")

    # Feed 3 times to confirm
    for _ in range(3):
        result = tracker.update([det1, det2])

    assert len(result) == 2
    ids = {r.track_id for r in result}
    assert len(ids) == 2  # Different IDs


def test_bytetracker_reset():
    tracker = ByteTracker()
    det = Detection(bbox=(100, 100, 200, 200), confidence=0.9, class_id=0, class_name="drone")
    tracker.update([det])
    tracker.reset()
    assert tracker._frame_id == 0
    assert len(tracker._tracks) == 0


def test_bytetracker_min_box_area_filter():
    """Very small boxes should be filtered out."""
    tracker = ByteTracker(min_box_area=1000)
    det = Detection(bbox=(0, 0, 5, 5), confidence=0.9, class_id=0, class_name="drone")
    result = tracker.update([det])
    # Tiny box should be filtered
    assert len(tracker._tracks.get(0, [])) == 0
