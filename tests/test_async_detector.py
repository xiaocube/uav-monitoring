"""Tests for SkyGuard async real-time detection pipeline."""
from __future__ import annotations

import threading
import time
from queue import Queue

import numpy as np
import pytest

from skyguard.vision.async_detector import (
    AsyncDetector,
    DetectionFrame,
    InferenceWorker,
    StreamCaptureThread,
)


def test_detection_frame_dataclass():
    frame = DetectionFrame(
        stream_id="cam_01",
        frame_id=1,
        timestamp_ms=1000.0,
        image=np.zeros((480, 640, 3), dtype=np.uint8),
        detections=[],
        inference_ms=10.0,
    )
    assert frame.stream_id == "cam_01"
    assert frame.frame_id == 1
    assert frame.inference_ms == 10.0


def test_stream_capture_queue_bounded():
    q: "Queue" = Queue(maxsize=2)
    stop_event = threading.Event()

    # Simulate putting frames
    capture = StreamCaptureThread(
        stream_id="test",
        source="dummy",
        output_queue=q,
        stop_event=stop_event,
    )

    from skyguard.vision.stream import Frame

    frame1 = Frame(image=np.zeros((10, 10, 3), dtype=np.uint8), frame_id=1, timestamp_ms=0)
    frame2 = Frame(image=np.zeros((10, 10, 3), dtype=np.uint8), frame_id=2, timestamp_ms=0)
    frame3 = Frame(image=np.zeros((10, 10, 3), dtype=np.uint8), frame_id=3, timestamp_ms=0)

    capture._put_frame(frame1)
    capture._put_frame(frame2)
    capture._put_frame(frame3)  # Should drop oldest (frame1)

    assert q.qsize() == 2
    # frame2 should still be in queue (frame1 dropped)
    f = q.get_nowait()
    assert f.frame_id == 2


def test_async_detector_init():
    detector = AsyncDetector(
        model_path="models/trained/drone-v1-2/weights/best.pt",
        backend="pytorch",
        input_size=(640, 640),
        conf_threshold=0.5,
        iou_threshold=0.6,
        frame_skip=2,
    )
    assert detector.model_path == "models/trained/drone-v1-2/weights/best.pt"
    assert detector.backend == "pytorch"
    assert detector.conf_threshold == 0.5
    assert detector.iou_threshold == 0.6
    assert detector.frame_skip == 2
    assert len(detector._streams) == 0


def test_async_detector_add_stream():
    detector = AsyncDetector(model_path="dummy.pt")
    # We can't actually open a stream in tests, but we can verify the structure
    # Use a non-existent source - it will fail to connect but the thread starts
    detector.add_stream("/dev/null", stream_id="test_cam")
    assert "test_cam" in detector._streams
    assert "test_cam" in detector._raw_queues
    detector._stop_event.set()
    # Give thread time to exit
    time.sleep(0.1)


def test_async_detector_cannot_add_duplicate_stream():
    detector = AsyncDetector(model_path="dummy.pt")
    detector.add_stream("/dev/null", stream_id="cam_01")
    with pytest.raises(ValueError, match="already exists"):
        detector.add_stream("/dev/null", stream_id="cam_01")
    detector._stop_event.set()
    time.sleep(0.1)


def test_inference_worker_frame_skip():
    """Test that frame skip correctly drops frames."""
    input_q: "Queue" = Queue()
    output_q: "Queue" = Queue()
    stop_event = threading.Event()

    # Create a dummy backend
    class DummyBackend:
        def infer(self, image):
            class Result:
                output = np.zeros((1, 5, 8400))
                inference_ms = 5.0
            return Result()

        def postprocess(self, output, orig_shape):
            return []

    worker = InferenceWorker(
        input_queue=input_q,
        output_queue=output_q,
        backend=DummyBackend(),
        stop_event=stop_event,
        frame_skip=2,  # Process 1 in every 3 frames
    )
    worker.start()

    from skyguard.vision.stream import Frame

    # Put 3 frames
    for i in range(3):
        frame = Frame(image=np.zeros((10, 10, 3), dtype=np.uint8), frame_id=i, timestamp_ms=0)
        input_q.put(("cam_01", frame))

    # Wait for processing
    time.sleep(0.2)
    stop_event.set()
    worker.join(timeout=1.0)

    # Only 1 frame should have been processed (frame_skip=2 means 1 in 3)
    # Output queue should have at most 1 item
    assert output_q.qsize() <= 1


def test_async_detector_context_manager():
    detector = AsyncDetector(model_path="dummy.pt")
    # Context manager should not crash even with no streams
    with detector:
        pass
    assert detector._stop_event.is_set()
