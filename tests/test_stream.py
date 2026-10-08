from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from skyguard.core.exceptions import VideoError
from skyguard.vision.stream import VideoStream


def test_stream_missing_file_raises():
    with pytest.raises(VideoError):
        with VideoStream("/this/path/does/not/exist.mp4").open():
            pass


def test_stream_unsupported_backend_raises():
    with pytest.raises(VideoError):
        with VideoStream("/dev/null", backend="cuda").open():
            pass


def test_stream_context_manager_closes(tmp_path):
    # Create a tiny valid video using OpenCV.
    import cv2

    video_path = tmp_path / "test.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 1.0, (320, 240))
    for _ in range(3):
        writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
    writer.release()

    stream = VideoStream(str(video_path))
    with stream.open() as s:
        assert s.is_opened
        assert s.width == 320
        assert s.height == 240
        frame = s.read()
        assert frame is not None
        assert frame.frame_id == 1
    assert not stream.is_opened
