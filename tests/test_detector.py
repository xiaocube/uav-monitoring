from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from skyguard.core.exceptions import ModelError
from skyguard.vision.classes import SkyGuardClass
from skyguard.vision.detector import YOLODetector


def _fake_result(num: int = 1) -> MagicMock:
    """Build a fake ultralytics Results object with `num` detections."""
    import torch

    boxes = MagicMock()
    xyxy = torch.tensor([[50.0, 50.0, 150.0, 150.0]] * num)
    conf = torch.tensor([0.87] * num)
    cls = torch.tensor([14.0] * num)  # COCO bird
    boxes.xyxy = xyxy
    boxes.conf = conf
    boxes.cls = cls
    boxes.__len__ = lambda self: num
    boxes.__bool__ = lambda self: num > 0

    r = MagicMock()
    r.orig_shape = (480, 640)
    r.boxes = boxes if num else None
    r.speed = {"inference": 12.3}
    return r


@patch("skyguard.vision.detector.YOLO")
def test_predict_returns_detection_result(mock_yolo, settings):  # noqa: F811
    detector = YOLODetector(device="cpu", conf=0.25, iou=0.45, imgsz=640)
    fake_model = MagicMock()
    fake_model.predict.return_value = [_fake_result(num=1)]
    mock_yolo.return_value = fake_model

    img = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detector.predict(img)

    assert result.model_name == detector.model_path
    assert result.img_width == 640
    assert result.img_height == 480
    assert len(result.detections) == 1

    d = result.detections[0]
    assert d.xyxy == (50, 50, 150, 150)
    assert d.confidence == pytest.approx(0.87)
    assert d.class_id == 14
    assert d.class_name == "bird"
    assert d.skyguard_class == SkyGuardClass.BIRD
    assert d.cx == 100
    assert d.cy == 100


@patch("skyguard.vision.detector.YOLO")
def test_predict_empty_result(mock_yolo):  # noqa: F811
    detector = YOLODetector(device="cpu")
    fake_model = MagicMock()
    r = MagicMock()
    r.orig_shape = (480, 640)
    r.boxes = None
    r.speed = {}
    fake_model.predict.return_value = [r]
    mock_yolo.return_value = fake_model

    img = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detector.predict(img)
    assert result.detections == []


@patch("skyguard.vision.detector.YOLO")
def test_predict_batch(mock_yolo):  # noqa: F811
    detector = YOLODetector(device="cpu", conf=0.25)
    fake_model = MagicMock()
    fake_model.return_value = [_fake_result(num=1), _fake_result(num=2)]
    mock_yolo.return_value = fake_model

    imgs = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(2)]
    results = detector.predict_batch(imgs)
    assert len(results) == 2
    assert len(results[0].detections) == 1
    assert len(results[1].detections) == 2


@patch("skyguard.vision.detector.YOLO", side_effect=RuntimeError("boom"))
def test_load_model_failure_raises_model_error(mock_yolo):  # noqa: F811
    detector = YOLODetector(device="cpu")
    with pytest.raises(ModelError):
        detector.predict(np.zeros((100, 100, 3), dtype=np.uint8))
