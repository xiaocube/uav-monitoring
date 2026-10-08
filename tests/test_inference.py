"""Tests for SkyGuard inference backends."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from skyguard.inference.backend import (
    BaseBackend,
    InferenceBackend,
    Detection,
    InferenceResult,
)


def test_detection_dataclass():
    det = Detection(bbox=(10, 20, 100, 200), class_id=0, class_name="drone", confidence=0.95)
    assert det.bbox == (10, 20, 100, 200)
    assert det.class_id == 0
    assert det.class_name == "drone"
    assert det.confidence == 0.95


def test_inference_result_dataclass():
    result = InferenceResult(
        output=np.zeros((1, 5, 8400)),
        preprocess_ms=1.0,
        inference_ms=10.0,
        postprocess_ms=0.5,
        total_ms=11.5,
        backend="onnx",
        input_shape=(1, 3, 640, 640),
    )
    assert result.backend == "onnx"
    assert result.total_ms == 11.5


def test_backend_preprocess():
    """Test preprocessing logic."""
    # Create a mock backend to test preprocess
    class MockBackend(BaseBackend):
        def infer(self, image):
            pass

        def warmup(self, runs=3):
            pass

    backend = MockBackend("dummy.pt", input_size=(640, 640))
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    tensor, meta = backend.preprocess(image)

    assert tensor.shape == (1, 3, 640, 640)
    orig_h, orig_w, scale, (pad_h, pad_w) = meta
    assert orig_h == 480
    assert orig_w == 640


def test_inference_backend_create_unknown():
    with pytest.raises(Exception):  # ModelError
        InferenceBackend.create("unknown_backend", "model.pt")


def test_inference_backend_from_path_pt():
    backend = InferenceBackend.from_path("models/trained/drone-v1-2/weights/best.pt")
    assert backend.__class__.__name__ == "PyTorchBackend"


def test_inference_backend_from_path_onnx():
    backend = InferenceBackend.from_path("models/best.onnx")
    assert backend.__class__.__name__ == "ONNXBackend"


def test_inference_backend_from_path_unknown():
    with pytest.raises(Exception):  # ModelError
        InferenceBackend.from_path("model.unknown")