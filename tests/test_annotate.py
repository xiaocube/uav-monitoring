from __future__ import annotations

import cv2
import numpy as np

from skyguard.vision.annotate import annotate


def _blank_image(h: int = 480, w: int = 640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_annotate_returns_new_image_same_shape():
    img = _blank_image()
    out = annotate(
        img,
        boxes=[(100, 100, 200, 200)],
        class_ids=[2],  # AIRPLANE
        scores=[0.88],
    )
    assert out is not img
    assert out.shape == img.shape
    assert out.dtype == img.dtype


def test_annotate_with_empty_inputs():
    img = _blank_image()
    out = annotate(img, boxes=[], class_ids=[], scores=[])
    assert out.shape == img.shape
    # No boxes means the canvas should remain a copy of the original.
    assert np.array_equal(out, img)


def test_annotate_clips_out_of_bounds_boxes():
    img = _blank_image(100, 100)
    out = annotate(
        img,
        boxes=[(-10, -10, 120, 120)],
        class_ids=[0],  # DRONE
        scores=[0.5],
    )
    assert out.shape == img.shape


def test_annotate_with_track_ids():
    img = _blank_image()
    out = annotate(
        img,
        boxes=[(50, 50, 100, 100)],
        class_ids=[1],  # BIRD
        scores=[0.7],
        track_ids=[42],
    )
    assert out.shape == img.shape


def test_annotate_ignores_invalid_class_id():
    img = _blank_image()
    # class_id 99 is not a SkyGuardClass; should not raise.
    out = annotate(
        img,
        boxes=[(10, 10, 50, 50)],
        class_ids=[99],
        scores=[0.6],
    )
    assert out.shape == img.shape
