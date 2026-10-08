from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from skyguard.data.cleaning import QualityFilters


def _blank_image(h: int = 100, w: int = 100, value: int = 128) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


def _noise_image(h: int = 100, w: int = 100, mean: int = 128, std: int = 20) -> np.ndarray:
    np.random.seed(42)
    noise = np.random.normal(mean, std, (h, w, 3)).astype(np.uint8)
    return np.clip(noise, 0, 255).astype(np.uint8)


def test_quality_filter_accepts_normal_image():
    img = _noise_image(mean=128)
    filters = QualityFilters()
    keep, reason = filters.check(img)
    assert keep is True
    assert reason == "ok"


def test_quality_filter_rejects_unreadable():
    filters = QualityFilters()
    keep, reason = filters.check(Path("/does/not/exist.jpg"))
    assert keep is False
    assert reason == "unreadable"


def test_quality_filter_rejects_blur():
    img = _blank_image(value=128)
    filters = QualityFilters(blur_threshold=1.0)
    keep, reason = filters.check(img)
    assert keep is False
    assert reason.startswith("blur")


def test_quality_filter_rejects_overexposure():
    img = np.full((100, 100, 3), 254, dtype=np.uint8)
    filters = QualityFilters(blur_threshold=0.0, overexposure_threshold=250.0)
    keep, reason = filters.check(img)
    assert keep is False
    assert reason.startswith("overexposed")


def test_quality_filter_rejects_underexposure():
    img = np.full((100, 100, 3), 5, dtype=np.uint8)
    filters = QualityFilters(blur_threshold=0.0, underexposure_threshold=10.0)
    keep, reason = filters.check(img)
    assert keep is False
    assert reason.startswith("underexposed")


def test_quality_filter_detects_duplicate():
    img1 = _noise_image(mean=128)
    img2 = _noise_image(mean=128)
    filters = QualityFilters(duplicate_threshold=0.99)
    keep, reason = filters.check(img2, reference=img1)
    assert keep is False
    assert reason.startswith("duplicate")
