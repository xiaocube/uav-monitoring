"""
Data quality filters for the SkyGuard dataset.

Low-quality frames kill model performance. We filter BEFORE annotation
so human labelers don't waste time on blurry/overexposed/duplicate images.

Supported checks:
  * Blur (Laplacian variance)
  * Overexposure / underexposure (mean brightness)
  * Near-duplicate frames (perceptual hash / histogram correlation)
  * Empty or unreadable images

All functions accept a Path or numpy array and return (keep: bool, reason: str).
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import cv2
import numpy as np

from skyguard.core.logger import get_logger

log = get_logger(__name__)


class QualityFilters:
    """Configurable image quality filters."""

    def __init__(
        self,
        blur_threshold: float = 100.0,
        overexposure_threshold: float = 250.0,
        underexposure_threshold: float = 10.0,
        duplicate_threshold: float = 0.98,
    ) -> None:
        self.blur_threshold = blur_threshold
        self.overexposure_threshold = overexposure_threshold
        self.underexposure_threshold = underexposure_threshold
        self.duplicate_threshold = duplicate_threshold

    def check(
        self,
        image: Union[Path, np.ndarray, str],
        reference: Union[Path, np.ndarray, str, None] = None,
    ) -> Tuple[bool, str]:
        """Run all quality checks. Returns (keep, reason)."""
        img = _load_image(image)
        if img is None:
            return False, "unreadable"

        ok, reason = self._check_blur(img)
        if not ok:
            return False, reason

        ok, reason = self._check_exposure(img)
        if not ok:
            return False, reason

        if reference is not None:
            ref = _load_image(reference)
            if ref is not None:
                ok, reason = self._check_duplicate(img, ref)
                if not ok:
                    return False, reason

        return True, "ok"

    def _check_blur(self, img: np.ndarray) -> Tuple[bool, str]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        if variance < self.blur_threshold:
            return False, f"blur:{variance:.1f}"
        return True, "ok"

    def _check_exposure(self, img: np.ndarray) -> Tuple[bool, str]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean = float(gray.mean())
        if mean > self.overexposure_threshold:
            return False, f"overexposed:{mean:.1f}"
        if mean < self.underexposure_threshold:
            return False, f"underexposed:{mean:.1f}"
        return True, "ok"

    def _check_duplicate(
        self,
        img: np.ndarray,
        ref: np.ndarray,
    ) -> Tuple[bool, str]:
        # Resize to small fixed size for robust comparison, convert gray.
        size = (64, 64)
        g1 = cv2.cvtColor(cv2.resize(img, size), cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(cv2.resize(ref, size), cv2.COLOR_BGR2GRAY)
        corr = cv2.compareHist(
            cv2.calcHist([g1], [0], None, [256], [0, 256]),
            cv2.calcHist([g2], [0], None, [256], [0, 256]),
            cv2.HISTCMP_CORREL,
        )
        if corr > self.duplicate_threshold:
            return False, f"duplicate:{corr:.3f}"
        return True, "ok"


def _load_image(image: Union[Path, np.ndarray, str]) -> np.ndarray | None:
    if isinstance(image, np.ndarray):
        return image
    path = Path(image)
    try:
        img = cv2.imread(str(path))
        return img
    except Exception as e:
        log.warning("Failed to load image {}: {}", path, e)
        return None
