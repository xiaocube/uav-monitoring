"""
Visualization helpers for detection results.

Goals:
  * Fast (avoid expensive text measurement per frame).
  * Consistent (same color/font/thickness everywhere).
  * Localized (English + Chinese labels; Chinese needs freetype/ttf support,
    so we keep a simple pipeline for now and expose a flag for future i18n).
  * Non-destructive: annotate() returns a new BGR ndarray; it never mutates
    the source unless explicitly requested.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from skyguard.vision.classes import SkyGuardClass, get_color, get_label_cn, get_name

# OpenCV default font; good enough for 1920x1080 frames.
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.6
_FONT_THICKNESS = 1
_BOX_THICKNESS = 2


def _text_size(text: str) -> Tuple[Tuple[int, int], int]:
    return cv2.getTextSize(text, _FONT, _FONT_SCALE, _FONT_THICKNESS)


def annotate(
    image: np.ndarray,
    boxes: List[Tuple[int, int, int, int]],
    class_ids: List[int],
    scores: List[float],
    track_ids: Optional[List[int]] = None,
    show_label_cn: bool = False,
    alpha: float = 0.0,
) -> np.ndarray:
    """
    Draw bounding boxes and labels on a BGR image.

    Parameters
    ----------
    image : np.ndarray
        Input image in BGR format, shape (H, W, 3), dtype uint8.
    boxes : list of (x1, y1, x2, y2)
        Bounding boxes in pixel coordinates.
    class_ids : list of int
        Class IDs (COCO for now; SkyGuardClass after Sprint 4 retraining).
    scores : list of float
        Confidence scores in [0, 1].
    track_ids : list of int, optional
        If provided, appended to the label as "#id".
    show_label_cn : bool
        If True, use Chinese labels. Requires the host to render Chinese glyphs
        correctly; OpenCV default font may show '?' if not supported.
    alpha : float
        If > 0, fill the box region with a translucent overlay.

    Returns
    -------
    np.ndarray
        A new annotated image.
    """
    canvas = image.copy()
    track_ids = track_ids or [None] * len(boxes)

    for (x1, y1, x2, y2), cls_id, conf, tid in zip(boxes, class_ids, scores, track_ids):
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(image.shape[1], int(x2)), min(image.shape[0], int(y2))
        if x2 <= x1 or y2 <= y1:
            continue

        # Resolve color: try SkyGuard palette first, fall back to white.
        try:
            color = get_color(SkyGuardClass(cls_id))
        except (ValueError, KeyError):
            color = (255, 255, 255)

        # Optional translucent fill.
        if alpha > 0:
            overlay = canvas.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)

        # Box border.
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, _BOX_THICKNESS)

        # Label.
        try:
            label = get_label_cn(SkyGuardClass(cls_id)) if show_label_cn else get_name(SkyGuardClass(cls_id))
        except (ValueError, KeyError):
            label = f"class_{cls_id}"

        suffix = f" {conf:.2f}"
        if tid is not None:
            suffix += f" #{tid}"
        text = f"{label}{suffix}"

        (tw, th), baseline = _text_size(text)
        ty = max(y1 + th + 4, th + 4)
        cv2.rectangle(canvas, (x1, y1), (x1 + tw + 6, ty), color, -1)
        cv2.putText(
            canvas,
            text,
            (x1 + 3, ty - 3),
            _FONT,
            _FONT_SCALE,
            (255, 255, 255),
            _FONT_THICKNESS,
            cv2.LINE_AA,
        )

    return canvas
