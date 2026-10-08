from __future__ import annotations

from pathlib import Path

import pytest

from skyguard.data.formats import Annotation, BoundingBox, COCODataset, YOLODataset


def test_bounding_box_normalized_to_xyxy():
    bbox = BoundingBox(
        x=0.25,
        y=0.25,
        w=0.5,
        h=0.5,
        image_width=640,
        image_height=480,
        normalized=True,
    )
    assert bbox.to_xyxy() == (160, 120, 480, 360)


def test_bounding_box_pixel_to_xyxy():
    bbox = BoundingBox.from_xyxy(
        x1=100, y1=100, x2=200, y2=200, image_width=640, image_height=480, normalized=False
    )
    assert bbox.to_xyxy() == (100, 100, 200, 200)


def test_bounding_box_to_yolo_line_pixel():
    bbox = BoundingBox.from_xyxy(
        x1=100, y1=100, x2=300, y2=300, image_width=640, image_height=480, normalized=False
    )
    line = bbox.to_yolo_line()
    xc, yc, w, h = map(float, line.split())
    assert pytest.approx(xc, rel=1e-4) == 200 / 640
    assert pytest.approx(yc, rel=1e-4) == 200 / 480
    assert pytest.approx(w, rel=1e-4) == 200 / 640
    assert pytest.approx(h, rel=1e-4) == 200 / 480


def test_bounding_box_to_yolo_line_normalized():
    bbox = BoundingBox(
        x=0.0,
        y=0.0,
        w=0.5,
        h=0.5,
        normalized=True,
    )
    line = bbox.to_yolo_line()
    xc, yc, w, h = map(float, line.split())
    assert pytest.approx(xc) == 0.25
    assert pytest.approx(yc) == 0.25
    assert pytest.approx(w) == 0.5
    assert pytest.approx(h) == 0.5


def test_yolo_dataset_read_write_labels(tmp_path):
    ds = YOLODataset(root=tmp_path)
    annotations = [
        Annotation(class_id=0, bbox=BoundingBox(x=0.1, y=0.1, w=0.2, h=0.2, normalized=True)),
        Annotation(class_id=1, bbox=BoundingBox(x=0.5, y=0.5, w=0.3, h=0.3, normalized=True)),
    ]
    ds.write_labels("img1", "train", annotations)

    read = ds.read_labels("img1", "train")
    assert len(read) == 2
    assert read[0].class_id == 0
    assert read[1].class_id == 1


def test_yolo_dataset_read_missing_returns_empty(tmp_path):
    ds = YOLODataset(root=tmp_path)
    assert ds.read_labels("missing", "train") == []


def test_coco_dataset_to_coco():
    coco = COCODataset(classes=["drone", "bird"])
    image_entries = [{"id": 0, "file_name": "img1.jpg", "width": 640, "height": 480}]
    bbox = BoundingBox.from_xyxy(
        x1=100, y1=100, x2=200, y2=200, image_width=640, image_height=480, normalized=False
    )
    out = coco.to_coco(image_entries, [(0, 0, bbox)])
    assert out["images"][0]["file_name"] == "img1.jpg"
    assert len(out["annotations"]) == 1
    assert out["annotations"][0]["bbox"] == [100, 100, 100, 100]


def test_coco_dataset_save_and_load(tmp_path):
    coco = COCODataset(classes=["drone"])
    image_entries = [{"id": 0, "file_name": "img1.jpg", "width": 640, "height": 480}]
    bbox = BoundingBox.from_xyxy(
        x1=10, y1=10, x2=50, y2=50, image_width=640, image_height=480, normalized=False
    )
    path = tmp_path / "coco.json"
    coco.save(path, image_entries, [(0, 0, bbox)])

    loaded = coco.load(path)
    assert loaded["images"][0]["width"] == 640
    assert loaded["annotations"][0]["category_id"] == 0
