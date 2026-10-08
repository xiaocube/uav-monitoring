from __future__ import annotations

from pathlib import Path

import pytest

from skyguard.core.exceptions import ConfigurationError
from skyguard.data.dataset import DatasetManifest, DatasetVersion


def test_manifest_add_image(tmp_path):
    manifest = DatasetManifest(name="skyguard-test", version="0.0.1")
    img_path = tmp_path / "img1.jpg"
    img_path.write_bytes(b"fake image")
    entry = manifest.add_image(
        image_path=img_path,
        split="train",
        width=640,
        height=480,
        scene="sunny",
        location="solar",
    )
    assert entry.split == "train"
    assert entry.width == 640
    assert entry.scene == "sunny"
    assert manifest.splits["train"] == 1


def test_manifest_save_and_load(tmp_path):
    manifest = DatasetManifest(name="skyguard-test", version="0.0.1")
    img_path = tmp_path / "img1.jpg"
    img_path.write_bytes(b"fake image")
    manifest.add_image(image_path=img_path, split="val", width=640, height=480)

    save_path = tmp_path / "manifest.json"
    manifest.save(save_path)
    assert save_path.exists()

    loaded = DatasetManifest.load(save_path)
    assert loaded.name == manifest.name
    assert len(loaded.images) == 1
    assert loaded.splits.get("val") == 1


def test_manifest_load_missing_raises():
    with pytest.raises(ConfigurationError):
        DatasetManifest.load(Path("/does/not/exist.json"))


def test_dataset_version_ensure_dirs(tmp_path):
    dv = DatasetVersion(root=tmp_path, version="1.0.0")
    dv.ensure_dirs()
    assert (tmp_path / "raw" / "videos").exists()
    assert (tmp_path / "interim" / "frames").exists()
    assert (tmp_path / "processed" / "images").exists()
    assert (tmp_path / "annotations").exists()
