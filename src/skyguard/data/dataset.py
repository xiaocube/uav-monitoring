"""
SkyGuard dataset manifest and version management.

A dataset version is the SINGLE source of truth for:
  * which images belong to train/val/test
  * class names and IDs
  * annotation file paths
  * augmentation / cleaning provenance
  * checksums for reproducibility

This module is deliberately lightweight (JSON on disk) so it works with or
without DVC installed. When DVC is present, large binary blobs are tracked
separately and the manifest stores the .dvc file reference.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from skyguard.core.exceptions import ConfigurationError
from skyguard.core.logger import get_logger

log = get_logger(__name__)


class ImageEntry(BaseModel):
    """A single image in the dataset manifest."""

    id: str
    path: str                    # relative to dataset root
    split: str                   # train | val | test
    width: int
    height: int
    md5: Optional[str] = None
    source: str = ""             # video filename, drone model, scene tag, ...
    scene: str = ""              # sunny / cloudy / rainy / dusk / night / ...
    location: str = ""           # city / mountain / solar / wind / industrial / campus


class DatasetManifest(BaseModel):
    """JSON-serializable manifest for a SkyGuard dataset version."""

    name: str = "skyguard-v1"
    version: str = "1.0.0"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    description: str = ""
    classes: List[str] = Field(default_factory=list)
    images: List[ImageEntry] = Field(default_factory=list)
    splits: Dict[str, int] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)

    def add_image(
        self,
        image_path: Path,
        split: str,
        width: int,
        height: int,
        source: str = "",
        scene: str = "",
        location: str = "",
    ) -> ImageEntry:
        entry = ImageEntry(
            id=image_path.stem,
            path=str(image_path),
            split=split,
            width=width,
            height=height,
            md5=_md5_file(image_path),
            source=source,
            scene=scene,
            location=location,
        )
        self.images.append(entry)
        self.splits[split] = self.splits.get(split, 0) + 1
        return entry

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, indent=2, ensure_ascii=False)
        log.info("Dataset manifest saved: {}", path)

    @classmethod
    def load(cls, path: Path) -> "DatasetManifest":
        if not path.exists():
            raise ConfigurationError(f"Dataset manifest not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


class DatasetVersion:
    """High-level helper around a versioned dataset folder."""

    def __init__(self, root: Path, version: str = "1.0.0") -> None:
        self.root = Path(root)
        self.version = version
        self.manifest_path = self.root / f"manifest-v{version}.json"

    @property
    def raw_dir(self) -> Path:
        return self.root / "raw"

    @property
    def interim_dir(self) -> Path:
        return self.root / "interim"

    @property
    def processed_dir(self) -> Path:
        return self.root / "processed"

    @property
    def annotations_dir(self) -> Path:
        return self.root / "annotations"

    def ensure_dirs(self) -> None:
        for d in [
            self.raw_dir / "videos",
            self.raw_dir / "images",
            self.interim_dir / "frames",
            self.interim_dir / "cleaned",
            self.processed_dir / "images",
            self.processed_dir / "labels",
            self.annotations_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    def load_manifest(self) -> DatasetManifest:
        return DatasetManifest.load(self.manifest_path)


def _md5_file(path: Path, chunk_size: int = 8192) -> Optional[str]:
    if not path.exists():
        return None
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()
