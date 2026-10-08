"""SkyGuard Database Module.

PostgreSQL + TimescaleDB integration for persistent storage and event streaming.
"""

from __future__ import annotations

from .manager import DatabaseManager
from .models import (
    AlertModel,
    CameraConfigModel,
    DetectionModel,
    DeviceModel,
    PTZConfigModel,
    TrackModel,
    ZoneConfigModel,
)
from .types import DatabaseConfig

__all__ = [
    "DatabaseManager",
    "DatabaseConfig",
    "DeviceModel",
    "DetectionModel",
    "TrackModel",
    "AlertModel",
    "CameraConfigModel",
    "PTZConfigModel",
    "ZoneConfigModel",
]