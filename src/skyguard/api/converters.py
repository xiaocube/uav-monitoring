"""Converters between database models and API models."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

from skyguard.api.models import (
    AlertInfo,
    CameraConfig,
    DeviceInfo,
    DetectionBox,
    DetectionInfo,
    PTZConfig,
    TrackInfo,
    ZoneConfig,
)
from skyguard.db.models import (
    AlertModel,
    CameraConfigModel,
    DetectionModel,
    DeviceModel,
    PTZConfigModel,
    TrackModel,
    ZoneConfigModel,
)


def device_to_api(model: DeviceModel) -> DeviceInfo:
    """Convert DeviceModel to DeviceInfo."""
    return DeviceInfo(
        id=model.id,
        name=model.name,
        type=model.type,
        status=model.status,
        host=model.host,
        port=model.port,
        config=model.config or {},
        last_seen=model.last_seen,
        error=model.error,
    )


def device_from_api(info: DeviceInfo) -> Dict[str, Any]:
    """Convert DeviceInfo to dict for DeviceModel."""
    return {
        "id": info.id,
        "name": info.name,
        "type": info.type.value if hasattr(info.type, "value") else info.type,
        "status": info.status.value if hasattr(info.status, "value") else info.status,
        "host": info.host,
        "port": info.port,
        "config": info.config,
        "last_seen": info.last_seen,
        "error": info.error,
    }


def detection_to_api(model: DetectionModel) -> DetectionInfo:
    """Convert DetectionModel to DetectionInfo."""
    return DetectionInfo(
        id=model.id,
        class_id=model.class_id,
        class_name=model.class_name,
        confidence=model.confidence,
        bbox=DetectionBox(
            x1=model.bbox_x1,
            y1=model.bbox_y1,
            x2=model.bbox_x2,
            y2=model.bbox_y2,
        ),
        timestamp=model.timestamp,
    )


def detection_from_api(info: DetectionInfo, device_id: str, frame_id: str, frame_width: int, frame_height: int, fps: float = 0.0) -> Dict[str, Any]:
    """Convert DetectionInfo to dict for DetectionModel."""
    return {
        "id": info.id,
        "frame_id": frame_id,
        "device_id": device_id,
        "timestamp": info.timestamp,
        "class_id": info.class_id,
        "class_name": info.class_name,
        "confidence": info.confidence,
        "bbox_x1": info.bbox.x1,
        "bbox_y1": info.bbox.y1,
        "bbox_x2": info.bbox.x2,
        "bbox_y2": info.bbox.y2,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "fps": fps,
    }


def track_to_api(model: TrackModel) -> TrackInfo:
    """Convert TrackModel to TrackInfo."""
    return TrackInfo(
        track_id=model.track_id,
        class_id=model.class_id,
        class_name=model.class_name,
        confidence=model.confidence,
        bbox=DetectionBox(
            x1=model.bbox_x1,
            y1=model.bbox_y1,
            x2=model.bbox_x2,
            y2=model.bbox_y2,
        ),
        velocity=(model.velocity_x, model.velocity_y) if model.velocity_x is not None else None,
        age=model.age,
        status=model.status,
    )


def track_from_api(info: TrackInfo, device_id: str, timestamp: datetime) -> Dict[str, Any]:
    """Convert TrackInfo to dict for TrackModel."""
    return {
        "id": f"{device_id}_{info.track_id}_{timestamp.timestamp()}",
        "track_id": info.track_id,
        "device_id": device_id,
        "timestamp": timestamp,
        "class_id": info.class_id,
        "class_name": info.class_name,
        "confidence": info.confidence,
        "bbox_x1": info.bbox.x1,
        "bbox_y1": info.bbox.y1,
        "bbox_x2": info.bbox.x2,
        "bbox_y2": info.bbox.y2,
        "velocity_x": info.velocity[0] if info.velocity else None,
        "velocity_y": info.velocity[1] if info.velocity else None,
        "age": info.age,
        "status": info.status,
    }


def alert_to_api(model: AlertModel) -> AlertInfo:
    """Convert AlertModel to AlertInfo."""
    return AlertInfo(
        id=model.id,
        type=model.type,
        level=model.level,
        message=model.message,
        device_id=model.device_id,
        track_id=model.track_id,
        timestamp=model.timestamp,
        acknowledged=model.acknowledged,
        metadata=model.alert_metadata or {},
    )


def alert_from_api(info: AlertInfo) -> Dict[str, Any]:
    """Convert AlertInfo to dict for AlertModel."""
    return {
        "id": info.id,
        "type": info.type.value if hasattr(info.type, "value") else info.type,
        "level": info.level.value if hasattr(info.level, "value") else info.level,
        "message": info.message,
        "device_id": info.device_id,
        "track_id": info.track_id,
        "timestamp": info.timestamp or datetime.utcnow(),
        "acknowledged": info.acknowledged,
        "alert_metadata": info.metadata or {},
    }


def camera_config_to_api(model: CameraConfigModel) -> CameraConfig:
    """Convert CameraConfigModel to CameraConfig."""
    return CameraConfig(
        id=model.id,
        name=model.name,
        host=model.host,
        port=model.port,
        username=model.username,
        password=model.password,
        rtsp_url=model.rtsp_url,
        enabled=model.enabled,
        fps=model.fps,
        resolution=model.resolution,
    )


def camera_config_from_api(config: CameraConfig) -> Dict[str, Any]:
    """Convert CameraConfig to dict for CameraConfigModel."""
    return {
        "id": config.id,
        "name": config.name,
        "host": config.host,
        "port": config.port,
        "username": config.username,
        "password": config.password,
        "rtsp_url": config.rtsp_url,
        "enabled": config.enabled,
        "fps": config.fps,
        "resolution": config.resolution,
    }


def ptz_config_to_api(model: PTZConfigModel) -> PTZConfig:
    """Convert PTZConfigModel to PTZConfig."""
    return PTZConfig(
        id=model.id,
        name=model.name,
        host=model.host,
        port=model.port,
        username=model.username,
        password=model.password,
        enabled=model.enabled,
        auto_tracking=model.auto_tracking,
    )


def ptz_config_from_api(config: PTZConfig) -> Dict[str, Any]:
    """Convert PTZConfig to dict for PTZConfigModel."""
    return {
        "id": config.id,
        "name": config.name,
        "host": config.host,
        "port": config.port,
        "username": config.username,
        "password": config.password,
        "enabled": config.enabled,
        "auto_tracking": config.auto_tracking,
    }


def zone_config_to_api(model: ZoneConfigModel) -> ZoneConfig:
    """Convert ZoneConfigModel to ZoneConfig."""
    coords = model.coordinates or []
    return ZoneConfig(
        id=model.id,
        name=model.name,
        coordinates=[(c[0], c[1]) for c in coords],
        enabled=model.enabled,
        alert_on_entry=model.alert_on_entry,
        alert_on_exit=model.alert_on_exit,
    )


def zone_config_from_api(config: ZoneConfig) -> Dict[str, Any]:
    """Convert ZoneConfig to dict for ZoneConfigModel."""
    return {
        "id": config.id,
        "name": config.name,
        "coordinates": [[c[0], c[1]] for c in config.coordinates],
        "enabled": config.enabled,
        "alert_on_entry": config.alert_on_entry,
        "alert_on_exit": config.alert_on_exit,
    }
