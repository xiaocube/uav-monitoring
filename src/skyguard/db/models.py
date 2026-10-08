"""SQLAlchemy database models."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    ARRAY,
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all database models."""
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        nullable=False,
        index=True,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )


class DeviceModel(Base):
    """Device model."""
    
    __tablename__ = "devices"
    
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    host: Mapped[str] = mapped_column(String(256), nullable=False)
    port: Mapped[int] = mapped_column(Integer, nullable=False)
    config: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    last_seen: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    error: Mapped[Optional[str]] = mapped_column(Text)
    
    UniqueConstraint("host", "port", name="uq_devices_host_port")


class DetectionModel(Base):
    """Detection model (time-series)."""
    
    __tablename__ = "detections"
    
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    frame_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    device_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("devices.id"),
        nullable=False,
        index=True,
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        default=datetime.utcnow,
    )
    class_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    class_name: Mapped[str] = mapped_column(String(64), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_x1: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_y1: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_x2: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_y2: Mapped[float] = mapped_column(Float, nullable=False)
    frame_width: Mapped[int] = mapped_column(Integer, nullable=False)
    frame_height: Mapped[int] = mapped_column(Integer, nullable=False)
    fps: Mapped[float] = mapped_column(Float, default=0.0)
    
    device: Mapped[DeviceModel] = relationship("DeviceModel", lazy="selectin")


class TrackModel(Base):
    """Track model (time-series)."""
    
    __tablename__ = "tracks"
    
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    track_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    device_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("devices.id"),
        nullable=False,
        index=True,
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        default=datetime.utcnow,
    )
    class_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    class_name: Mapped[str] = mapped_column(String(64), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_x1: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_y1: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_x2: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_y2: Mapped[float] = mapped_column(Float, nullable=False)
    velocity_x: Mapped[Optional[float]] = mapped_column(Float)
    velocity_y: Mapped[Optional[float]] = mapped_column(Float)
    age: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[str] = mapped_column(String(32), default="active")
    
    device: Mapped[DeviceModel] = relationship("DeviceModel", lazy="selectin")


class AlertModel(Base):
    """Alert model."""
    
    __tablename__ = "alerts"
    
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    level: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    device_id: Mapped[Optional[str]] = mapped_column(
        String(64),
        ForeignKey("devices.id"),
        index=True,
    )
    track_id: Mapped[Optional[int]] = mapped_column(Integer, index=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        default=datetime.utcnow,
    )
    acknowledged: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    alert_metadata: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    
    device: Mapped[Optional[DeviceModel]] = relationship(
        "DeviceModel",
        lazy="selectin",
    )


class CameraConfigModel(Base):
    """Camera configuration model."""
    
    __tablename__ = "camera_configs"
    
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    host: Mapped[str] = mapped_column(String(256), nullable=False)
    port: Mapped[int] = mapped_column(Integer, default=554)
    username: Mapped[str] = mapped_column(String(64), default="")
    password: Mapped[str] = mapped_column(String(128), default="")
    rtsp_url: Mapped[str] = mapped_column(String(512), default="")
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    fps: Mapped[int] = mapped_column(Integer, default=30)
    resolution: Mapped[str] = mapped_column(String(32), default="1920x1080")


class PTZConfigModel(Base):
    """PTZ configuration model."""
    
    __tablename__ = "ptz_configs"
    
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    host: Mapped[str] = mapped_column(String(256), nullable=False)
    port: Mapped[int] = mapped_column(Integer, default=80)
    username: Mapped[str] = mapped_column(String(64), default="admin")
    password: Mapped[str] = mapped_column(String(128), default="")
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    auto_tracking: Mapped[bool] = mapped_column(Boolean, default=True)


class ZoneConfigModel(Base):
    """Geofence zone configuration model."""
    
    __tablename__ = "zone_configs"
    
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    coordinates: Mapped[List[List[float]]] = mapped_column(JSON, nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    alert_on_entry: Mapped[bool] = mapped_column(Boolean, default=True)
    alert_on_exit: Mapped[bool] = mapped_column(Boolean, default=False)