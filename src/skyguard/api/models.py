"""API Data Models.

Pydantic models for request/response schemas.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field


class APIResponse(BaseModel):
    """Standard API response wrapper."""
    success: bool = True
    message: str = "Success"
    data: Optional[Any] = None
    error: Optional[str] = None
    
    @classmethod
    def ok(cls, data: Any = None, message: str = "Success") -> "APIResponse":
        """Create success response."""
        return cls(success=True, message=message, data=data)
    
    @classmethod
    def fail(cls, error: str, message: str = "Failed") -> "APIResponse":
        """Create failure response."""
        return cls(success=False, message=message, error=error)


class DeviceStatus(str, Enum):
    """Device connection status."""
    ONLINE = "online"
    OFFLINE = "offline"
    CONNECTING = "connecting"
    ERROR = "error"


class DeviceType(str, Enum):
    """Device type."""
    CAMERA = "camera"
    PTZ = "ptz"
    DETECTOR = "detector"
    TRACKER = "tracker"
    STREAM = "stream"


class DeviceInfo(BaseModel):
    """Device information."""
    id: str = Field(..., description="Unique device ID")
    name: str = Field(..., description="Device display name")
    type: DeviceType = Field(..., description="Device type")
    status: DeviceStatus = Field(..., description="Connection status")
    host: str = Field(..., description="Device host/IP")
    port: int = Field(..., description="Device port")
    config: Dict[str, Any] = Field(default_factory=dict, description="Device configuration")
    last_seen: Optional[datetime] = Field(None, description="Last activity timestamp")
    error: Optional[str] = Field(None, description="Error message if any")
    
    class Config:
        from_attributes = True


class DetectionBox(BaseModel):
    """Detection bounding box."""
    x1: float = Field(..., description="Left x coordinate")
    y1: float = Field(..., description="Top y coordinate")
    x2: float = Field(..., description="Right x coordinate")
    y2: float = Field(..., description="Bottom y coordinate")
    
    @property
    def width(self) -> float:
        """Bounding box width."""
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        """Bounding box height."""
        return self.y2 - self.y1
    
    @property
    def center(self) -> Tuple[float, float]:
        """Bounding box center."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


class DetectionInfo(BaseModel):
    """Single detection information."""
    id: str = Field(..., description="Detection ID")
    class_id: int = Field(..., description="Class ID")
    class_name: str = Field(..., description="Class name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    bbox: DetectionBox = Field(..., description="Bounding box")
    timestamp: datetime = Field(..., description="Detection timestamp")


class DetectionResult(BaseModel):
    """Detection results for a frame."""
    frame_id: str = Field(..., description="Frame ID")
    timestamp: datetime = Field(..., description="Frame timestamp")
    width: int = Field(..., description="Frame width")
    height: int = Field(..., description="Frame height")
    detections: List[DetectionInfo] = Field(default_factory=list, description="List of detections")
    fps: float = Field(0.0, description="Processing FPS")


class TrackInfo(BaseModel):
    """Single track information."""
    track_id: int = Field(..., description="Track ID")
    class_id: int = Field(..., description="Class ID")
    class_name: str = Field(..., description="Class name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    bbox: DetectionBox = Field(..., description="Bounding box")
    velocity: Optional[Tuple[float, float]] = Field(None, description="Track velocity (px/s)")
    age: float = Field(0.0, description="Track age in seconds")
    status: str = Field("active", description="Track status")


class TrackingResult(BaseModel):
    """Tracking results for a frame."""
    frame_id: str = Field(..., description="Frame ID")
    timestamp: datetime = Field(..., description="Frame timestamp")
    width: int = Field(..., description="Frame width")
    height: int = Field(..., description="Frame height")
    tracks: List[TrackInfo] = Field(default_factory=list, description="List of active tracks")
    fps: float = Field(0.0, description="Processing FPS")


class AlertLevel(str, Enum):
    """Alert severity level."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertType(str, Enum):
    """Alert type."""
    DRONE_DETECTED = "drone_detected"
    DRONE_ENTERED_ZONE = "drone_entered_zone"
    DRONE_LOST = "drone_lost"
    DEVICE_OFFLINE = "device_offline"
    DEVICE_ERROR = "device_error"
    SYSTEM_ERROR = "system_error"


class AlertInfo(BaseModel):
    """Alert information."""
    id: str = Field(..., description="Alert ID")
    type: AlertType = Field(..., description="Alert type")
    level: AlertLevel = Field(..., description="Alert severity")
    message: str = Field(..., description="Alert message")
    device_id: Optional[str] = Field(None, description="Related device ID")
    track_id: Optional[int] = Field(None, description="Related track ID")
    timestamp: datetime = Field(..., description="Alert timestamp")
    acknowledged: bool = Field(False, description="Whether alert is acknowledged")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class CameraConfig(BaseModel):
    """Camera configuration."""
    id: str = Field(..., description="Camera ID")
    name: str = Field(..., description="Camera name")
    host: str = Field(..., description="Camera IP address")
    port: int = Field(554, description="RTSP port")
    username: str = Field("", description="Authentication username")
    password: str = Field("", description="Authentication password")
    rtsp_url: str = Field("", description="RTSP stream URL")
    enabled: bool = Field(True, description="Whether camera is enabled")
    fps: int = Field(30, description="Target FPS")
    resolution: str = Field("1920x1080", description="Video resolution")
    
    class Config:
        from_attributes = True


class PTZConfig(BaseModel):
    """PTZ configuration."""
    id: str = Field(..., description="PTZ ID")
    name: str = Field(..., description="PTZ name")
    host: str = Field(..., description="PTZ IP address")
    port: int = Field(80, description="ONVIF port")
    username: str = Field("admin", description="ONVIF username")
    password: str = Field("", description="ONVIF password")
    enabled: bool = Field(True, description="Whether PTZ is enabled")
    auto_tracking: bool = Field(True, description="Enable auto tracking")
    
    class Config:
        from_attributes = True


class ZoneConfig(BaseModel):
    """Geofence zone configuration."""
    id: str = Field(..., description="Zone ID")
    name: str = Field(..., description="Zone name")
    coordinates: List[Tuple[float, float]] = Field(..., description="Zone polygon coordinates")
    enabled: bool = Field(True, description="Whether zone is enabled")
    alert_on_entry: bool = Field(True, description="Alert on zone entry")
    alert_on_exit: bool = Field(False, description="Alert on zone exit")


class SystemStatus(BaseModel):
    """System status information."""
    uptime: float = Field(..., description="System uptime in seconds")
    cpu_usage: float = Field(..., ge=0.0, le=100.0, description="CPU usage percentage")
    memory_usage: float = Field(..., ge=0.0, le=100.0, description="Memory usage percentage")
    gpu_usage: Optional[float] = Field(None, ge=0.0, le=100.0, description="GPU usage percentage")
    active_streams: int = Field(..., description="Number of active video streams")
    active_tracks: int = Field(..., description="Number of active tracks")
    avg_fps: float = Field(..., description="Average processing FPS")
    models_loaded: int = Field(..., description="Number of loaded models")


class WSMessageType(str, Enum):
    """WebSocket message type."""
    DETECTION = "detection"
    TRACKING = "tracking"
    ALERT = "alert"
    STATUS = "status"
    FRAME = "frame"


class WSMessage(BaseModel):
    """WebSocket message structure."""
    type: WSMessageType = Field(..., description="Message type")
    timestamp: datetime = Field(..., description="Message timestamp")
    data: Any = Field(..., description="Message data")