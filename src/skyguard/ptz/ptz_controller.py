"""PTZ Controller Abstract Interface and Data Models.

This module defines the unified PTZ control interface that abstracts
different PTZ protocols (ONVIF, Pelco-D, etc.).

Coordinate System:
    - Pan: [-180°, +180°], 0° = North, +90° = East, -90° = West
    - Tilt: [-90°, +90°], 0° = Horizon, +90° = Up, -90° = Down
    - Zoom: [1.0, max_zoom], optical zoom ratio

Speed units:
    - Pan speed: degrees/second
    - Tilt speed: degrees/second
    - Zoom speed: ratio/second
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from loguru import logger
from pydantic import BaseModel, Field, field_validator


class PTZMode(str, Enum):
    """PTZ operation mode."""
    IDLE = "idle"
    ABSOLUTE = "absolute"       # Move to absolute position
    CONTINUOUS = "continuous"   # Continuous move at velocity
    RELATIVE = "relative"       # Relative move from current position
    HOMING = "homing"           # Returning to home position
    PRESET = "preset"           # Moving to preset position
    TRACKING = "tracking"       # Auto-tracking mode


class PTZLimits(BaseModel):
    """PTZ movement limits."""
    pan_min: float = Field(-180.0, ge=-360.0, le=0.0, description="Minimum pan angle (degrees)")
    pan_max: float = Field(180.0, ge=0.0, le=360.0, description="Maximum pan angle (degrees)")
    tilt_min: float = Field(-90.0, ge=-90.0, le=0.0, description="Minimum tilt angle (degrees)")
    tilt_max: float = Field(90.0, ge=0.0, le=90.0, description="Maximum tilt angle (degrees)")
    zoom_min: float = Field(1.0, ge=1.0, description="Minimum zoom (1.0 = no zoom)")
    zoom_max: float = Field(20.0, ge=1.0, description="Maximum zoom ratio")
    
    # Speed limits
    pan_speed_max: float = Field(180.0, ge=1.0, description="Max pan speed (°/s)")
    tilt_speed_max: float = Field(120.0, ge=1.0, description="Max tilt speed (°/s)")
    zoom_speed_max: float = Field(2.0, ge=0.1, description="Max zoom speed (ratio/s)")
    
    @field_validator('pan_max')
    @classmethod
    def validate_pan_range(cls, v: float, info) -> float:
        if 'pan_min' in info.data and v <= info.data['pan_min']:
            raise ValueError('pan_max must be greater than pan_min')
        return v
    
    @field_validator('tilt_max')
    @classmethod
    def validate_tilt_range(cls, v: float, info) -> float:
        if 'tilt_min' in info.data and v <= info.data['tilt_min']:
            raise ValueError('tilt_max must be greater than tilt_min')
        return v
    
    @field_validator('zoom_max')
    @classmethod
    def validate_zoom_range(cls, v: float, info) -> float:
        if 'zoom_min' in info.data and v <= info.data['zoom_min']:
            raise ValueError('zoom_max must be greater than zoom_min')
        return v
    
    def clamp_pan(self, pan: float) -> float:
        """Clamp pan angle to valid range."""
        return max(self.pan_min, min(self.pan_max, pan))
    
    def clamp_tilt(self, tilt: float) -> float:
        """Clamp tilt angle to valid range."""
        return max(self.tilt_min, min(self.tilt_max, tilt))
    
    def clamp_zoom(self, zoom: float) -> float:
        """Clamp zoom ratio to valid range."""
        return max(self.zoom_min, min(self.zoom_max, zoom))
    
    def clamp_pan_speed(self, speed: float) -> float:
        """Clamp pan speed to valid range."""
        return max(-self.pan_speed_max, min(self.pan_speed_max, speed))
    
    def clamp_tilt_speed(self, speed: float) -> float:
        """Clamp tilt speed to valid range."""
        return max(-self.tilt_speed_max, min(self.tilt_speed_max, speed))


@dataclass
class PTZState:
    """Current PTZ state (position, mode, timestamp)."""
    pan: float = 0.0          # Current pan angle (degrees)
    tilt: float = 0.0         # Current tilt angle (degrees)
    zoom: float = 1.0         # Current zoom ratio
    mode: PTZMode = PTZMode.IDLE
    
    # Velocity (for continuous mode)
    pan_speed: float = 0.0    # Current pan velocity (°/s)
    tilt_speed: float = 0.0   # Current tilt velocity (°/s)
    zoom_speed: float = 0.0   # Current zoom velocity (ratio/s)
    
    # Metadata
    timestamp: float = field(default_factory=time.time)
    is_moving: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pan": self.pan,
            "tilt": self.tilt,
            "zoom": self.zoom,
            "mode": self.mode.value,
            "pan_speed": self.pan_speed,
            "tilt_speed": self.tilt_speed,
            "zoom_speed": self.zoom_speed,
            "timestamp": self.timestamp,
            "is_moving": self.is_moving,
        }
    
    def position_tuple(self) -> Tuple[float, float, float]:
        """Return position as (pan, tilt, zoom) tuple."""
        return (self.pan, self.tilt, self.zoom)


class PTZPreset(BaseModel):
    """PTZ preset position."""
    id: int = Field(..., ge=0, description="Preset ID")
    name: str = Field(..., min_length=1, description="Preset name")
    pan: float = Field(..., ge=-180.0, le=180.0, description="Pan angle")
    tilt: float = Field(..., ge=-90.0, le=90.0, description="Tilt angle")
    zoom: float = Field(1.0, ge=1.0, description="Zoom ratio")
    description: Optional[str] = Field(None, description="Preset description")


class PTZController(ABC):
    """Abstract base class for PTZ controllers.
    
    This defines the unified interface for all PTZ protocols.
    Implementations must provide concrete methods for each operation.
    
    All positions are in degrees for pan/tilt, and ratio for zoom.
    Speeds are in degrees/second or ratio/second.
    """
    
    def __init__(
        self,
        limits: Optional[PTZLimits] = None,
        home_position: Tuple[float, float, float] = (0.0, 0.0, 1.0),
    ):
        """Initialize PTZ controller.
        
        Args:
            limits: PTZ movement limits. If None, uses default limits.
            home_position: Home position (pan, tilt, zoom).
        """
        self.limits = limits or PTZLimits()
        self.home_position = home_position
        self._state = PTZState()
        self._connected = False
        self._lock = asyncio.Lock()
        self._last_command_time = 0.0
        self._command_interval = 0.05  # 50ms minimum between commands
        
        logger.info(f"PTZController initialized with limits: {self.limits}")
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to PTZ device.
        
        Returns:
            True if connection successful, False otherwise.
        
        Raises:
            ConnectionError: If connection fails.
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from PTZ device."""
        pass
    
    @abstractmethod
    async def get_state(self) -> PTZState:
        """Get current PTZ state (position, mode, etc.).
        
        Returns:
            Current PTZ state.
        """
        pass
    
    @abstractmethod
    async def move_absolute(
        self,
        pan: float,
        tilt: float,
        zoom: float,
        pan_speed: Optional[float] = None,
        tilt_speed: Optional[float] = None,
        zoom_speed: Optional[float] = None,
    ) -> bool:
        """Move to absolute position.
        
        Args:
            pan: Target pan angle (degrees).
            tilt: Target tilt angle (degrees).
            zoom: Target zoom ratio.
            pan_speed: Pan movement speed (°/s). None for max speed.
            tilt_speed: Tilt movement speed (°/s). None for max speed.
            zoom_speed: Zoom movement speed (ratio/s). None for max speed.
        
        Returns:
            True if command sent successfully.
        """
        pass
    
    @abstractmethod
    async def move_continuous(
        self,
        pan_speed: float,
        tilt_speed: float,
        zoom_speed: float = 0.0,
    ) -> bool:
        """Start continuous movement at specified velocity.
        
        Args:
            pan_speed: Pan velocity (°/s), positive=right, negative=left.
            tilt_speed: Tilt velocity (°/s), positive=up, negative=down.
            zoom_speed: Zoom velocity (ratio/s), positive=zoom_in, negative=zoom_out.
        
        Returns:
            True if command sent successfully.
        """
        pass
    
    @abstractmethod
    async def stop(self) -> bool:
        """Stop all movement.
        
        Returns:
            True if command sent successfully.
        """
        pass
    
    @abstractmethod
    async def move_to_home(self) -> bool:
        """Move to home position.
        
        Returns:
            True if command sent successfully.
        """
        pass
    
    @abstractmethod
    async def get_presets(self) -> Dict[int, PTZPreset]:
        """Get all presets.
        
        Returns:
            Dictionary of preset_id -> PTZPreset.
        """
        pass
    
    @abstractmethod
    async def move_to_preset(self, preset_id: int) -> bool:
        """Move to preset position.
        
        Args:
            preset_id: Preset ID to move to.
        
        Returns:
            True if command sent successfully.
        """
        pass
    
    @abstractmethod
    async def set_preset(self, preset_id: int, name: str) -> bool:
        """Save current position as preset.
        
        Args:
            preset_id: Preset ID (must be within device limits).
            name: Preset name.
        
        Returns:
            True if command sent successfully.
        """
        pass
    
    @abstractmethod
    async def delete_preset(self, preset_id: int) -> bool:
        """Delete preset.
        
        Args:
            preset_id: Preset ID to delete.
        
        Returns:
            True if command sent successfully.
        """
        pass
    
    # ========== High-level helper methods ==========
    
    async def move_relative(
        self,
        pan_delta: float,
        tilt_delta: float,
        zoom_delta: float = 0.0,
        pan_speed: Optional[float] = None,
        tilt_speed: Optional[float] = None,
        zoom_speed: Optional[float] = None,
    ) -> bool:
        """Move relative to current position.
        
        Args:
            pan_delta: Pan angle change (degrees).
            tilt_delta: Tilt angle change (degrees).
            zoom_delta: Zoom ratio change.
            pan_speed: Pan movement speed (°/s).
            tilt_speed: Tilt movement speed (°/s).
            zoom_speed: Zoom movement speed (ratio/s).
        
        Returns:
            True if command sent successfully.
        """
        current = await self.get_state()
        target_pan = self.limits.clamp_pan(current.pan + pan_delta)
        target_tilt = self.limits.clamp_tilt(current.tilt + tilt_delta)
        target_zoom = self.limits.clamp_zoom(current.zoom + zoom_delta)
        
        return await self.move_absolute(
            target_pan, target_tilt, target_zoom,
            pan_speed, tilt_speed, zoom_speed
        )
    
    async def wait_for_completion(self, timeout: float = 30.0, poll_interval: float = 0.1) -> bool:
        """Wait for movement to complete.
        
        Args:
            timeout: Maximum wait time in seconds.
            poll_interval: Polling interval in seconds.
        
        Returns:
            True if movement completed, False if timeout.
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            state = await self.get_state()
            if not state.is_moving:
                return True
            await asyncio.sleep(poll_interval)
        
        logger.warning(f"PTZ movement did not complete within {timeout}s")
        return False
    
    async def _rate_limit(self) -> None:
        """Rate limit commands to avoid overwhelming device."""
        elapsed = time.time() - self._last_command_time
        if elapsed < self._command_interval:
            await asyncio.sleep(self._command_interval - elapsed)
        self._last_command_time = time.time()
    
    def is_connected(self) -> bool:
        """Check if connected to device."""
        return self._connected
    
    async def __aenter__(self) -> "PTZController":
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.disconnect()


class PTZException(Exception):
    """Base exception for PTZ operations."""
    pass


class PTZConnectionError(PTZException):
    """PTZ connection error."""
    pass


class PTZCommandError(PTZException):
    """PTZ command execution error."""
    pass


class PTZTimeoutError(PTZException):
    """PTZ operation timeout."""
    pass