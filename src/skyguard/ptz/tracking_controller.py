"""PTZ Tracking Controller for Drone Tracking.

This module integrates ByteTrack output with PTZ control to automatically
track detected drones. It converts target position (bbox center) to PTZ
movement commands.

Key Features:
- Convert pixel coordinates to pan/tilt angles using camera calibration
- Smooth tracking with velocity limiting (anti-shake)
- Adaptive zoom based on target size
- Multi-target prioritization (largest/closest)
- Lost target handling with prediction

Architecture:
    TrackingPipeline (ByteTrack) -> TrackingController -> PTZController
    
    ┌─────────────────┐
    │ TrackingResult  │  (ByteTrack output)
    │  - tracks[]     │
    │  - bbox, id     │
    └────────┬────────┘
             │
    ┌────────▼────────────────────────────────────┐
    │         TrackingController                   │
    │  - Select primary target                    │
    │  - Calculate angular error                  │
    │  - Apply smoothing/filtering                │
    │  - Generate PTZ velocity commands           │
    └────────┬────────────────────────────────────┘
             │
    ┌────────▼────────┐
    │ PTZController   │  (ONVIF/Serial)
    │  - move_continuous() │
    └─────────────────┘
"""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger
from pydantic import BaseModel, Field, field_validator

from skyguard.ptz.ptz_controller import PTZController, PTZState, PTZLimits
from skyguard.vision.tracker import Track, Detection


class TrackingMode(str, Enum):
    """Tracking operation mode."""
    IDLE = "idle"                   # No target, PTZ idle
    SEARCHING = "searching"         # Searching for target
    TRACKING = "tracking"           # Actively tracking target
    LOST = "lost"                   # Target lost, attempting recovery
    MANUAL = "manual"               # Manual control mode


class TargetPriority(str, Enum):
    """Target selection priority."""
    LARGEST = "largest"         # Largest/closest to camera
    CLOSEST = "closest"         # Alias for largest
    CENTER = "center"           # Closest to frame center
    HIGHEST_CONF = "highest_conf"  # Highest detection confidence
    FIRST = "first"             # First detected target


@dataclass
class CameraCalibration:
    """Camera intrinsic parameters for coordinate conversion.
    
    Used to convert pixel coordinates to pan/tilt angles.
    """
    fx: float = 800.0       # Focal length x (pixels)
    fy: float = 800.0       # Focal length y (pixels)
    cx: float = 960.0       # Principal point x (pixels)
    cy: float = 540.0       # Principal point y (pixels)
    
    # Image dimensions
    image_width: int = 1920
    image_height: int = 1080
    
    # Field of view (degrees)
    h_fov: float = 60.0     # Horizontal FOV
    v_fov: float = 35.0     # Vertical FOV
    
    @classmethod
    def from_fov(cls, h_fov: float, v_fov: float, width: int, height: int) -> "CameraCalibration":
        """Create calibration from field of view."""
        fx = width / (2.0 * math.tan(math.radians(h_fov / 2.0)))
        fy = height / (2.0 * math.tan(math.radians(v_fov / 2.0)))
        return cls(
            fx=fx, fy=fy,
            cx=width / 2.0, cy=height / 2.0,
            image_width=width, image_height=height,
            h_fov=h_fov, v_fov=v_fov,
        )
    
    def pixel_to_angle(self, x: float, y: float) -> Tuple[float, float]:
        """Convert pixel coordinates to pan/tilt angles.
        
        Args:
            x: Pixel x coordinate.
            y: Pixel y coordinate.
        
        Returns:
            Tuple of (pan_offset, tilt_offset) in degrees.
        """
        # Offset from principal point
        dx = x - self.cx
        dy = y - self.cy
        
        # Convert to angles using FOV
        pan_offset = math.degrees(math.atan2(dx, self.fx))
        tilt_offset = math.degrees(math.atan2(dy, self.fy))
        
        return (pan_offset, tilt_offset)
    
    def angle_to_pixel(self, pan: float, tilt: float) -> Tuple[float, float]:
        """Convert pan/tilt angles to pixel coordinates."""
        x = self.cx + self.fx * math.tan(math.radians(pan))
        y = self.cy + self.fy * math.tan(math.radians(tilt))
        
        # Clamp to image bounds
        x = max(0, min(self.image_width - 1, x))
        y = max(0, min(self.image_height - 1, y))
        
        return (x, y)


class SmoothingFilter:
    """Exponential smoothing filter for PTZ control.
    
    Prevents jittery movement by smoothing velocity commands.
    """
    
    def __init__(
        self,
        alpha: float = 0.3,
        pan_threshold: float = 0.5,
        tilt_threshold: float = 0.3,
    ):
        """Initialize smoothing filter.
        
        Args:
            alpha: Smoothing factor (0-1). Lower = smoother.
            pan_threshold: Minimum pan error to trigger movement (degrees).
            tilt_threshold: Minimum tilt error to trigger movement (degrees).
        """
        self.alpha = alpha
        self.pan_threshold = pan_threshold
        self.tilt_threshold = tilt_threshold
        
        # Filter state
        self._pan_velocity = 0.0
        self._tilt_velocity = 0.0
        self._zoom_velocity = 0.0
    
    def filter(
        self,
        pan_error: float,
        tilt_error: float,
        zoom_error: float = 0.0,
    ) -> Tuple[float, float, float]:
        """Apply smoothing filter to errors.
        
        Args:
            pan_error: Pan angle error (degrees).
            tilt_error: Tilt angle error (degrees).
            zoom_error: Zoom ratio error.
        
        Returns:
            Tuple of (pan_velocity, tilt_velocity, zoom_velocity).
        """
        # Dead zone - ignore small errors
        if abs(pan_error) < self.pan_threshold:
            pan_error = 0.0
        if abs(tilt_error) < self.tilt_threshold:
            tilt_error = 0.0
        
        # Exponential smoothing
        self._pan_velocity = self.alpha * pan_error + (1 - self.alpha) * self._pan_velocity
        self._tilt_velocity = self.alpha * tilt_error + (1 - self.alpha) * self._tilt_velocity
        self._zoom_velocity = self.alpha * zoom_error + (1 - self.alpha) * self._zoom_velocity
        
        return (self._pan_velocity, self._tilt_velocity, self._zoom_velocity)
    
    def reset(self) -> None:
        """Reset filter state."""
        self._pan_velocity = 0.0
        self._tilt_velocity = 0.0
        self._zoom_velocity = 0.0


class TrackingConfig(BaseModel):
    """Tracking controller configuration."""
    
    # Target selection
    target_priority: TargetPriority = Field(
        TargetPriority.LARGEST,
        description="Target selection priority"
    )
    min_confidence: float = Field(0.5, ge=0.0, le=1.0, description="Minimum detection confidence")
    
    # Tracking parameters
    tracking_gain_pan: float = Field(2.0, ge=0.1, description="Pan tracking gain (velocity = gain * error)")
    tracking_gain_tilt: float = Field(2.0, ge=0.1, description="Tilt tracking gain")
    tracking_gain_zoom: float = Field(0.5, ge=0.1, description="Zoom tracking gain")
    
    # Velocity limits (°/s)
    max_pan_speed: float = Field(30.0, ge=1.0, description="Maximum pan tracking speed")
    max_tilt_speed: float = Field(20.0, ge=1.0, description="Maximum tilt tracking speed")
    max_zoom_speed: float = Field(0.5, ge=0.1, description="Maximum zoom speed")
    
    # Smoothing
    smoothing_alpha: float = Field(0.3, ge=0.0, le=1.0, description="Smoothing factor")
    pan_dead_zone: float = Field(0.5, ge=0.0, description="Pan dead zone (degrees)")
    tilt_dead_zone: float = Field(0.3, ge=0.0, description="Tilt dead zone (degrees)")
    
    # Lost target handling
    lost_timeout: float = Field(3.0, ge=0.5, description="Seconds before declaring target lost")
    recovery_timeout: float = Field(5.0, ge=1.0, description="Seconds before stopping search")
    
    # Adaptive zoom
    adaptive_zoom: bool = Field(True, description="Enable adaptive zoom")
    target_size_min: float = Field(0.02, ge=0.01, le=0.5, description="Minimum target size (fraction of image)")
    target_size_max: float = Field(0.15, ge=0.05, le=0.5, description="Maximum target size")
    zoom_range: Tuple[float, float] = Field((1.0, 10.0), description="Zoom range")
    
    @field_validator('zoom_range')
    @classmethod
    def validate_zoom_range(cls, v: Tuple[float, float]) -> Tuple[float, float]:
        if len(v) != 2 or v[0] >= v[1]:
            raise ValueError('zoom_range must be (min, max) with min < max')
        return v


@dataclass
class TrackedTarget:
    """Information about currently tracked target."""
    track_id: int
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[float, float]      # Center in pixels
    confidence: float
    class_id: int
    class_name: str
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    
    # Calculated values
    angular_error: Tuple[float, float] = (0.0, 0.0)  # pan_error, tilt_error
    size_ratio: float = 0.0  # Fraction of image area
    
    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]
    
    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]
    
    @property
    def area(self) -> int:
        return self.width * self.height
    
    @property
    def age(self) -> float:
        return time.time() - self.first_seen


class TrackingController:
    """PTZ Tracking Controller for automatic target tracking.
    
    This controller integrates with ByteTrack output to automatically
    control PTZ camera movement.
    
    Example:
        # Initialize
        ptz = ONVIFClient(host="192.168.1.100", username="admin", password="pass")
        await ptz.connect()
        
        tracking = TrackingController(ptz, calibration)
        
        # In tracking loop
        result = tracking_pipeline.process_frame(frame)
        cmd = tracking.update(result.tracks, frame.shape[:2])
        
        if cmd:
            await ptz.move_continuous(cmd.pan_speed, cmd.tilt_speed)
    """
    
    def __init__(
        self,
        ptz: PTZController,
        calibration: Optional[CameraCalibration] = None,
        config: Optional[TrackingConfig] = None,
    ):
        """Initialize tracking controller.
        
        Args:
            ptz: PTZ controller instance.
            calibration: Camera calibration parameters.
            config: Tracking configuration.
        """
        self.ptz = ptz
        self.calibration = calibration or CameraCalibration()
        self.config = config or TrackingConfig()
        
        # State
        self.mode = TrackingMode.IDLE
        self.current_target: Optional[TrackedTarget] = None
        self.target_history: Dict[int, TrackedTarget] = {}
        
        # Filters
        self.smoothing_filter = SmoothingFilter(
            alpha=self.config.smoothing_alpha,
            pan_threshold=self.config.pan_dead_zone,
            tilt_threshold=self.config.tilt_dead_zone,
        )
        
        # Timing
        self.last_update_time = 0.0
        self.last_track_time = 0.0
        
        logger.info(f"TrackingController initialized with config: {self.config}")
    
    def select_target(
        self,
        tracks: List[Track],
        image_size: Tuple[int, int],
    ) -> Optional[TrackedTarget]:
        """Select primary target from multiple tracks.
        
        Args:
            tracks: List of tracks from ByteTrack.
            image_size: Image dimensions (height, width).
        
        Returns:
            Selected target or None.
        """
        if not tracks:
            return None
        
        height, width = image_size
        candidates = []
        
        for track in tracks:
            # Filter by confidence
            if track.confidence < self.config.min_confidence:
                continue
            
            # Calculate bbox (Track has bbox field directly)
            bbox = track.bbox
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            
            # Create target
            target = TrackedTarget(
                track_id=track.track_id,
                bbox=bbox,
                center=(center_x, center_y),
                confidence=track.confidence,
                class_id=track.class_id,
                class_name=track.class_name,
                size_ratio=((x2 - x1) * (y2 - y1)) / (width * height),
            )
            candidates.append(target)
        
        if not candidates:
            return None
        
        # Select based on priority
        if self.config.target_priority == TargetPriority.LARGEST:
            return max(candidates, key=lambda t: t.area)
        elif self.config.target_priority == TargetPriority.CENTER:
            center_x, center_y = width / 2, height / 2
            return min(candidates, key=lambda t: 
                (t.center[0] - center_x)**2 + (t.center[1] - center_y)**2)
        elif self.config.target_priority == TargetPriority.HIGHEST_CONF:
            return max(candidates, key=lambda t: t.confidence)
        else:  # FIRST
            return candidates[0]
    
    def calculate_angular_error(
        self,
        target: TrackedTarget,
        image_size: Tuple[int, int],
    ) -> Tuple[float, float]:
        """Calculate angular error from target center to image center.
        
        Args:
            target: Tracked target.
            image_size: Image dimensions (height, width).
        
        Returns:
            Tuple of (pan_error, tilt_error) in degrees.
        """
        # Target center in pixels
        tx, ty = target.center
        
        # Image center
        height, width = image_size
        cx, cy = width / 2, height / 2
        
        # Pixel offset
        px_offset = tx - cx
        py_offset = ty - cy
        
        # Convert to angles
        pan_error, tilt_error = self.calibration.pixel_to_angle(tx, ty)
        
        # Error is negative of offset (we want to move camera to center target)
        # If target is at right of center, we need to pan right (positive)
        # But pixel_to_angle returns angle from optical center
        # So error = -angle (we need to move opposite to error)
        
        return (pan_error, -tilt_error)
    
    def calculate_adaptive_zoom(self, target: TrackedTarget) -> float:
        """Calculate adaptive zoom based on target size.
        
        Args:
            target: Tracked target.
        
        Returns:
            Desired zoom ratio.
        """
        if not self.config.adaptive_zoom:
            return self.config.zoom_range[0]
        
        size = target.size_ratio
        
        # Desired size in frame
        desired_size = 0.05  # 5% of frame
        
        # Calculate zoom needed
        if size > 0:
            zoom = desired_size / size
        else:
            zoom = self.config.zoom_range[0]
        
        # Clamp to range
        zoom = max(self.config.zoom_range[0], min(self.config.zoom_range[1], zoom))
        
        return zoom
    
    def update(
        self,
        tracks: List[Track],
        image_size: Tuple[int, int],
    ) -> Optional[PTZCommand]:
        """Update tracking with new detections.
        
        This should be called for each frame with ByteTrack output.
        
        Args:
            tracks: List of tracks from ByteTrack.
            image_size: Image dimensions (height, width).
        
        Returns:
            PTZ command to execute, or None if no command needed.
        """
        now = time.time()
        self.last_update_time = now
        
        # Select target
        target = self.select_target(tracks, image_size)
        
        if target:
            # Update or set target
            if self.current_target and self.current_target.track_id == target.track_id:
                # Update existing target
                self.current_target.bbox = target.bbox
                self.current_target.center = target.center
                self.current_target.confidence = target.confidence
                self.current_target.size_ratio = target.size_ratio
                self.current_target.last_seen = now
            else:
                # New target
                self.current_target = target
                self.mode = TrackingMode.TRACKING
                logger.info(f"Tracking new target: id={target.track_id}, class={target.class_name}")
            
            self.last_track_time = now
            
            # Calculate errors
            pan_error, tilt_error = self.calculate_angular_error(target, image_size)
            target.angular_error = (pan_error, tilt_error)
            
            # Adaptive zoom
            desired_zoom = self.calculate_adaptive_zoom(target)
            
            # Apply smoothing
            pan_velocity, tilt_velocity, zoom_velocity = self.smoothing_filter.filter(
                pan_error, tilt_error, 0.0  # Zoom handled separately
            )
            
            # Apply tracking gains
            pan_speed = pan_velocity * self.config.tracking_gain_pan
            tilt_speed = tilt_velocity * self.config.tracking_gain_tilt
            zoom_speed = (desired_zoom - self.ptz._state.zoom) * self.config.tracking_gain_zoom
            
            # Clamp to velocity limits
            pan_speed = max(-self.config.max_pan_speed, min(self.config.max_pan_speed, pan_speed))
            tilt_speed = max(-self.config.max_tilt_speed, min(self.config.max_tilt_speed, tilt_speed))
            zoom_speed = max(-self.config.max_zoom_speed, min(self.config.max_zoom_speed, zoom_speed))
            
            return PTZCommand(
                pan_speed=pan_speed,
                tilt_speed=tilt_speed,
                zoom_speed=zoom_speed,
                mode=self.mode,
                target_id=target.track_id,
            )
        
        else:
            # No target detected
            time_since_last_track = now - self.last_track_time
            
            if time_since_last_track < self.config.lost_timeout:
                # Still in lost recovery window
                self.mode = TrackingMode.LOST
                logger.debug(f"Target lost, waiting for recovery ({time_since_last_track:.1f}s)")
                
                # Continue with last known velocity (prediction)
                # Or stop moving
                self.smoothing_filter.reset()
                return PTZCommand(
                    pan_speed=0.0,
                    tilt_speed=0.0,
                    zoom_speed=0.0,
                    mode=self.mode,
                    target_id=self.current_target.track_id if self.current_target else None,
                )
            
            elif time_since_last_track < self.config.recovery_timeout:
                # Search mode - slowly pan to find target
                self.mode = TrackingMode.SEARCHING
                logger.debug("Searching for target")
                
                # TODO: Implement search pattern
                return PTZCommand(
                    pan_speed=0.0,
                    tilt_speed=0.0,
                    zoom_speed=0.0,
                    mode=self.mode,
                    target_id=None,
                )
            
            else:
                # Give up, return to idle
                if self.mode != TrackingMode.IDLE:
                    logger.info("Target lost timeout, returning to idle")
                    self.mode = TrackingMode.IDLE
                    self.current_target = None
                    self.smoothing_filter.reset()
                
                return PTZCommand(
                    pan_speed=0.0,
                    tilt_speed=0.0,
                    zoom_speed=0.0,
                    mode=self.mode,
                    target_id=None,
                )
    
    def reset(self) -> None:
        """Reset tracking state."""
        self.mode = TrackingMode.IDLE
        self.current_target = None
        self.target_history.clear()
        self.smoothing_filter.reset()
        logger.info("Tracking controller reset")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current tracking status."""
        return {
            "mode": self.mode.value,
            "target_id": self.current_target.track_id if self.current_target else None,
            "target_confidence": self.current_target.confidence if self.current_target else 0.0,
            "angular_error": self.current_target.angular_error if self.current_target else (0.0, 0.0),
            "last_update": self.last_update_time,
            "last_track": self.last_track_time,
        }


@dataclass
class PTZCommand:
    """PTZ movement command."""
    pan_speed: float      # °/s
    tilt_speed: float     # °/s
    zoom_speed: float     # ratio/s
    mode: TrackingMode
    target_id: Optional[int] = None
    
    def __str__(self) -> str:
        return f"PTZCommand(pan={self.pan_speed:.1f}°/s, tilt={self.tilt_speed:.1f}°/s, mode={self.mode.value})"
    
    def is_zero(self) -> bool:
        """Check if command is zero velocity."""
        return abs(self.pan_speed) < 0.1 and abs(self.tilt_speed) < 0.1 and abs(self.zoom_speed) < 0.01
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pan_speed": self.pan_speed,
            "tilt_speed": self.tilt_speed,
            "zoom_speed": self.zoom_speed,
            "mode": self.mode.value,
            "target_id": self.target_id,
        }


# ========== Integration with TrackingPipeline ==========

async def run_tracking_loop(
    tracking_controller: TrackingController,
    tracking_pipeline,  # TrackingPipeline from vision.tracking_pipeline
    ptz_controller: PTZController,
    video_source,
    fps: float = 30.0,
) -> None:
    """Run tracking loop.
    
    This is a helper function that integrates all components:
    1. Read frame from video source
    2. Run detection + tracking (TrackingPipeline)
    3. Update PTZ position (TrackingController)
    4. Send commands to PTZ
    
    Args:
        tracking_controller: Tracking controller.
        tracking_pipeline: Tracking pipeline (detection + ByteTrack).
        ptz_controller: PTZ controller.
        video_source: Video source (e.g., VideoStream).
        fps: Target FPS.
    """
    logger.info("Starting tracking loop")
    
    frame_interval = 1.0 / fps
    last_frame_time = 0.0
    
    try:
        while True:
            # Read frame
            frame = await video_source.read()
            if frame is None:
                logger.warning("No frame from video source")
                await asyncio.sleep(0.1)
                continue
            
            # Run tracking
            result = tracking_pipeline.process_frame(frame)
            
            # Update PTZ
            command = tracking_controller.update(
                result.tracks,
                (frame.shape[0], frame.shape[1])
            )
            
            if command and not command.is_zero():
                # Send command to PTZ
                await ptz_controller.move_continuous(
                    command.pan_speed,
                    command.tilt_speed,
                    command.zoom_speed,
                )
            
            # Maintain frame rate
            elapsed = time.time() - last_frame_time
            if elapsed < frame_interval:
                await asyncio.sleep(frame_interval - elapsed)
            last_frame_time = time.time()
            
    except asyncio.CancelledError:
        logger.info("Tracking loop cancelled")
    except Exception as e:
        logger.error(f"Tracking loop error: {e}")
        raise
    finally:
        # Stop PTZ
        await ptz_controller.stop()
        logger.info("Tracking loop stopped")