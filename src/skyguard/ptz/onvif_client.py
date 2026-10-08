"""ONVIF PTZ Client Implementation.

ONVIF (Open Network Video Interface Forum) is an open industry standard
for IP-based video surveillance equipment. This module provides a Pythonic
interface to ONVIF PTZ services.

ONVIF uses SOAP/WSDL web services. The main services:
- DeviceService: Device management and discovery
- MediaService: Media configuration
- PTZService: Pan-Tilt-Zoom control
- ImagingService: Image settings

Coordinate mapping:
    ONVIF uses normalized space [0.0, 1.0] for positions.
    This module converts between degrees and normalized values.

Requirements:
    pip install onvif-zeep[async]
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from loguru import logger
from pydantic import BaseModel, Field

from skyguard.ptz.ptz_controller import (
    PTZController,
    PTZState,
    PTZLimits,
    PTZPreset,
    PTZMode,
    PTZConnectionError,
    PTZCommandError,
)


@dataclass
class ONVIFCapabilities:
    """ONVIF device capabilities."""
    ptz: bool = False
    presets: bool = False
    home_position: bool = False
    continuous_move: bool = False
    relative_move: bool = False
    absolute_move: bool = False
    zoom: bool = False
    pan_tilt: bool = False


class ONVIFConfig(BaseModel):
    """ONVIF connection configuration."""
    host: str = Field(..., description="Camera IP address")
    port: int = Field(80, ge=1, le=65535, description="ONVIF port")
    username: str = Field("admin", description="Username")
    password: str = Field("", min_length=0, description="Password")
    
    # Optional settings
    wsdl_path: Optional[str] = Field(None, description="Custom WSDL directory path")
    timeout: float = Field(10.0, ge=1.0, description="Connection timeout in seconds")
    retry_attempts: int = Field(3, ge=0, description="Connection retry attempts")
    
    @property
    def endpoint(self) -> str:
        """Get ONVIF endpoint URL."""
        return f"http://{self.host}:{self.port}"


class ONVIFClient(PTZController):
    """ONVIF PTZ Controller implementation.
    
    This client communicates with ONVIF-compliant IP cameras to control
    pan-tilt-zoom functions.
    
    Example:
        async with ONVIFClient(
            host="192.168.1.100",
            username="admin",
            password="password"
        ) as ptz:
            # Move to absolute position
            await ptz.move_absolute(pan=45.0, tilt=10.0, zoom=2.0)
            
            # Get current position
            state = await ptz.get_state()
            print(f"Position: pan={state.pan}, tilt={state.tilt}, zoom={state.zoom}")
            
            # Stop movement
            await ptz.stop()
    """
    
    def __init__(
        self,
        host: str,
        port: int = 80,
        username: str = "admin",
        password: str = "",
        limits: Optional[PTZLimits] = None,
        home_position: Tuple[float, float, float] = (0.0, 0.0, 1.0),
        wsdl_path: Optional[str] = None,
    ):
        """Initialize ONVIF client.
        
        Args:
            host: Camera IP address.
            port: ONVIF service port (default: 80).
            username: Authentication username.
            password: Authentication password.
            limits: PTZ movement limits.
            home_position: Home position (pan, tilt, zoom).
            wsdl_path: Custom WSDL directory path.
        """
        super().__init__(limits=limits, home_position=home_position)
        
        self.config = ONVIFConfig(
            host=host,
            port=port,
            username=username,
            password=password,
            wsdl_path=wsdl_path,
        )
        
        # ONVIF service clients (initialized on connect)
        self._device = None
        self._media = None
        self._ptz = None
        self._imaging = None
        
        # ONVIF-specific data
        self._media_profile = None
        self._ptz_config = None
        self._capabilities = ONVIFCapabilities()
        self._presets: Dict[int, PTZPreset] = {}
        
        # Coordinate mapping (degrees <-> normalized)
        self._pan_range = (-180.0, 180.0)
        self._tilt_range = (-90.0, 90.0)
        self._zoom_range = (1.0, 20.0)
        
        logger.info(f"ONVIFClient created for {host}:{port}")
    
    async def connect(self) -> bool:
        """Connect to ONVIF device.
        
        Returns:
            True if connection successful.
        
        Raises:
            PTZConnectionError: If connection fails.
        """
        if self._connected:
            logger.warning("Already connected to ONVIF device")
            return True
        
        try:
            # Try to import ONVIF library
            try:
                from onvif import ONVIFCamera
            except ImportError:
                logger.warning("onvif-zeep not installed, using mock implementation")
                return await self._connect_mock()
            
            # Create ONVIF camera client
            self._device = ONVIFCamera(
                self.config.host,
                self.config.port,
                self.config.username,
                self.config.password,
                wsdl_dir=self.config.wsdl_path,
            )
            
            # Start services
            await self._device.start()
            
            # Get services
            self._media = self._device.create_media_service()
            self._ptz = self._device.create_ptz_service()
            
            # Get media profile
            profiles = await self._media.GetProfiles()
            if not profiles:
                raise PTZConnectionError("No media profiles found")
            
            self._media_profile = profiles[0]  # Use first profile
            logger.info(f"Using media profile: {self._media_profile.Name}")
            
            # Get PTZ configuration
            configs = await self._ptz.GetConfigurations()
            if configs:
                self._ptz_config = configs[0]
                self._parse_ptz_config(self._ptz_config)
            
            # Query capabilities
            await self._query_capabilities()
            
            self._connected = True
            logger.success(f"Connected to ONVIF device at {self.config.host}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to ONVIF device: {e}")
            raise PTZConnectionError(f"Connection failed: {e}")
    
    async def disconnect(self) -> None:
        """Disconnect from ONVIF device."""
        if not self._connected:
            return
        
        try:
            if self._device:
                await self._device.stop()
            
            self._device = None
            self._media = None
            self._ptz = None
            self._imaging = None
            self._connected = False
            
            logger.info("Disconnected from ONVIF device")
            
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")
    
    async def get_state(self) -> PTZState:
        """Get current PTZ state."""
        if not self._connected:
            return self._state
        
        try:
            # Query current position from ONVIF
            if self._ptz and self._media_profile:
                status = await self._ptz.GetStatus({
                    'ProfileToken': self._media_profile.token
                })
                
                # Parse position
                if status.Position:
                    pan, tilt, zoom = self._from_normalized(status.Position)
                    self._state.pan = pan
                    self._state.tilt = tilt
                    self._state.zoom = zoom
                
                # Check if moving
                self._state.is_moving = status.MoveStatus is not None
                
                self._state.timestamp = time.time()
            
            return self._state
            
        except Exception as e:
            logger.warning(f"Failed to get PTZ state: {e}")
            return self._state
    
    async def move_absolute(
        self,
        pan: float,
        tilt: float,
        zoom: float,
        pan_speed: Optional[float] = None,
        tilt_speed: Optional[float] = None,
        zoom_speed: Optional[float] = None,
    ) -> bool:
        """Move to absolute position."""
        if not self._connected:
            logger.warning("Not connected to ONVIF device")
            return False
        
        await self._rate_limit()
        
        try:
            # Clamp values to limits
            pan = self.limits.clamp_pan(pan)
            tilt = self.limits.clamp_tilt(tilt)
            zoom = self.limits.clamp_zoom(zoom)
            
            # Convert to ONVIF normalized space
            position = self._to_normalized(pan, tilt, zoom)
            
            # Build speed parameters
            speed = self._build_speed_vector(pan_speed, tilt_speed, zoom_speed)
            
            # Send AbsoluteMove command
            request = {
                'ProfileToken': self._media_profile.token,
                'Position': position,
                'Speed': speed,
            }
            
            await self._ptz.AbsoluteMove(request)
            
            # Update internal state
            self._state.pan = pan
            self._state.tilt = tilt
            self._state.zoom = zoom
            self._state.mode = PTZMode.ABSOLUTE
            self._state.is_moving = True
            
            logger.debug(f"AbsoluteMove: pan={pan:.1f}°, tilt={tilt:.1f}°, zoom={zoom:.1f}x")
            
            return True
            
        except Exception as e:
            logger.error(f"AbsoluteMove failed: {e}")
            return False
    
    async def move_continuous(
        self,
        pan_speed: float,
        tilt_speed: float,
        zoom_speed: float = 0.0,
    ) -> bool:
        """Start continuous movement."""
        if not self._connected:
            logger.warning("Not connected to ONVIF device")
            return False
        
        if not self._capabilities.continuous_move:
            logger.error("Device does not support ContinuousMove")
            return False
        
        await self._rate_limit()
        
        try:
            # Clamp speeds
            pan_speed = self.limits.clamp_pan_speed(pan_speed)
            tilt_speed = self.limits.clamp_tilt_speed(tilt_speed)
            
            # Convert to normalized velocity (-1.0 to 1.0)
            velocity = self._to_normalized_velocity(pan_speed, tilt_speed, zoom_speed)
            
            # Send ContinuousMove command
            request = {
                'ProfileToken': self._media_profile.token,
                'Velocity': velocity,
            }
            
            await self._ptz.ContinuousMove(request)
            
            # Update internal state
            self._state.pan_speed = pan_speed
            self._state.tilt_speed = tilt_speed
            self._state.zoom_speed = zoom_speed
            self._state.mode = PTZMode.CONTINUOUS
            self._state.is_moving = True
            
            logger.debug(f"ContinuousMove: pan_speed={pan_speed:.1f}°/s, tilt_speed={tilt_speed:.1f}°/s")
            
            return True
            
        except Exception as e:
            logger.error(f"ContinuousMove failed: {e}")
            return False
    
    async def stop(self) -> bool:
        """Stop all movement."""
        if not self._connected:
            return False
        
        try:
            request = {
                'ProfileToken': self._media_profile.token,
            }
            
            await self._ptz.Stop(request)
            
            # Update internal state
            self._state.pan_speed = 0.0
            self._state.tilt_speed = 0.0
            self._state.zoom_speed = 0.0
            self._state.mode = PTZMode.IDLE
            self._state.is_moving = False
            
            logger.debug("PTZ movement stopped")
            
            return True
            
        except Exception as e:
            logger.error(f"Stop failed: {e}")
            return False
    
    async def move_to_home(self) -> bool:
        """Move to home position."""
        if not self._connected:
            return False
        
        if not self._capabilities.home_position:
            # Use absolute move instead
            pan, tilt, zoom = self.home_position
            return await self.move_absolute(pan, tilt, zoom)
        
        try:
            request = {
                'ProfileToken': self._media_profile.token,
            }
            
            await self._ptz.GotoHomePosition(request)
            
            self._state.mode = PTZMode.HOMING
            self._state.is_moving = True
            
            logger.info("Moving to home position")
            
            return True
            
        except Exception as e:
            logger.error(f"GotoHomePosition failed: {e}")
            return False
    
    async def get_presets(self) -> Dict[int, PTZPreset]:
        """Get all presets."""
        if not self._connected or not self._capabilities.presets:
            return {}
        
        try:
            presets = await self._ptz.GetPresets({
                'ProfileToken': self._media_profile.token
            })
            
            result = {}
            for preset in presets:
                preset_id = preset.token
                pan, tilt, zoom = self._from_normalized(preset.Position)
                
                result[preset_id] = PTZPreset(
                    id=preset_id,
                    name=preset.Name,
                    pan=pan,
                    tilt=tilt,
                    zoom=zoom,
                )
            
            self._presets = result
            return result
            
        except Exception as e:
            logger.error(f"GetPresets failed: {e}")
            return {}
    
    async def move_to_preset(self, preset_id: int) -> bool:
        """Move to preset position."""
        if not self._connected or not self._capabilities.presets:
            return False
        
        try:
            request = {
                'ProfileToken': self._media_profile.token,
                'PresetToken': preset_id,
            }
            
            await self._ptz.GotoPreset(request)
            
            self._state.mode = PTZMode.PRESET
            self._state.is_moving = True
            
            logger.info(f"Moving to preset {preset_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"GotoPreset failed: {e}")
            return False
    
    async def set_preset(self, preset_id: int, name: str) -> bool:
        """Save current position as preset."""
        if not self._connected or not self._capabilities.presets:
            return False
        
        try:
            request = {
                'ProfileToken': self._media_profile.token,
                'PresetToken': str(preset_id),
                'PresetName': name,
            }
            
            await self._ptz.SetPreset(request)
            
            logger.info(f"Set preset {preset_id}: {name}")
            
            return True
            
        except Exception as e:
            logger.error(f"SetPreset failed: {e}")
            return False
    
    async def delete_preset(self, preset_id: int) -> bool:
        """Delete preset."""
        if not self._connected or not self._capabilities.presets:
            return False
        
        try:
            request = {
                'ProfileToken': self._media_profile.token,
                'PresetToken': preset_id,
            }
            
            await self._ptz.RemovePreset(request)
            
            logger.info(f"Deleted preset {preset_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"RemovePreset failed: {e}")
            return False
    
    # ========== Internal methods ==========
    
    async def _query_capabilities(self) -> None:
        """Query device capabilities."""
        try:
            if self._ptz:
                # Get PTZ service capabilities
                caps = await self._ptz.GetServiceCapabilities()
                
                self._capabilities.ptz = True
                self._capabilities.presets = getattr(caps, 'SupportedPreset', False)
                self._capabilities.home_position = getattr(caps, 'HomePosition', False)
                self._capabilities.continuous_move = getattr(caps, 'ContinuousMove', True)
                self._capabilities.relative_move = getattr(caps, 'RelativeMove', True)
                self._capabilities.absolute_move = getattr(caps, 'AbsoluteMove', True)
                
            logger.info(f"ONVIF capabilities: {self._capabilities}")
            
        except Exception as e:
            logger.warning(f"Failed to query capabilities: {e}")
            # Assume basic capabilities
            self._capabilities.ptz = True
            self._capabilities.absolute_move = True
            self._capabilities.continuous_move = True
    
    def _parse_ptz_config(self, config: Any) -> None:
        """Parse PTZ configuration to get ranges."""
        try:
            if hasattr(config, 'PanTiltLimits'):
                limits = config.PanTiltLimits
                if limits:
                    self._pan_range = (limits.Range.XMin, limits.Range.XMax)
                    self._tilt_range = (limits.Range.YMin, limits.Range.YMax)
            
            if hasattr(config, 'ZoomLimits'):
                limits = config.ZoomLimits
                if limits:
                    self._zoom_range = (limits.Range.XMin, limits.Range.XMax)
            
            logger.debug(f"PTZ ranges: pan={self._pan_range}, tilt={self._tilt_range}, zoom={self._zoom_range}")
            
        except Exception as e:
            logger.warning(f"Failed to parse PTZ config: {e}")
    
    def _to_normalized(self, pan: float, tilt: float, zoom: float) -> Dict[str, float]:
        """Convert degrees/ratio to ONVIF normalized space [0, 1]."""
        # Pan: [-180, 180] -> [0, 1]
        norm_pan = (pan - self._pan_range[0]) / (self._pan_range[1] - self._pan_range[0])
        
        # Tilt: [-90, 90] -> [0, 1]
        norm_tilt = (tilt - self._tilt_range[0]) / (self._tilt_range[1] - self._tilt_range[0])
        
        # Zoom: [1, max] -> [0, 1]
        norm_zoom = (zoom - self._zoom_range[0]) / (self._zoom_range[1] - self._zoom_range[0])
        
        return {
            'PanTilt': {'x': norm_pan, 'y': norm_tilt},
            'Zoom': {'x': norm_zoom},
        }
    
    def _from_normalized(self, position: Any) -> Tuple[float, float, float]:
        """Convert ONVIF normalized space [0, 1] to degrees/ratio."""
        try:
            # Extract normalized values
            norm_pan = getattr(position.PanTilt, 'x', 0.5)
            norm_tilt = getattr(position.PanTilt, 'y', 0.5)
            norm_zoom = getattr(position.Zoom, 'x', 0.0)
            
            # Convert to real values
            pan = self._pan_range[0] + norm_pan * (self._pan_range[1] - self._pan_range[0])
            tilt = self._tilt_range[0] + norm_tilt * (self._tilt_range[1] - self._tilt_range[0])
            zoom = self._zoom_range[0] + norm_zoom * (self._zoom_range[1] - self._zoom_range[0])
            
            return (pan, tilt, zoom)
            
        except Exception as e:
            logger.warning(f"Failed to parse position: {e}")
            return (0.0, 0.0, 1.0)
    
    def _to_normalized_velocity(self, pan_speed: float, tilt_speed: float, zoom_speed: float) -> Dict[str, float]:
        """Convert speed (°/s) to ONVIF normalized velocity (-1 to 1)."""
        # Scale to normalized range
        norm_pan = pan_speed / self.limits.pan_speed_max
        norm_tilt = tilt_speed / self.limits.tilt_speed_max
        norm_zoom = zoom_speed / self.limits.zoom_speed_max
        
        # Clamp to [-1, 1]
        norm_pan = max(-1.0, min(1.0, norm_pan))
        norm_tilt = max(-1.0, min(1.0, norm_tilt))
        norm_zoom = max(-1.0, min(1.0, norm_zoom))
        
        return {
            'PanTilt': {'x': norm_pan, 'y': norm_tilt},
            'Zoom': {'x': norm_zoom},
        }
    
    def _build_speed_vector(
        self,
        pan_speed: Optional[float],
        tilt_speed: Optional[float],
        zoom_speed: Optional[float],
    ) -> Dict[str, float]:
        """Build speed vector for absolute move."""
        # Use max speed if not specified
        if pan_speed is None:
            pan_speed = self.limits.pan_speed_max
        if tilt_speed is None:
            tilt_speed = self.limits.tilt_speed_max
        if zoom_speed is None:
            zoom_speed = self.limits.zoom_speed_max
        
        return self._to_normalized_velocity(pan_speed, tilt_speed, zoom_speed)
    
    # ========== Mock implementation for testing ==========
    
    async def _connect_mock(self) -> bool:
        """Mock connection for testing without real camera."""
        logger.warning("Using mock ONVIF connection")
        
        self._capabilities = ONVIFCapabilities(
            ptz=True,
            presets=True,
            home_position=True,
            continuous_move=True,
            relative_move=True,
            absolute_move=True,
            zoom=True,
            pan_tilt=True,
        )
        
        self._connected = True
        self._state = PTZState()
        
        logger.success(f"Mock ONVIF connection established to {self.config.host}")
        
        return True


# ========== Mock PTZ for testing ==========

class MockONVIFClient(ONVIFClient):
    """Mock ONVIF client for testing without real hardware."""
    
    def __init__(self, **kwargs):
        """Initialize mock client."""
        super().__init__(**kwargs)
        self._mock_position = [0.0, 0.0, 1.0]  # pan, tilt, zoom
        self._mock_velocity = [0.0, 0.0, 0.0]
        self._mock_presets: Dict[int, PTZPreset] = {}
        self._mock_moving = False
    
    async def connect(self) -> bool:
        """Mock connection."""
        await asyncio.sleep(0.1)  # Simulate connection delay
        return await self._connect_mock()
    
    async def disconnect(self) -> None:
        """Mock disconnect."""
        self._connected = False
    
    async def get_state(self) -> PTZState:
        """Get mock state."""
        # Simulate movement
        if self._mock_moving and any(self._mock_velocity):
            dt = time.time() - self._state.timestamp
            self._mock_position[0] += self._mock_velocity[0] * dt
            self._mock_position[1] += self._mock_velocity[1] * dt
            self._mock_position[2] += self._mock_velocity[2] * dt
            
            # Clamp to limits
            self._mock_position[0] = self.limits.clamp_pan(self._mock_position[0])
            self._mock_position[1] = self.limits.clamp_tilt(self._mock_position[1])
            self._mock_position[2] = self.limits.clamp_zoom(self._mock_position[2])
        
        self._state.pan = self._mock_position[0]
        self._state.tilt = self._mock_position[1]
        self._state.zoom = self._mock_position[2]
        self._state.pan_speed = self._mock_velocity[0]
        self._state.tilt_speed = self._mock_velocity[1]
        self._state.zoom_speed = self._mock_velocity[2]
        self._state.is_moving = self._mock_moving
        self._state.timestamp = time.time()
        
        return self._state
    
    async def move_to_home(self) -> bool:
        """Mock move to home."""
        pan, tilt, zoom = self.home_position
        self._mock_position = [pan, tilt, zoom]
        self._mock_velocity = [0.0, 0.0, 0.0]
        self._mock_moving = False
        return True
    
    async def move_absolute(self, pan: float, tilt: float, zoom: float, **kwargs) -> bool:
        """Mock absolute move."""
        self._mock_position = [pan, tilt, zoom]
        self._mock_velocity = [0.0, 0.0, 0.0]
        self._mock_moving = False
        return True
    
    async def move_continuous(self, pan_speed: float, tilt_speed: float, zoom_speed: float = 0.0) -> bool:
        """Mock continuous move."""
        self._mock_velocity = [pan_speed, tilt_speed, zoom_speed]
        self._mock_moving = True
        return True
    
    async def stop(self) -> bool:
        """Mock stop."""
        self._mock_velocity = [0.0, 0.0, 0.0]
        self._mock_moving = False
        return True
    
    async def get_presets(self) -> Dict[int, PTZPreset]:
        """Get mock presets."""
        return self._mock_presets
    
    async def set_preset(self, preset_id: int, name: str) -> bool:
        """Set mock preset."""
        self._mock_presets[preset_id] = PTZPreset(
            id=preset_id,
            name=name,
            pan=self._mock_position[0],
            tilt=self._mock_position[1],
            zoom=self._mock_position[2],
        )
        return True
    
    async def delete_preset(self, preset_id: int) -> bool:
        """Delete mock preset."""
        if preset_id in self._mock_presets:
            del self._mock_presets[preset_id]
        return True