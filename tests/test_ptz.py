"""Unit tests for PTZ module.

Tests cover:
- PTZController abstract interface
- PTZLimits validation
- PTZState data class
- ONVIFClient mock implementation
- TrackingController
- CameraCalibration coordinate conversion
- SmoothingFilter

Run with: pytest tests/test_ptz.py -v
"""

import asyncio
import math
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pytest

from skyguard.ptz.ptz_controller import (
    PTZController,
    PTZState,
    PTZLimits,
    PTZPreset,
    PTZMode,
    PTZException,
    PTZConnectionError,
    PTZCommandError,
)
from skyguard.ptz.onvif_client import (
    ONVIFClient,
    MockONVIFClient,
    ONVIFConfig,
    ONVIFCapabilities,
)
from skyguard.ptz.tracking_controller import (
    TrackingController,
    TrackingConfig,
    TrackingMode,
    TargetPriority,
    CameraCalibration,
    SmoothingFilter,
    TrackedTarget,
    PTZCommand,
)


# ========== Test Fixtures ==========

@pytest.fixture
def ptz_limits():
    """Default PTZ limits."""
    return PTZLimits()


@pytest.fixture
def mock_ptz():
    """Mock PTZ controller."""
    return MockONVIFClient(
        host="192.168.1.100",
        username="admin",
        password="password",
    )


@pytest.fixture
def camera_calibration():
    """Default camera calibration."""
    return CameraCalibration()


@pytest.fixture
def tracking_config():
    """Default tracking config."""
    return TrackingConfig()


@pytest.fixture
def tracking_controller(mock_ptz, camera_calibration, tracking_config):
    """Tracking controller with mock PTZ."""
    return TrackingController(
        ptz=mock_ptz,
        calibration=camera_calibration,
        config=tracking_config,
    )


# ========== PTZLimits Tests ==========

class TestPTZLimits:
    """Tests for PTZLimits."""
    
    def test_default_limits(self):
        """Test default limit values."""
        limits = PTZLimits()
        
        assert limits.pan_min == -180.0
        assert limits.pan_max == 180.0
        assert limits.tilt_min == -90.0
        assert limits.tilt_max == 90.0
        assert limits.zoom_min == 1.0
        assert limits.zoom_max == 20.0
    
    def test_clamp_pan(self, ptz_limits):
        """Test pan clamping."""
        assert ptz_limits.clamp_pan(0.0) == 0.0
        assert ptz_limits.clamp_pan(90.0) == 90.0
        assert ptz_limits.clamp_pan(200.0) == 180.0  # Clamped to max
        assert ptz_limits.clamp_pan(-200.0) == -180.0  # Clamped to min
    
    def test_clamp_tilt(self, ptz_limits):
        """Test tilt clamping."""
        assert ptz_limits.clamp_tilt(0.0) == 0.0
        assert ptz_limits.clamp_tilt(45.0) == 45.0
        assert ptz_limits.clamp_tilt(100.0) == 90.0
        assert ptz_limits.clamp_tilt(-100.0) == -90.0
    
    def test_clamp_zoom(self, ptz_limits):
        """Test zoom clamping."""
        assert ptz_limits.clamp_zoom(1.0) == 1.0
        assert ptz_limits.clamp_zoom(10.0) == 10.0
        assert ptz_limits.clamp_zoom(0.5) == 1.0  # Min is 1.0
        assert ptz_limits.clamp_zoom(30.0) == 20.0  # Max is 20.0
    
    def test_clamp_pan_speed(self, ptz_limits):
        """Test pan speed clamping."""
        assert ptz_limits.clamp_pan_speed(50.0) == 50.0
        assert ptz_limits.clamp_pan_speed(200.0) == 180.0
        assert ptz_limits.clamp_pan_speed(-200.0) == -180.0
    
    def test_invalid_limits(self):
        """Test that invalid limits raise ValueError."""
        with pytest.raises(ValueError):
            PTZLimits(pan_min=100.0, pan_max=50.0)  # min > max
        
        with pytest.raises(ValueError):
            PTZLimits(zoom_min=10.0, zoom_max=5.0)  # min > max


# ========== PTZState Tests ==========

class TestPTZState:
    """Tests for PTZState."""
    
    def test_default_state(self):
        """Test default state values."""
        state = PTZState()
        
        assert state.pan == 0.0
        assert state.tilt == 0.0
        assert state.zoom == 1.0
        assert state.mode == PTZMode.IDLE
        assert state.is_moving is False
    
    def test_position_tuple(self):
        """Test position tuple extraction."""
        state = PTZState(pan=45.0, tilt=10.0, zoom=2.5)
        
        assert state.position_tuple() == (45.0, 10.0, 2.5)
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        state = PTZState(pan=30.0, tilt=-15.0, zoom=3.0)
        d = state.to_dict()
        
        assert d["pan"] == 30.0
        assert d["tilt"] == -15.0
        assert d["zoom"] == 3.0
        assert d["mode"] == "idle"


# ========== MockONVIFClient Tests ==========

class TestMockONVIFClient:
    """Tests for MockONVIFClient."""
    
    @pytest.mark.asyncio
    async def test_connect(self, mock_ptz):
        """Test mock connection."""
        connected = await mock_ptz.connect()
        
        assert connected is True
        assert mock_ptz.is_connected() is True
    
    @pytest.mark.asyncio
    async def test_disconnect(self, mock_ptz):
        """Test mock disconnect."""
        await mock_ptz.connect()
        await mock_ptz.disconnect()
        
        assert mock_ptz.is_connected() is False
    
    @pytest.mark.asyncio
    async def test_get_state(self, mock_ptz):
        """Test getting state."""
        await mock_ptz.connect()
        state = await mock_ptz.get_state()
        
        assert isinstance(state, PTZState)
        assert state.pan == 0.0
        assert state.tilt == 0.0
    
    @pytest.mark.asyncio
    async def test_move_absolute(self, mock_ptz):
        """Test absolute move."""
        await mock_ptz.connect()
        result = await mock_ptz.move_absolute(pan=45.0, tilt=15.0, zoom=2.0)
        
        assert result is True
        
        state = await mock_ptz.get_state()
        assert state.pan == 45.0
        assert state.tilt == 15.0
        assert state.zoom == 2.0
    
    @pytest.mark.asyncio
    async def test_move_continuous(self, mock_ptz):
        """Test continuous move."""
        await mock_ptz.connect()
        result = await mock_ptz.move_continuous(pan_speed=10.0, tilt_speed=5.0)
        
        assert result is True
        
        state = await mock_ptz.get_state()
        assert state.pan_speed == 10.0
        assert state.tilt_speed == 5.0
        assert state.is_moving is True
    
    @pytest.mark.asyncio
    async def test_stop(self, mock_ptz):
        """Test stop."""
        await mock_ptz.connect()
        await mock_ptz.move_continuous(pan_speed=10.0, tilt_speed=5.0)
        result = await mock_ptz.stop()
        
        assert result is True
        
        state = await mock_ptz.get_state()
        assert state.pan_speed == 0.0
        assert state.is_moving is False
    
    @pytest.mark.asyncio
    async def test_move_to_home(self, mock_ptz):
        """Test move to home."""
        await mock_ptz.connect()
        result = await mock_ptz.move_to_home()
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_presets(self, mock_ptz):
        """Test preset operations."""
        await mock_ptz.connect()
        
        # Set preset
        result = await mock_ptz.set_preset(1, "Home")
        assert result is True
        
        # Get presets
        presets = await mock_ptz.get_presets()
        assert 1 in presets
        assert presets[1].name == "Home"
        
        # Delete preset
        result = await mock_ptz.delete_preset(1)
        assert result is True
        
        presets = await mock_ptz.get_presets()
        assert 1 not in presets
    
    @pytest.mark.asyncio
    async def test_context_manager(self, mock_ptz):
        """Test async context manager."""
        async with mock_ptz as ptz:
            assert ptz.is_connected() is True
        
        assert mock_ptz.is_connected() is False


# ========== CameraCalibration Tests ==========

class TestCameraCalibration:
    """Tests for CameraCalibration."""
    
    def test_default_calibration(self, camera_calibration):
        """Test default calibration values."""
        assert camera_calibration.fx == 800.0
        assert camera_calibration.fy == 800.0
        assert camera_calibration.image_width == 1920
        assert camera_calibration.image_height == 1080
    
    def test_from_fov(self):
        """Test creating calibration from FOV."""
        calib = CameraCalibration.from_fov(
            h_fov=60.0,
            v_fov=35.0,
            width=1920,
            height=1080,
        )
        
        # Check FOV is preserved
        assert calib.h_fov == 60.0
        assert calib.v_fov == 35.0
        assert calib.image_width == 1920
        assert calib.image_height == 1080
    
    def test_pixel_to_angle_center(self, camera_calibration):
        """Test conversion at image center."""
        pan, tilt = camera_calibration.pixel_to_angle(960, 540)
        
        # Center should give zero angles
        assert abs(pan) < 0.01
        assert abs(tilt) < 0.01
    
    def test_pixel_to_angle_offset(self, camera_calibration):
        """Test conversion at offset position."""
        # Right of center should give positive pan
        pan_right, _ = camera_calibration.pixel_to_angle(1440, 540)  # 960 + 480
        assert pan_right > 0
        
        # Left of center should give negative pan
        pan_left, _ = camera_calibration.pixel_to_angle(480, 540)  # 960 - 480
        assert pan_left < 0
    
    def test_angle_to_pixel_inverse(self, camera_calibration):
        """Test that angle_to_pixel is inverse of pixel_to_angle."""
        # Test round-trip conversion
        test_pixels = [(960, 540), (1200, 600), (800, 400)]
        
        for px, py in test_pixels:
            pan, tilt = camera_calibration.pixel_to_angle(px, py)
            px2, py2 = camera_calibration.angle_to_pixel(pan, tilt)
            
            # Should be close to original
            assert abs(px - px2) < 2.0  # Within 2 pixels
            assert abs(py - py2) < 2.0


# ========== SmoothingFilter Tests ==========

class TestSmoothingFilter:
    """Tests for SmoothingFilter."""
    
    def test_filter_basic(self):
        """Test basic filtering."""
        sf = SmoothingFilter(alpha=0.5)
        pan_vel, tilt_vel, zoom_vel = sf.filter(10.0, 5.0, 0.0)
        
        # With alpha=0.5, first iteration should give half the error
        assert pan_vel == 5.0  # 0.5 * 10.0
        assert tilt_vel == 2.5  # 0.5 * 5.0
    
    def test_filter_dead_zone(self):
        """Test dead zone filtering."""
        sf = SmoothingFilter(alpha=0.5, pan_threshold=1.0)
        
        # Error below threshold should be zeroed
        pan_vel, _, _ = sf.filter(0.5, 0.0, 0.0)
        assert pan_vel == 0.0
        
        # Error above threshold should pass
        pan_vel, _, _ = sf.filter(2.0, 0.0, 0.0)
        assert pan_vel == 1.0  # 0.5 * 2.0
    
    def test_filter_smoothing(self):
        """Test exponential smoothing."""
        sf = SmoothingFilter(alpha=0.3)
        
        # Apply same error multiple times
        results = []
        for _ in range(10):
            pan_vel, _, _ = sf.filter(10.0, 0.0, 0.0)
            results.append(pan_vel)
        
        # Should converge towards 10.0
        assert results[-1] > results[0]
    
    def test_reset(self):
        """Test filter reset."""
        sf = SmoothingFilter(alpha=0.5)
        sf.filter(10.0, 5.0, 0.0)
        
        sf.reset()
        
        # After reset, velocity should start from zero
        pan_vel, tilt_vel, zoom_vel = sf.filter(0.0, 0.0, 0.0)
        assert pan_vel == 0.0
        assert tilt_vel == 0.0


# ========== TrackedTarget Tests ==========

class TestTrackedTarget:
    """Tests for TrackedTarget."""
    
    def test_bbox_properties(self):
        """Test bbox property calculations."""
        target = TrackedTarget(
            track_id=1,
            bbox=(100, 100, 200, 200),  # 100x100 box
            center=(150, 150),
            confidence=0.9,
            class_id=0,
            class_name="drone",
        )
        
        assert target.width == 100
        assert target.height == 100
        assert target.area == 10000
    
    def test_age(self):
        """Test target age calculation."""
        target = TrackedTarget(
            track_id=1,
            bbox=(100, 100, 200, 200),
            center=(150, 150),
            confidence=0.9,
            class_id=0,
            class_name="drone",
        )
        
        time.sleep(0.1)
        assert target.age >= 0.1


# ========== PTZCommand Tests ==========

class TestPTZCommand:
    """Tests for PTZCommand."""
    
    def test_is_zero(self):
        """Test zero detection."""
        cmd = PTZCommand(
            pan_speed=0.0,
            tilt_speed=0.0,
            zoom_speed=0.0,
            mode=TrackingMode.IDLE,
        )
        assert cmd.is_zero() is True
        
        cmd2 = PTZCommand(
            pan_speed=1.0,
            tilt_speed=0.0,
            zoom_speed=0.0,
            mode=TrackingMode.TRACKING,
        )
        assert cmd2.is_zero() is False
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        cmd = PTZCommand(
            pan_speed=10.0,
            tilt_speed=5.0,
            zoom_speed=0.0,
            mode=TrackingMode.TRACKING,
            target_id=1,
        )
        d = cmd.to_dict()
        
        assert d["pan_speed"] == 10.0
        assert d["mode"] == "tracking"
        assert d["target_id"] == 1


# ========== TrackingController Tests ==========

class TestTrackingController:
    """Tests for TrackingController."""
    
    def test_initialization(self, tracking_controller):
        """Test controller initialization."""
        assert tracking_controller.mode == TrackingMode.IDLE
        assert tracking_controller.current_target is None
    
    def test_select_target_empty(self, tracking_controller):
        """Test target selection with no tracks."""
        result = tracking_controller.select_target([], (1080, 1920))
        assert result is None
    
    def test_select_target_largest(self, tracking_controller):
        """Test selecting largest target."""
        from skyguard.vision.tracker import Track
        
        tracks = [
            Track(
                track_id=1,
                class_id=0,
                class_name="drone",
                bbox=(100, 100, 150, 150),  # 50x50 = 2500 area
                confidence=0.9,
            ),
            Track(
                track_id=2,
                class_id=0,
                class_name="drone",
                bbox=(100, 100, 200, 200),  # 100x100 = 10000 area
                confidence=0.9,
            ),
        ]
        
        target = tracking_controller.select_target(tracks, (1080, 1920))
        
        assert target is not None
        assert target.track_id == 2
    
    def test_select_target_confidence_filter(self, tracking_controller):
        """Test filtering by confidence."""
        from skyguard.vision.tracker import Track
        
        tracks = [
            Track(
                track_id=1,
                class_id=0,
                class_name="drone",
                bbox=(100, 100, 200, 200),
                confidence=0.3,  # Below min_confidence (0.5)
            ),
        ]
        
        target = tracking_controller.select_target(tracks, (1080, 1920))
        assert target is None
    
    def test_calculate_angular_error(self, tracking_controller):
        """Test angular error calculation."""
        target = TrackedTarget(
            track_id=1,
            bbox=(960, 540, 1000, 580),  # Center at (980, 560)
            center=(980, 560),  # Slightly right and down from center
            confidence=0.9,
            class_id=0,
            class_name="drone",
        )
        
        pan_error, tilt_error = tracking_controller.calculate_angular_error(
            target, (1080, 1920)
        )
        
        # Should have non-zero error
        assert pan_error != 0.0
    
    def test_update_with_target(self, tracking_controller):
        """Test update with target."""
        from skyguard.vision.tracker import Track
        
        tracks = [
            Track(
                track_id=1,
                class_id=0,
                class_name="drone",
                bbox=(860, 440, 1060, 640),  # Center at (960, 540)
                confidence=0.9,
            ),
        ]
        
        cmd = tracking_controller.update(tracks, (1080, 1920))
        
        assert cmd is not None
        assert cmd.mode == TrackingMode.TRACKING
        assert cmd.target_id == 1
    
    def test_update_no_target(self, tracking_controller):
        """Test update without target."""
        from skyguard.vision.tracker import Track
        
        tracks = [
            Track(
                track_id=1,
                class_id=0,
                class_name="drone",
                bbox=(860, 440, 1060, 640),
                confidence=0.9,
            ),
        ]
        tracking_controller.update(tracks, (1080, 1920))
        
        # Then, remove target
        tracking_controller.last_track_time = time.time()
        cmd = tracking_controller.update([], (1080, 1920))
        
        # Should be in LOST mode initially
        assert cmd.mode == TrackingMode.LOST
    
    def test_reset(self, tracking_controller):
        """Test controller reset."""
        tracking_controller.mode = TrackingMode.TRACKING
        tracking_controller.reset()
        
        assert tracking_controller.mode == TrackingMode.IDLE
        assert tracking_controller.current_target is None
    
    def test_get_status(self, tracking_controller):
        """Test status retrieval."""
        status = tracking_controller.get_status()
        
        assert "mode" in status
        assert status["mode"] == "idle"


# ========== Integration Tests ==========

class TestIntegration:
    """Integration tests."""
    
    @pytest.mark.asyncio
    async def test_full_tracking_flow(self):
        """Test complete tracking flow."""
        from skyguard.vision.tracker import Track
        
        # Create mock PTZ
        ptz = MockONVIFClient(host="192.168.1.100")
        await ptz.connect()
        
        # Create tracking controller
        calibration = CameraCalibration()
        config = TrackingConfig()
        tracking = TrackingController(ptz, calibration, config)
        
        # Simulate tracking
        tracks = [
            Track(
                track_id=1,
                class_id=0,
                class_name="drone",
                bbox=(860, 440, 1060, 640),
                confidence=0.9,
            ),
        ]
        
        cmd = tracking.update(tracks, (1080, 1920))
        
        # Should generate command
        assert cmd is not None
        
        # Execute on PTZ
        if not cmd.is_zero():
            await ptz.move_continuous(cmd.pan_speed, cmd.tilt_speed)
        
        # Stop
        await ptz.stop()
        await ptz.disconnect()
    
    @pytest.mark.asyncio
    async def test_context_manager_usage(self):
        """Test PTZ context manager."""
        async with MockONVIFClient(host="192.168.1.100") as ptz:
            assert ptz.is_connected()
            
            # Move
            await ptz.move_absolute(45.0, 15.0, 2.0)
            
            state = await ptz.get_state()
            assert state.pan == 45.0
        
        # Should be disconnected after context
        assert not ptz.is_connected()


# ========== Run Tests ==========

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])