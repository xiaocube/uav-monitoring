"""PTZ (Pan-Tilt-Zoom) Camera Control Module for SkyGuard.

This module provides:
- Unified PTZ control interface abstraction
- ONVIF protocol implementation for IP cameras
- Serial/Pelco-D protocol for traditional PTZ systems
- Tracking controller integration with ByteTrack
- Smooth control algorithms (anti-shake, velocity limiting)
- Adaptive zoom based on target size

Architecture:
    ┌─────────────────────────────────────────────┐
    │         TrackingController                   │
    │  (Integrates ByteTrack output -> PTZ cmd)   │
    └──────────────────┬──────────────────────────┘
                       │
    ┌──────────────────▼──────────────────────────┐
    │          PTZController (Abstract)           │
    └──────────────────┬──────────────────────────┘
                       │
          ┌────────────┴────────────┐
          │                         │
    ┌─────▼─────┐           ┌───────▼───────┐
    │  ONVIF    │           │ Serial/PelcoD │
    │  Client   │           │   Protocol    │
    └───────────┘           └───────────────┘

Usage:
    from skyguard.ptz import PTZController, ONVIFClient
    
    # Create PTZ controller
    ptz = ONVIFClient(
        host="192.168.1.100",
        port=80,
        username="admin",
        password="admin123"
    )
    
    # Move to absolute position
    ptz.move_absolute(pan=90.0, tilt=15.0, zoom=2.0)
    
    # Or use continuous move for tracking
    ptz.move_continuous(pan_speed=5.0, tilt_speed=3.0)
"""

from skyguard.ptz.ptz_controller import PTZController, PTZState, PTZLimits
from skyguard.ptz.onvif_client import ONVIFClient
from skyguard.ptz.tracking_controller import TrackingController, TrackingConfig

__all__ = [
    # Abstract interfaces
    "PTZController",
    "PTZState",
    "PTZLimits",
    # ONVIF implementation
    "ONVIFClient",
    # Tracking integration
    "TrackingController",
    "TrackingConfig",
]

__version__ = "1.0.0"