"""SkyGuard Web API Module.

This module provides:
- FastAPI-based REST API for device management, monitoring, and configuration
- WebSocket real-time streaming of detection and tracking data
- Authentication and authorization
- API clients for frontend integration

Architecture:
    ┌─────────────────────────────────────────────────────┐
    │                    FastAPI App                       │
    ├─────────────┬─────────────┬─────────────────────────┤
    │   REST API  │   WebSocket  │   Middleware           │
    │  (CRUD)     │  (Realtime)  │  (Auth, CORS, Logging) │
    └─────────────┴─────────────┴─────────────────────────┘
            │              │
            ▼              ▼
    ┌──────────────┐  ┌──────────────┐
    │ API Routes   │  │ Stream       │
    │  - devices   │  │  - frames    │
    │  - monitor   │  │  - detections│
    │  - alerts    │  │  - tracks    │
    │  - config    │  └──────────────┘
    └──────────────┘

API Version: v1
"""

from skyguard.api.app import create_app
from skyguard.api.models import (
    DeviceInfo,
    DetectionResult,
    TrackingResult,
    AlertInfo,
    CameraConfig,
    APIResponse,
)

__all__ = [
    "create_app",
    "DeviceInfo",
    "DetectionResult",
    "TrackingResult",
    "AlertInfo",
    "CameraConfig",
    "APIResponse",
]

__version__ = "1.0.0"