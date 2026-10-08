"""
SkyGuard exception hierarchy.

We define a single base SkyGuardError so callers can catch "any SkyGuard issue"
while still allowing precise handling of specific failure modes.
"""
from __future__ import annotations


class SkyGuardError(Exception):
    """Base class for all SkyGuard errors."""


class ConfigurationError(SkyGuardError):
    """Raised when configuration loading or validation fails."""


class DeviceError(SkyGuardError):
    """Raised when compute device detection or initialization fails."""


class ModelError(SkyGuardError):
    """Raised when a model cannot be loaded or executed."""


class VideoError(SkyGuardError):
    """Raised on video capture / decoding / streaming issues."""


class TrackingError(SkyGuardError):
    """Raised on tracking pipeline issues."""


class PTZError(SkyGuardError):
    """Raised on PTZ control failures."""
