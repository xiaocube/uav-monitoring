"""API Routes Module."""

from skyguard.api.routes.devices import router as devices_router
from skyguard.api.routes.monitor import router as monitor_router
from skyguard.api.routes.alerts import router as alerts_router
from skyguard.api.routes.config import router as config_router
from skyguard.api.routes.ws import router as ws_router

__all__ = [
    "devices_router",
    "monitor_router",
    "alerts_router",
    "config_router",
    "ws_router",
]