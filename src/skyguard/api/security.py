"""API security middleware for SkyGuard (Sprint 12).

Provides:
  * API Key authentication via X-API-Key header
  * Rate limiting via slowapi (token bucket per client IP)
  * Request ID injection for correlation
  * Public path bypass (health, docs, metrics don't need auth)

Design:
  * Security is opt-in: if settings.security.enabled is False,
    all requests pass through without auth (dev mode).
  * Rate limiting is always on when security.enabled is True,
    configurable via settings.security.rate_limit_per_minute.
  * The API key is compared with hmac.compare_digest to prevent
    timing attacks.
"""
from __future__ import annotations

import hmac
import uuid
from typing import Optional

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware

from skyguard.core.config import get_settings


# ---------------------------------------------------------------------------
# Rate Limiter
# ---------------------------------------------------------------------------

def create_limiter() -> Limiter:
    """Create a slowapi Limiter configured from settings."""
    settings = get_settings()
    return Limiter(
        key_func=get_remote_address,
        default_limits=[
            f"{settings.security.rate_limit_per_minute}/minute"
        ] if settings.security.rate_limit_enabled else [],
        enabled=settings.security.enabled and settings.security.rate_limit_enabled,
    )


# ---------------------------------------------------------------------------
# API Key Auth Middleware
# ---------------------------------------------------------------------------

class APIKeyMiddleware(BaseHTTPMiddleware):
    """Validate X-API-Key header on non-public paths.

    If settings.security.enabled is False, this middleware is a no-op
    (all requests pass through). This keeps dev mode frictionless.
    """

    def __init__(self, app, public_paths: Optional[list] = None) -> None:
        super().__init__(app)
        self.public_paths = set(public_paths or [])

    async def dispatch(self, request: Request, call_next):
        settings = get_settings()

        # If security is disabled, pass through
        if not settings.security.enabled:
            return await call_next(request)

        path = request.url.path

        # Public paths bypass auth
        if path in self.public_paths or path.startswith(("/docs", "/redoc", "/openapi.json", "/static", "/metrics")):
            return await call_next(request)

        # Check API key
        provided = request.headers.get(settings.security.api_key_header, "")
        expected = settings.security.api_key

        if not expected:
            # No API key configured -> reject all non-public requests
            return JSONResponse(
                status_code=503,
                content={"detail": "API security is enabled but no API key is configured."},
            )

        if not provided or not hmac.compare_digest(provided, expected):
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key."},
                headers={"WWW-Authenticate": settings.security.api_key_header},
            )

        return await call_next(request)


# ---------------------------------------------------------------------------
# Request ID Middleware
# ---------------------------------------------------------------------------

class RequestIDMiddleware(BaseHTTPMiddleware):
    """Inject a unique request ID into every request for correlation.

    - Reads X-Request-ID header if present (from upstream proxy).
    - Otherwise generates a UUID4.
    - Sets X-Request-ID on the response.
    """

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())

        # Store on request state for downstream handlers
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


# ---------------------------------------------------------------------------
# Auth dependency (for per-route use)
# ---------------------------------------------------------------------------

async def verify_api_key(request: Request) -> bool:
    """FastAPI dependency that verifies the API key.

    Usage:
        from skyguard.api.security import verify_api_key
        @router.get("/secure", dependencies=[Depends(verify_api_key)])
        async def secure_endpoint():
            ...
    """
    settings = get_settings()
    if not settings.security.enabled:
        return True

    provided = request.headers.get(settings.security.api_key_header, "")
    expected = settings.security.api_key

    if not provided or not hmac.compare_digest(provided, expected):
        from fastapi import HTTPException, status
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": settings.security.api_key_header},
        )
    return True
