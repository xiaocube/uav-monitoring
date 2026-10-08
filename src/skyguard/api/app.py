"""SkyGuard FastAPI Application (Sprint 12 - Industrial Grade).

This module creates and configures the FastAPI application with:
- CORS middleware
- GZip compression middleware
- Request ID correlation middleware
- API Key authentication middleware (opt-in via settings.security.enabled)
- Rate limiting (slowapi)
- Prometheus metrics instrumentation (/metrics endpoint)
- API routing
- WebSocket routing
- Liveness (/health) and readiness (/ready) probes
- Error handling with structured JSON responses
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from skyguard import __version__
from skyguard.api.metrics import init_metrics, metrics_middleware, set_db_connected
from skyguard.api.routes import devices, monitor, alerts, config, ws
from skyguard.api.security import APIKeyMiddleware, RequestIDMiddleware, create_limiter
from skyguard.core.config import settings
from skyguard.db import DatabaseConfig, DatabaseManager

STATIC_DIR = Path(__file__).parent / "static"

_start_time = time.time()


class APILoggingHandler(logging.Handler):
    """Logging handler for FastAPI."""

    def emit(self, record: logging.LogRecord) -> None:
        """Emit log record to loguru."""
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelname

        logger.log(level, record.getMessage())


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan event handler."""
    logger.info("SkyGuard API v{} starting up", __version__)

    from skyguard.db.types import DatabaseConfig as DBConfig

    db_config = DBConfig(
        host=settings.database.host,
        port=settings.database.port,
        database=settings.database.database,
        username=settings.database.username,
        password=settings.database.password,
        pool_size=settings.database.pool_size,
    )

    app.state.db_manager = DatabaseManager(db_config)

    db_connected = False
    if settings.database.enabled and app.state.db_manager.connect():
        app.state.db_manager.create_tables()
        db_connected = True
        logger.success("Database connected and tables created")
    else:
        if not settings.database.enabled:
            logger.warning("Database disabled in config - API running with in-memory fallback")
        else:
            logger.warning("Database not available - API running with in-memory fallback")

    set_db_connected(db_connected)
    app.state.db_connected = db_connected
    app.state.start_time = _start_time

    yield

    if hasattr(app.state, "db_manager") and app.state.db_manager:
        app.state.db_manager.disconnect()

    logger.info("SkyGuard API shutting down")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="SkyGuard API",
        description="AI Low-Altitude Intelligent Monitoring System API",
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ---- Rate limiter (slowapi) ----
    limiter = create_limiter()
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # ---- Middleware stack (order matters: outermost first) ----
    # Request ID must be outermost so all downstream middleware/handlers see it.
    app.add_middleware(RequestIDMiddleware)

    # GZip compression
    if settings.security.gzip_enabled:
        app.add_middleware(GZipMiddleware, minimum_size=settings.security.gzip_min_size)

    # API Key auth (no-op when security.enabled is False)
    app.add_middleware(APIKeyMiddleware, public_paths=settings.security.public_paths)

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.web.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Prometheus metrics middleware + /metrics endpoint
    app.middleware("http")(metrics_middleware)
    init_metrics(app)

    # ---- Include API routes ----
    app.include_router(devices.router, prefix="/api/v1/devices", tags=["Devices"])
    app.include_router(monitor.router, prefix="/api/v1/monitor", tags=["Monitoring"])
    app.include_router(alerts.router, prefix="/api/v1/alerts", tags=["Alerts"])
    app.include_router(config.router, prefix="/api/v1/config", tags=["Config"])
    app.include_router(ws.router, prefix="/ws", tags=["WebSocket"])

    # ---- Static files (CSS, JS, images) ----
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
        logger.info(f"Static files mounted from {STATIC_DIR}")

    # ---- Root: serve dashboard ----
    @app.get("/", response_class=FileResponse)
    async def root():
        """Serve the SkyGuard dashboard."""
        index_path = STATIC_DIR / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path))
        return JSONResponse(
            content={
                "status": "ok",
                "service": "SkyGuard API",
                "version": __version__,
                "docs": "/docs",
                "note": "Dashboard not built. Visit /docs for API.",
            }
        )

    # ---- Liveness probe ----
    @app.get("/health")
    async def health() -> Dict[str, Any]:
        """Liveness check — is the process alive?"""
        return {
            "status": "ok",
            "service": "SkyGuard API",
            "version": __version__,
            "docs": "/docs",
        }

    # ---- Readiness probe ----
    @app.get("/ready")
    async def ready(request: Request) -> Dict[str, Any]:
        """Readiness check — is the service ready to accept traffic?

        Checks DB connectivity (if enabled). Returns 503 if not ready.
        """
        checks: Dict[str, Any] = {}
        ready = True

        # DB check
        db_enabled = settings.database.enabled
        db_connected = getattr(request.app.state, "db_connected", False)
        if db_enabled:
            checks["database"] = "ok" if db_connected else "unavailable"
            if not db_connected:
                ready = False
        else:
            checks["database"] = "disabled"

        # Uptime
        uptime = time.time() - _start_time
        checks["uptime_sec"] = round(uptime, 1)

        response = {
            "status": "ready" if ready else "not_ready",
            "version": __version__,
            "checks": checks,
        }
        if not ready:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=response,
            )
        return response

    # ---- Error handlers ----
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        """Handle validation errors."""
        logger.error(f"Validation error: {exc.errors()}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": exc.errors(), "body": exc.body},
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Handle generic exceptions."""
        request_id = getattr(request.state, "request_id", "unknown")
        logger.error(f"[{request_id}] Unexpected error: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": "Internal server error",
                "request_id": request_id,
            },
        )

    logger.success("SkyGuard API v{} application created", __version__)

    return app


# For development/testing
if __name__ == "__main__":
    import uvicorn

    app = create_app()
    uvicorn.run(
        app,
        host=settings.web.host,
        port=settings.web.port,
        log_level="info",
    )
