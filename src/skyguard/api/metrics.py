"""Prometheus metrics instrumentation for SkyGuard API (Sprint 12).

Exposes standard RED metrics (Rate, Errors, Duration) plus custom
business metrics for detections, tracks, and alerts. The metrics
are exposed at /metrics in Prometheus exposition format.

Usage in app.py:
    from skyguard.api.metrics import init_metrics, metrics_middleware, prometheus_endpoint

    init_metrics(app)
    app.add_middleware(BaseHTTPMiddleware, dispatch=metrics_middleware)
    app.add_route("/metrics", prometheus_endpoint)
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    make_asgi_app,
)

if TYPE_CHECKING:
    from fastapi import FastAPI, Request
    from starlette.responses import Response


# Use a dedicated registry to avoid polluting the global one in tests.
REGISTRY = CollectorRegistry()

# ---- HTTP RED metrics ----
HTTP_REQUESTS = Counter(
    "skyguard_http_requests_total",
    "Total HTTP requests",
    labelnames=("method", "path", "status"),
    registry=REGISTRY,
)

HTTP_REQUEST_DURATION = Histogram(
    "skyguard_http_request_duration_seconds",
    "HTTP request latency in seconds",
    labelnames=("method", "path"),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    registry=REGISTRY,
)

HTTP_IN_PROGRESS = Gauge(
    "skyguard_http_requests_in_progress",
    "HTTP requests currently in flight",
    registry=REGISTRY,
)

# ---- Business metrics ----
DETECTIONS_TOTAL = Counter(
    "skyguard_detections_total",
    "Total detections produced",
    labelnames=("class_name",),
    registry=REGISTRY,
)

TRACKS_ACTIVE = Gauge(
    "skyguard_tracks_active",
    "Currently active tracks",
    registry=REGISTRY,
)

ALERTS_TOTAL = Counter(
    "skyguard_alerts_total",
    "Total alerts raised",
    labelnames=("severity",),
    registry=REGISTRY,
)

STREAMS_ACTIVE = Gauge(
    "skyguard_streams_active",
    "Active video streams",
    registry=REGISTRY,
)

MODEL_INFERENCE_SECONDS = Histogram(
    "skyguard_model_inference_seconds",
    "Model inference time per frame",
    buckets=(0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0),
    registry=REGISTRY,
)

DB_CONNECTION_STATUS = Gauge(
    "skyguard_db_connected",
    "Database connection status (1=connected, 0=disconnected)",
    registry=REGISTRY,
)


def init_metrics(app: "FastAPI") -> None:
    """Mount the Prometheus ASGI app at /metrics on the FastAPI app."""
    app.mount("/metrics", make_asgi_app(registry=REGISTRY))


async def metrics_middleware(request: "Request", call_next):
    """ASGI middleware that records RED metrics for every request.

    Usage:
        app.middleware("http")(metrics_middleware)
    """
    # Skip metrics endpoint itself to avoid recursive counting
    if request.url.path.startswith("/metrics"):
        return await call_next(request)

    HTTP_IN_PROGRESS.inc()
    start = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    except Exception:
        status_code = 500
        raise
    finally:
        duration = time.perf_counter() - start
        HTTP_IN_PROGRESS.dec()
        path = _normalize_path(request.url.path)
        HTTP_REQUESTS.labels(
            method=request.method,
            path=path,
            status=str(status_code),
        ).inc()
        HTTP_REQUEST_DURATION.labels(
            method=request.method,
            path=path,
        ).observe(duration)


def _normalize_path(path: str) -> str:
    """Collapse resource IDs in paths to avoid high-cardinality label explosion.

    /api/v1/devices/123  ->  /api/v1/devices/{id}
    /api/v1/alerts/abc   ->  /api/v1/alerts/{id}
    """
    parts = path.strip("/").split("/")
    if len(parts) <= 1:
        return path
    # If the last segment looks like an ID (not purely alphabetic), replace it.
    if parts[-1] and not parts[-1].isalpha():
        parts[-1] = "{id}"
    return "/" + "/".join(parts)


def record_detection(class_name: str) -> None:
    """Record a detection (callable from detection pipeline)."""
    DETECTIONS_TOTAL.labels(class_name=class_name).inc()


def record_alert(severity: str = "warning") -> None:
    """Record an alert (callable from alert pipeline)."""
    ALERTS_TOTAL.labels(severity=severity).inc()


def set_tracks_active(count: int) -> None:
    """Update the active tracks gauge."""
    TRACKS_ACTIVE.set(count)


def set_streams_active(count: int) -> None:
    """Update the active streams gauge."""
    STREAMS_ACTIVE.set(count)


def observe_inference_time(seconds: float) -> None:
    """Observe a model inference duration."""
    MODEL_INFERENCE_SECONDS.observe(seconds)


def set_db_connected(connected: bool) -> None:
    """Update the DB connection gauge."""
    DB_CONNECTION_STATUS.set(1 if connected else 0)
