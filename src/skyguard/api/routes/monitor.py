"""Monitoring API Routes.

Endpoints for real-time monitoring and status.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Dict

import psutil
from fastapi import APIRouter

from skyguard.api.models import APIResponse, SystemStatus

router = APIRouter()

_start_time = time.time()

# Initialize CPU monitoring (first call returns 0.0, subsequent calls return actual %)
psutil.cpu_percent(interval=None)


async def _get_cpu_percent() -> float:
    """Get CPU percentage non-blocking."""
    return await asyncio.to_thread(psutil.cpu_percent, interval=None)


@router.get("/status", response_model=APIResponse)
async def get_system_status():
    """Get current system status."""
    uptime = time.time() - _start_time

    cpu_percent = await _get_cpu_percent()
    memory = await asyncio.to_thread(psutil.virtual_memory)

    status = SystemStatus(
        uptime=uptime,
        cpu_usage=cpu_percent,
        memory_usage=memory.percent,
        gpu_usage=None,
        active_streams=0,
        active_tracks=0,
        avg_fps=0.0,
        models_loaded=0,
    )

    return APIResponse.ok(data=status)


@router.get("/metrics", response_model=APIResponse)
async def get_metrics():
    """Get system metrics."""
    uptime = time.time() - _start_time

    cpu_percent = await _get_cpu_percent()
    memory = await asyncio.to_thread(psutil.virtual_memory)
    disk = await asyncio.to_thread(psutil.disk_usage, "/")

    # net_connections() may raise AccessDenied on macOS without root
    try:
        net_connections = await asyncio.to_thread(lambda: len(psutil.net_connections()))
    except (psutil.AccessDenied, psutil.NoSuchProcess, PermissionError):
        net_connections = 0

    cpu_freq = await asyncio.to_thread(psutil.cpu_freq)
    freq_current = cpu_freq.current if cpu_freq else None

    metrics = {
        "uptime": uptime,
        "cpu": {
            "percent": cpu_percent,
            "count": psutil.cpu_count(),
            "freq": freq_current,
        },
        "memory": {
            "percent": memory.percent,
            "total": memory.total,
            "used": memory.used,
            "available": memory.available,
        },
        "disk": {
            "percent": disk.percent,
            "total": disk.total,
            "used": disk.used,
            "free": disk.free,
        },
        "network": {
            "connections": net_connections,
        },
        "timestamp": datetime.now().isoformat(),
    }

    return APIResponse.ok(data=metrics)


@router.get("/health", response_model=Dict)
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "uptime": time.time() - _start_time,
    }