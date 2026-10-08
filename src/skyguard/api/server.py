"""SkyGuard API Server.

Command-line interface for starting the API server.
"""

from __future__ import annotations

import asyncio

import uvicorn
from loguru import logger
from typer import Typer

from skyguard.api.app import create_app
from skyguard.core.config import settings

app = Typer()


@app.command()
def start(
    host: str = settings.web.host,
    port: int = settings.web.port,
    reload: bool = False,
    workers: int = 1,
):
    """Start the SkyGuard API server.
    
    Args:
        host: Host address to bind
        port: Port to listen on
        reload: Enable auto-reload (development only)
        workers: Number of worker processes
    """
    logger.info(f"Starting SkyGuard API server on {host}:{port}")
    
    api_app = create_app()
    
    uvicorn.run(
        api_app,
        host=host,
        port=port,
        reload=reload,
        workers=workers,
        log_level="info",
    )


@app.command()
def dev():
    """Start API server in development mode."""
    logger.info("Starting SkyGuard API in development mode")
    start(host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    app()