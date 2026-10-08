"""
Logging system for SkyGuard.

Design:
  * Built on loguru (thread-safe, multi-handler, simple API).
  * Single setup_logging() entry point; idempotent.
  * Console (rich-colored) + rotating file sink.
  * Optional JSON sink for production / log shippers.
  * Inhibit noisy 3rd-party loggers by default (ultralytics, urllib3, ...).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

from loguru import logger

from skyguard.core.config import LoggingConfig

# A module-level guard so setup_logging() can be called many times safely.
_configured = False

# Mapping from stdlib logging levels to loguru levels.
_STDLIB_TO_LOGURU = {
    "CRITICAL": "CRITICAL",
    "ERROR": "ERROR",
    "WARNING": "WARNING",
    "WARN": "WARNING",
    "INFO": "INFO",
    "DEBUG": "DEBUG",
    "NOTSET": "DEBUG",
}

# These libraries are noisy at INFO; raise their level to WARNING.
_QUIET_LOGGERS = (
    "ultralytics",
    "urllib3",
    "httpx",
    "httpcore",
    "asyncio",
    "PIL",
    "matplotlib",
    "filelock",
)


def _intercept_stdlib() -> None:
    """Route stdlib logging records into loguru so we have a single sink."""
    handler_id = logger.add(
        lambda m: m,
        level=0,
        format="{message}",
        enqueue=True,
    )

    class _InterceptHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:  # type: ignore[override]
            try:
                level = _STDLIB_TO_LOGURU.get(record.levelname, "INFO")
                logger.opt(depth=6, exception=record.exc_info).log(level, record.getMessage())
            except Exception:
                # never let logging break the program
                sys.stderr.write("SkyGuard log interceptor failed\n")

    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)
    # Silence the noisy ones.
    for name in _QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    _ = handler_id  # keep linter quiet


def setup_logging(config: Optional[LoggingConfig] = None) -> None:
    """Configure loguru + stdlib logging based on a LoggingConfig."""
    global _configured
    if _configured:
        return

    if config is None:
        # Lazy import to avoid a circular dependency at module import time.
        from skyguard.core.config import get_settings

        config = get_settings().logging

    logger.remove()

    # --- Console sink ---
    console_format = (
        "<green>{time:HH:mm:ss.SSS}</green> | "
        "<level>{level: <7}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
    logger.add(
        sys.stderr,
        level=config.level,
        format=console_format,
        colorize=config.colorize,
        enqueue=config.enqueue,
        backtrace=False,
        diagnose=False,
    )

    # --- File sink (rotated) ---
    log_dir: Path = Path("./logs")
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_dir / "skyguard.log",
            level=config.level,
            rotation=config.rotation,
            retention=config.retention,
            enqueue=True,
            serialize=config.serialize_json,
            backtrace=True,
            diagnose=False,
        )
        if config.serialize_json:
            # Additional structured sink for production log shippers
            logger.add(
                log_dir / "skyguard.json",
                level=config.level,
                rotation=config.rotation,
                retention=config.retention,
                enqueue=True,
                serialize=True,
                backtrace=True,
                diagnose=False,
            )
    except OSError as e:
        # Read-only filesystem or permission denied: fall back to stderr only.
        sys.stderr.write(f"[SkyGuard] file log disabled: {e}\n")

    _intercept_stdlib()
    _configured = True
    logger.debug("SkyGuard logging initialized at level={}", config.level)


def get_logger(name: Optional[str] = None):
    """Return a logger bound to the given module name (for traceability)."""
    if not _configured:
        setup_logging()
    return logger.bind(module=name) if name else logger
