"""FastAPI dependencies."""

from __future__ import annotations

from typing import Optional

from fastapi import Request

from skyguard.db import DatabaseManager


def get_db_manager(request: Request) -> Optional[DatabaseManager]:
    """Get database manager from app state."""
    return getattr(request.app.state, "db_manager", None)
