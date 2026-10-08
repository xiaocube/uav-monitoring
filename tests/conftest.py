"""Shared pytest fixtures for the SkyGuard test suite."""
from __future__ import annotations

import pytest

from skyguard.core.config import clear_settings_cache, get_settings


@pytest.fixture(autouse=True)
def _reset_settings_cache():
    """Each test gets a clean settings cache so YAML is re-read."""
    clear_settings_cache()
    yield
    clear_settings_cache()


@pytest.fixture
def settings():
    return get_settings()
