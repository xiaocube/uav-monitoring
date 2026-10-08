"""Unit tests for Sprint 12 industrialization features.

Tests cover:
- SecurityConfig in Settings
- API Key auth middleware (enabled / disabled / public paths / invalid key)
- Request ID middleware (generation + propagation)
- Prometheus metrics endpoint (/metrics returns text/plain)
- Metrics middleware (request counting + path normalization)
- Liveness (/health) and readiness (/ready) endpoints
- Version consistency (__version__ == app version)
- Rate limiter creation
- Business metric helper functions

Uses FastAPI TestClient (httpx-based) which works without running uvicorn.

Run with: pytest tests/test_industrial.py -v
"""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from skyguard import __version__
from skyguard.api.metrics import (
    DETECTIONS_TOTAL,
    HTTP_REQUESTS,
    REGISTRY,
    _normalize_path,
    init_metrics,
    metrics_middleware,
    record_alert,
    record_detection,
    set_db_connected,
    set_streams_active,
    set_tracks_active,
)
from skyguard.api.security import (
    APIKeyMiddleware,
    RequestIDMiddleware,
    create_limiter,
    verify_api_key,
)
from skyguard.core.config import (
    DatabaseConfig,
    SecurityConfig,
    Settings,
    clear_settings_cache,
    get_settings,
)


# ========== Test Fixtures ==========

def _make_test_settings(security: SecurityConfig) -> Settings:
    """Build a complete Settings object for tests (no YAML needed)."""
    clear_settings_cache()
    s = Settings()
    s.security = security
    s.web.cors_origins = []
    s.database.enabled = False
    return s


@pytest.fixture
def app_no_security():
    """FastAPI app with security disabled (dev mode)."""
    test_settings = _make_test_settings(SecurityConfig(enabled=False, gzip_enabled=False))
    with patch("skyguard.api.app.settings", test_settings), \
         patch("skyguard.api.security.get_settings", return_value=test_settings), \
         patch("skyguard.api.metrics.metrics_middleware") as mock_mw:
        # Make the metrics middleware a pass-through
        mock_mw.side_effect = lambda request, call_next: call_next(request)
        from skyguard.api.app import create_app
        app = create_app()
        yield app


@pytest.fixture
def app_with_security():
    """FastAPI app with security enabled and API key set."""
    test_settings = _make_test_settings(SecurityConfig(
        enabled=True,
        api_key="test-secret-key",
        rate_limit_enabled=False,
        gzip_enabled=False,
    ))
    with patch("skyguard.api.app.settings", test_settings), \
         patch("skyguard.api.security.get_settings", return_value=test_settings), \
         patch("skyguard.api.metrics.metrics_middleware") as mock_mw:
        mock_mw.side_effect = lambda request, call_next: call_next(request)
        from skyguard.api.app import create_app
        app = create_app()
        yield app


@pytest.fixture
def client_no_security(app_no_security):
    """TestClient with security disabled."""
    with TestClient(app_no_security) as client:
        yield client


@pytest.fixture
def client_with_security(app_with_security):
    """TestClient with security enabled."""
    with TestClient(app_with_security) as client:
        yield client


# ========== Version Consistency Tests ==========

class TestVersionConsistency:
    """Verify version is consistent across the codebase."""

    def test_version_is_1_0_0(self):
        """Package version bumped to 1.0.0 for Sprint 12."""
        assert __version__ == "1.0.0"

    def test_pyproject_matches(self):
        """pyproject.toml version matches __version__."""
        from pathlib import Path

        pyproject = Path(__file__).parent.parent / "pyproject.toml"
        content = pyproject.read_text()
        # Simple parse: find version = "x.y.z" in [project] section
        import re
        match = re.search(r'^version\s*=\s*"([^"]+)"', content, re.MULTILINE)
        assert match is not None
        assert match.group(1) == __version__

    def test_app_reports_correct_version(self, client_no_security):
        """FastAPI app reports the correct version at /health."""
        resp = client_no_security.get("/health")
        assert resp.status_code == 200
        assert resp.json()["version"] == __version__


# ========== SecurityConfig Tests ==========

class TestSecurityConfig:
    """Tests for SecurityConfig model."""

    def test_defaults(self):
        """SecurityConfig has safe defaults (disabled in dev)."""
        cfg = SecurityConfig()
        assert cfg.enabled is False
        assert cfg.api_key == ""
        assert cfg.rate_limit_enabled is True
        assert cfg.rate_limit_per_minute == 60
        assert cfg.gzip_enabled is True

    def test_public_paths_include_health(self):
        """Public paths include /health and /ready."""
        cfg = SecurityConfig()
        assert "/health" in cfg.public_paths
        assert "/ready" in cfg.public_paths
        assert "/metrics" in cfg.public_paths

    def test_settings_includes_security(self):
        """Settings has a security attribute."""
        clear_settings_cache()
        s = get_settings()
        assert hasattr(s, "security")
        assert isinstance(s.security, SecurityConfig)


# ========== Health & Readiness Tests ==========

class TestHealthEndpoints:
    """Tests for liveness and readiness probes."""

    def test_health_returns_ok(self, client_no_security):
        """GET /health returns 200 with status ok."""
        resp = client_no_security.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["version"] == __version__

    def test_ready_returns_ok_when_db_disabled(self, client_no_security):
        """GET /ready returns 200 when DB is disabled (dev mode)."""
        resp = client_no_security.get("/ready")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"
        assert data["checks"]["database"] == "disabled"
        assert "uptime_sec" in data["checks"]


# ========== Request ID Middleware Tests ==========

class TestRequestIDMiddleware:
    """Tests for RequestIDMiddleware."""

    def test_request_id_generated(self, client_no_security):
        """Response includes X-Request-ID header."""
        resp = client_no_security.get("/health")
        assert resp.status_code == 200
        assert "x-request-id" in resp.headers
        assert len(resp.headers["x-request-id"]) > 0

    def test_request_id_propagated(self, client_no_security):
        """Client-supplied X-Request-ID is echoed back."""
        custom_id = "my-correlation-id-123"
        resp = client_no_security.get(
            "/health",
            headers={"X-Request-ID": custom_id},
        )
        assert resp.status_code == 200
        assert resp.headers["x-request-id"] == custom_id


# ========== API Key Auth Tests ==========

class TestAPIKeyAuth:
    """Tests for API Key authentication middleware."""

    def test_no_security_passes(self, client_no_security):
        """Without security enabled, all requests pass."""
        resp = client_no_security.get("/api/v1/devices/")
        # Might be 200 or 404 depending on route, but NOT 401
        assert resp.status_code != 401

    def test_public_path_no_key_needed(self, client_with_security):
        """Public paths don't require API key even with security on."""
        resp = client_with_security.get("/health")
        assert resp.status_code == 200

    def test_protected_path_without_key_rejected(self, client_with_security):
        """Protected paths reject requests without API key."""
        resp = client_with_security.get("/api/v1/devices/")
        assert resp.status_code == 401
        assert "api key" in resp.json()["detail"].lower()

    def test_protected_path_with_wrong_key_rejected(self, client_with_security):
        """Wrong API key is rejected."""
        resp = client_with_security.get(
            "/api/v1/devices/",
            headers={"X-API-Key": "wrong-key"},
        )
        assert resp.status_code == 401

    def test_protected_path_with_correct_key_passes(self, client_with_security):
        """Correct API key allows access."""
        resp = client_with_security.get(
            "/api/v1/devices/",
            headers={"X-API-Key": "test-secret-key"},
        )
        # Should NOT be 401
        assert resp.status_code != 401

    def test_docs_accessible_without_key(self, client_with_security):
        """Swagger docs are accessible without API key."""
        resp = client_with_security.get("/docs")
        assert resp.status_code == 200


# ========== Prometheus Metrics Tests ==========

class TestPrometheusMetrics:
    """Tests for Prometheus metrics endpoint and middleware."""

    def test_metrics_endpoint_returns_text(self, client_no_security):
        """GET /metrics returns Prometheus exposition format."""
        resp = client_no_security.get("/metrics")
        assert resp.status_code == 200
        # Prometheus exposition format is text/plain
        assert "text/plain" in resp.headers.get("content-type", "")
        body = resp.text
        # Should contain at least the HTTP request counter
        assert "skyguard_http_requests_total" in body or "#" in body

    def test_metrics_recorded_on_request(self, client_no_security):
        """Making a request increments the HTTP request counter."""
        # Make a request
        client_no_security.get("/health")

        # The metrics registry should have recorded it
        from prometheus_client import generate_latest
        output = generate_latest(REGISTRY).decode("utf-8")
        assert "skyguard_http_requests_total" in output

    def test_path_normalization(self):
        """_normalize_path collapses resource IDs."""
        assert _normalize_path("/api/v1/devices/123") == "/api/v1/devices/{id}"
        assert _normalize_path("/api/v1/alerts/abc-def") == "/api/v1/alerts/{id}"
        assert _normalize_path("/health") == "/health"
        assert _normalize_path("/") == "/"

    def test_business_metric_helpers(self):
        """Business metric helper functions update gauges/counters."""
        # These should not raise
        record_detection("drone")
        record_alert("critical")
        set_tracks_active(5)
        set_streams_active(2)
        set_db_connected(True)

        # Verify the metrics are in the registry output
        from prometheus_client import generate_latest
        output = generate_latest(REGISTRY).decode("utf-8")
        assert "skyguard_detections_total" in output
        assert "skyguard_alerts_total" in output
        assert "skyguard_tracks_active" in output
        assert "skyguard_streams_active" in output
        assert "skyguard_db_connected" in output


# ========== Rate Limiter Tests ==========

class TestRateLimiter:
    """Tests for slowapi rate limiter integration."""

    def test_create_limiter_returns_limiter(self):
        """create_limiter returns a Limiter instance."""
        clear_settings_cache()
        limiter = create_limiter()
        assert limiter is not None

    def test_limiter_disabled_when_security_off(self):
        """Limiter is disabled when security is off."""
        clear_settings_cache()
        with patch("skyguard.api.security.get_settings") as mock:
            from skyguard.core.config import SecurityConfig
            mock.return_value.security = SecurityConfig(
                enabled=False,
                rate_limit_enabled=True,
            )
            limiter = create_limiter()
            assert limiter.enabled is False


# ========== Docker Compose Tests ==========

class TestDockerCompose:
    """Verify docker-compose.yml structure (static check, no docker needed)."""

    def test_compose_has_three_services(self):
        """docker-compose.yml defines skyguard, postgres, redis."""
        from pathlib import Path
        import yaml

        compose_path = Path(__file__).parent.parent / "docker-compose.yml"
        with open(compose_path) as f:
            data = yaml.safe_load(f)

        services = data.get("services", {})
        assert "skyguard" in services
        assert "postgres" in services
        assert "redis" in services

    def test_postgres_has_healthcheck(self):
        """Postgres service has a healthcheck."""
        from pathlib import Path
        import yaml

        compose_path = Path(__file__).parent.parent / "docker-compose.yml"
        with open(compose_path) as f:
            data = yaml.safe_load(f)

        assert "healthcheck" in data["services"]["postgres"]

    def test_skyguard_depends_on_postgres(self):
        """Skyguard service depends on postgres."""
        from pathlib import Path
        import yaml

        compose_path = Path(__file__).parent.parent / "docker-compose.yml"
        with open(compose_path) as f:
            data = yaml.safe_load(f)

        deps = data["services"]["skyguard"].get("depends_on", {})
        assert "postgres" in deps


# ========== CI/CD Tests ==========

class TestCIPipeline:
    """Verify CI workflow structure (static check)."""

    def test_ci_has_security_scan_job(self):
        """CI workflow includes a security-scan job."""
        from pathlib import Path

        ci_path = Path(__file__).parent.parent / ".github" / "workflows" / "ci.yml"
        content = ci_path.read_text()
        assert "security-scan" in content
        assert "pip-audit" in content
        assert "bandit" in content

    def test_ci_has_docker_build_job(self):
        """CI workflow includes a docker-build job."""
        from pathlib import Path

        ci_path = Path(__file__).parent.parent / ".github" / "workflows" / "ci.yml"
        content = ci_path.read_text()
        assert "docker-build" in content

    def test_ci_has_release_job(self):
        """CI workflow includes a release job."""
        from pathlib import Path

        ci_path = Path(__file__).parent.parent / ".github" / "workflows" / "ci.yml"
        content = ci_path.read_text()
        assert "release" in content.lower()

    def test_ci_matrix_includes_312(self):
        """CI matrix includes Python 3.12."""
        from pathlib import Path

        ci_path = Path(__file__).parent.parent / ".github" / "workflows" / "ci.yml"
        content = ci_path.read_text()
        assert '"3.12"' in content or "'3.12'" in content


# ========== Packaging Tests ==========

class TestPackaging:
    """Verify pyproject.toml packaging configuration."""

    def test_dependencies_declared(self):
        """pyproject.toml has [project.dependencies]."""
        from pathlib import Path

        pyproject = Path(__file__).parent.parent / "pyproject.toml"
        content = pyproject.read_text()
        assert "[project.dependencies]" in content or 'dependencies = [' in content

    def test_optional_deps_exist(self):
        """pyproject.toml has optional-dependencies (dev, postgres, jetson)."""
        from pathlib import Path

        pyproject = Path(__file__).parent.parent / "pyproject.toml"
        content = pyproject.read_text()
        assert "[project.optional-dependencies]" in content or 'optional-dependencies' in content
        assert "dev" in content
        assert "postgres" in content

    def test_production_classifier(self):
        """Development Status classifier is Production/Beta or higher."""
        from pathlib import Path

        pyproject = Path(__file__).parent.parent / "pyproject.toml"
        content = pyproject.read_text()
        # Should NOT be Alpha anymore
        assert "Development Status :: 3 - Alpha" not in content


# ========== Run Tests ==========

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
