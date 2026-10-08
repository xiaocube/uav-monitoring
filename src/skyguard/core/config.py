"""
Configuration system for SkyGuard.

Design principles:
  * Single source of truth (Pydantic v2 + pydantic-settings)
  * Layered resolution: defaults < YAML file < env-specific YAML < environment variables
  * Strict validation; surface errors loudly at startup, not silently in production
  * Re-importable: get_settings() is cached but the cache can be cleared in tests

Resolution order (later overrides earlier):
  1. config/default.yaml           -- baseline
  2. config/{env}.yaml             -- per-environment overrides
  3. SKYGUARD_* environment vars  -- runtime overrides
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from skyguard.core.exceptions import ConfigurationError

# --- Sub-models -------------------------------------------------------------


class AppConfig(BaseModel):
    name: str = "SkyGuard"
    version: str = "0.1.0"
    env: Literal["development", "staging", "production"] = "development"
    owner: str = "SkyGuard Engineering"


class PathsConfig(BaseModel):
    data_dir: Path = Path("./data")
    model_dir: Path = Path("./models")
    log_dir: Path = Path("./logs")
    config_dir: Path = Path("./config")


class ComputeConfig(BaseModel):
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    num_workers: int = 4
    benchmark: bool = False
    deterministic: bool = True
    matmul_precision: Literal["highest", "high", "medium"] = "high"

    @field_validator("num_workers")
    @classmethod
    def _validate_workers(cls, v: int) -> int:
        if v < 0:
            raise ValueError("num_workers must be >= 0")
        return v


class LoggingConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    rotation: str = "20 MB"
    retention: str = "14 days"
    colorize: bool = True
    serialize_json: bool = False
    enqueue: bool = True


class DetectionConfig(BaseModel):
    enabled: bool = False
    model_name: str = "yolov8n.pt"
    imgsz: int = 640
    conf: float = 0.25
    iou: float = 0.45
    classes: List[int] = Field(default_factory=list)


class TrackingConfig(BaseModel):
    enabled: bool = False
    algorithm: Literal["bytetrack", "deepsort", "botsort"] = "bytetrack"
    track_buffer: int = 30


class PTZConfig(BaseModel):
    enabled: bool = False
    protocol: Literal["onvif", "visca", "pelco_d"] = "onvif"


class WebConfig(BaseModel):
    enabled: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: List[str] = Field(default_factory=lambda: ["http://localhost:5173"])


class DatabaseConfig(BaseModel):
    enabled: bool = False
    host: str = "localhost"
    port: int = 5432
    database: str = "skyguard"
    username: str = "skyguard"
    password: str = ""
    pool_size: int = 10
    echo: bool = False


class EdgeConfig(BaseModel):
    target: Literal["cpu", "jetson", "coreml"] = "cpu"


class SecurityConfig(BaseModel):
    """Security and API access control settings (Sprint 12)."""
    enabled: bool = False
    api_key: str = ""
    api_key_header: str = "X-API-Key"
    # Paths that skip auth (health, docs, metrics, dashboard)
    public_paths: List[str] = Field(
        default_factory=lambda: ["/", "/health", "/ready", "/docs", "/redoc", "/openapi.json", "/metrics"]
    )
    # Rate limiting (requests per minute per client)
    rate_limit_enabled: bool = True
    rate_limit_per_minute: int = 60
    # Trusted hosts (empty = allow all)
    trusted_hosts: List[str] = Field(default_factory=list)
    # GZip compression
    gzip_enabled: bool = True
    gzip_min_size: int = 500


# --- Top-level Settings ------------------------------------------------------


class Settings(BaseSettings):
    """Top-level application settings.

    This object is normally created via get_settings(). Direct construction is
    supported for tests; in that case YAML loading is skipped.
    """

    app: AppConfig = AppConfig()
    paths: PathsConfig = PathsConfig()
    compute: ComputeConfig = ComputeConfig()
    logging: LoggingConfig = LoggingConfig()
    detection: DetectionConfig = DetectionConfig()
    tracking: TrackingConfig = TrackingConfig()
    ptz: PTZConfig = PTZConfig()
    web: WebConfig = WebConfig()
    database: DatabaseConfig = DatabaseConfig()
    edge: EdgeConfig = EdgeConfig()
    security: SecurityConfig = SecurityConfig()

    model_config = SettingsConfigDict(
        env_prefix="SKYGUARD_",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    # ----------------------------- helpers ----------------------------------

    def env_file_targets(self) -> List[str]:
        """Return a human-friendly list of paths the loader will look at."""
        return [str(self.paths.config_dir / "default.yaml")]


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ConfigurationError(f"YAML root in {path} must be a mapping, got {type(data).__name__}")
        return data
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Failed to parse YAML file {path}: {e}") from e


def _merge(base: dict, override: dict) -> dict:
    """Deep merge two dicts. Values in `override` win."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _resolve_config_dir() -> Path:
    """Find the config dir using SKYGUARD_CONFIG_DIR env or CWD default."""
    return Path(os.getenv("SKYGUARD_CONFIG_DIR", "./config")).resolve()


def _build_settings() -> Settings:
    cfg_dir = _resolve_config_dir()
    env_name = os.getenv("SKYGUARD_ENV", "development").lower()

    merged: dict = {}
    default_path = cfg_dir / "default.yaml"
    merged = _merge(merged, _load_yaml(default_path))

    env_path = cfg_dir / f"{env_name}.yaml"
    if env_path.exists():
        merged = _merge(merged, _load_yaml(env_path))

    if not merged:
        raise ConfigurationError(
            f"No configuration found. Looked at {default_path}. "
            "Set SKYGUARD_CONFIG_DIR or run from project root."
        )

    try:
        return Settings(**merged)
    except Exception as e:  # Pydantic ValidationError or similar
        raise ConfigurationError(f"Invalid SkyGuard configuration: {e}") from e


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""
    return _build_settings()


settings = get_settings()


def clear_settings_cache() -> None:
    """Used in tests to force re-reading YAML."""
    get_settings.cache_clear()


# --- CLI helper --------------------------------------------------------------


def describe() -> str:
    """Return a human-readable description of the current settings (for CLI/UI)."""
    s = get_settings()
    lines = [
        f"SkyGuard {s.app.version} ({s.app.env})",
        f"  device     : {s.compute.device}",
        f"  log_level  : {s.logging.level}",
        f"  data_dir   : {s.paths.data_dir}",
        f"  model_dir  : {s.paths.model_dir}",
        f"  log_dir    : {s.paths.log_dir}",
        f"  detection  : enabled={s.detection.enabled} model={s.detection.model_name}",
        f"  tracking   : enabled={s.tracking.enabled} algo={s.tracking.algorithm}",
        f"  web        : enabled={s.web.enabled} {s.web.host}:{s.web.port}",
        f"  database   : enabled={s.database.enabled}",
        f"  security   : enabled={s.security.enabled}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
