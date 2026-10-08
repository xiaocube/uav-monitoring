"""Database configuration types."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class DatabaseConfig(BaseModel):
    """Database connection configuration."""
    
    host: str = Field("localhost", description="Database host")
    port: int = Field(5432, description="Database port")
    database: str = Field("skyguard", description="Database name")
    username: str = Field("skyguard", description="Database username")
    password: str = Field("", description="Database password")
    ssl_mode: str = Field("prefer", description="SSL mode")
    pool_size: int = Field(10, description="Connection pool size")
    max_overflow: int = Field(20, description="Max pool overflow")
    connect_timeout: int = Field(10, description="Connection timeout in seconds")
    
    @property
    def dsn(self) -> str:
        """Database connection DSN."""
        return (
            f"postgresql://{self.username}:{self.password}@"
            f"{self.host}:{self.port}/{self.database}"
            f"?sslmode={self.ssl_mode}"
        )
    
    @property
    def async_dsn(self) -> str:
        """Async database connection DSN."""
        return (
            f"postgresql+asyncpg://{self.username}:{self.password}@"
            f"{self.host}:{self.port}/{self.database}"
            f"?sslmode={self.ssl_mode}"
        )


class TimescaleConfig(BaseModel):
    """TimescaleDB configuration."""
    
    enabled: bool = Field(True, description="Enable TimescaleDB")
    chunk_interval: str = Field("1 day", description="Hypertable chunk interval")
    compression_enabled: bool = Field(False, description="Enable compression")
    compression_interval: str = Field("7 days", description="Compression interval")