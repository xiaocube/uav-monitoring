"""Database CLI commands."""

from __future__ import annotations

import typer
from loguru import logger

from skyguard.db import DatabaseConfig, DatabaseManager


db_cli = typer.Typer(name="db", help="Database management commands")


@db_cli.command(name="init")
def db_init(
    host: str = typer.Option("localhost", "--host", help="Database host"),
    port: int = typer.Option(5432, "--port", help="Database port"),
    database: str = typer.Option("skyguard", "--database", help="Database name"),
    username: str = typer.Option("skyguard", "--username", help="Database username"),
    password: str = typer.Option("", "--password", help="Database password"),
):
    """Initialize database and create tables."""
    config = DatabaseConfig(
        host=host,
        port=port,
        database=database,
        username=username,
        password=password,
    )

    db = DatabaseManager(config)
    if not db.connect():
        logger.error("Failed to connect to database")
        raise typer.Exit(code=1)

    try:
        db.create_tables()
        logger.success("Database initialized successfully")
    finally:
        db.disconnect()


@db_cli.command(name="connect")
def db_connect(
    host: str = typer.Option("localhost", "--host", help="Database host"),
    port: int = typer.Option(5432, "--port", help="Database port"),
    database: str = typer.Option("skyguard", "--database", help="Database name"),
    username: str = typer.Option("skyguard", "--username", help="Database username"),
    password: str = typer.Option("", "--password", help="Database password"),
):
    """Test database connection."""
    config = DatabaseConfig(
        host=host,
        port=port,
        database=database,
        username=username,
        password=password,
    )

    db = DatabaseManager(config)
    if db.connect():
        logger.success("Database connection successful")
        db.disconnect()
    else:
        logger.error("Database connection failed")
        raise typer.Exit(code=1)


@db_cli.command(name="cleanup")
def db_cleanup(
    host: str = typer.Option("localhost", "--host", help="Database host"),
    port: int = typer.Option(5432, "--port", help="Database port"),
    database: str = typer.Option("skyguard", "--database", help="Database name"),
    username: str = typer.Option("skyguard", "--username", help="Database username"),
    password: str = typer.Option("", "--password", help="Database password"),
    days: int = typer.Option(30, "--days", help="Days to keep data"),
):
    """Clean up old data."""
    config = DatabaseConfig(
        host=host,
        port=port,
        database=database,
        username=username,
        password=password,
    )

    db = DatabaseManager(config)
    if not db.connect():
        logger.error("Failed to connect to database")
        raise typer.Exit(code=1)

    try:
        result = db.cleanup_old_data(days_to_keep=days)
        logger.success(
            f"Cleanup completed: {result['detections_deleted']} detections, "
            f"{result['tracks_deleted']} tracks deleted"
        )
    finally:
        db.disconnect()


@db_cli.command(name="stats")
def db_stats(
    host: str = typer.Option("localhost", "--host", help="Database host"),
    port: int = typer.Option(5432, "--port", help="Database port"),
    database: str = typer.Option("skyguard", "--database", help="Database name"),
    username: str = typer.Option("skyguard", "--username", help="Database username"),
    password: str = typer.Option("", "--password", help="Database password"),
    hours: int = typer.Option(24, "--hours", help="Hours to query"),
):
    """Get database statistics."""
    config = DatabaseConfig(
        host=host,
        port=port,
        database=database,
        username=username,
        password=password,
    )

    db = DatabaseManager(config)
    if not db.connect():
        logger.error("Failed to connect to database")
        raise typer.Exit(code=1)

    try:
        det_stats = db.get_detection_statistics(hours=hours)
        alert_stats = db.get_alert_statistics(hours=hours)

        logger.info("Detection Statistics:")
        logger.info(f"  Total: {det_stats['total_detections']}")
        for cls in det_stats.get("by_class", []):
            logger.info(f"  - {cls['class_name']}: {cls['count']}")

        logger.info("\nAlert Statistics:")
        logger.info(f"  Total: {alert_stats['total_alerts']}")
        logger.info(f"  Info: {alert_stats['by_level']['info']}")
        logger.info(f"  Warning: {alert_stats['by_level']['warning']}")
        logger.info(f"  Critical: {alert_stats['by_level']['critical']}")
    finally:
        db.disconnect()
