from __future__ import annotations

import io
import logging
import sys

from loguru import logger

from skyguard.core.logger import get_logger, setup_logging


def test_setup_logging_is_idempotent(tmp_path, monkeypatch):
    # Point logs to tmp so we don't pollute the repo
    monkeypatch.chdir(tmp_path)
    setup_logging()
    setup_logging()  # second call must be a no-op
    logger.info("hello from test")
    log_file = tmp_path / "logs" / "skyguard.log"
    assert log_file.exists()


def test_get_logger_returns_bound_logger(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    log = get_logger("unit.test")
    # loguru's logger.bind returns a new BoundLogger
    assert hasattr(log, "info")
    assert hasattr(log, "error")
    # Should not raise
    log.info("test message")


def test_intercept_routes_stdlib(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    setup_logging()
    stdlib_log = logging.getLogger("some.lib")
    stdlib_log.setLevel(logging.INFO)
    stdlib_log.info("from stdlib")
    captured = capsys.readouterr()
    # loguru prints to stderr; allow either but assert non-empty
    err = captured.err
    assert "from stdlib" in err or "from stdlib" in captured.out


def test_quiet_loggers_are_silenced(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    setup_logging()
    noisy = logging.getLogger("ultralytics")
    assert noisy.level >= logging.WARNING
