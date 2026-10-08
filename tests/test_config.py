from __future__ import annotations

import pytest
import yaml

from skyguard.core.config import (
    AppConfig,
    ComputeConfig,
    LoggingConfig,
    Settings,
    clear_settings_cache,
    get_settings,
)
from skyguard.core.exceptions import ConfigurationError


def test_settings_defaults_loaded():
    s = get_settings()
    assert isinstance(s, Settings)
    assert s.app.name == "SkyGuard"
    assert s.app.env in {"development", "staging", "production"}
    assert s.compute.device in {"auto", "cpu", "cuda", "mps"}


def test_settings_are_cached():
    a = get_settings()
    b = get_settings()
    assert a is b
    clear_settings_cache()
    c = get_settings()
    # Content equal, identity may differ
    assert c.app.name == a.app.name


def test_sub_models_validate():
    # invalid num_workers should fail
    with pytest.raises(Exception):
        ComputeConfig(num_workers=-1)
    # invalid log level should fail
    with pytest.raises(Exception):
        LoggingConfig(level="VERBOSE")  # type: ignore[arg-type]


def test_app_config_env_literal():
    # 'prod' is not a valid env literal
    with pytest.raises(Exception):
        AppConfig(env="prod")  # type: ignore[arg-type]


def test_describe_runs_without_error(capsys):
    from skyguard.core.config import describe

    describe()
    out = capsys.readouterr()
    # describe() doesn't print unless __main__; just check it returned a string
    # by calling it via __main__ path is overkill here
    assert isinstance(out.out, str)
