from __future__ import annotations

from skyguard.utils.environment import run_checks


def test_run_checks_returns_report():
    report = run_checks()
    # every check must have a name
    assert all(r.name for r in report.results)
    # at least the core ML imports were tested
    names = {r.name for r in report.results}
    assert "import:torch" in names
    assert "import:opencv" in names
    assert "import:ultralytics" in names
