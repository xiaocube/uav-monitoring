"""
Standalone verification entrypoint.

Usage:
  python scripts/verify_install.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running this file from the project root without installing the package.
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from skyguard.utils.environment import run_checks  # noqa: E402


def main() -> int:
    report = run_checks()
    print(report.pretty())
    print()
    print("Summary:", "PASS" if report.ok else "FAIL")
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
