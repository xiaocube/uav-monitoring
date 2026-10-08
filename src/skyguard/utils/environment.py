"""
Environment verification utilities.

Used by:
  * scripts/verify_install.py
  * the `skyguard-verify` console script
  * future /health endpoint
"""
from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, field
from typing import List, Tuple

from skyguard.core.config import describe, get_settings
from skyguard.core.logger import get_logger, setup_logging
from skyguard.utils.device import detect_device

log = get_logger(__name__)


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        flag = "OK " if self.ok else "FAIL"
        return f"[{flag}] {self.name:<30} {self.detail}"


@dataclass
class VerifyReport:
    results: List[CheckResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(r.ok for r in self.results)

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.results.append(CheckResult(name=name, ok=ok, detail=detail))

    def pretty(self) -> str:
        return "\n".join(str(r) for r in self.results)


# --- individual checks ------------------------------------------------------


def _check_import(name: str, import_path: str, min_version_attr: str = "__version__") -> Tuple[bool, str]:
    try:
        mod = importlib.import_module(import_path)
        ver = getattr(mod, min_version_attr, "unknown")
        return True, f"version={ver}"
    except Exception as e:
        return False, f"import failed: {e}"


def _check_paths() -> List[CheckResult]:
    s = get_settings()
    out: List[CheckResult] = []
    for label, p in [
        ("paths.data_dir", s.paths.data_dir),
        ("paths.model_dir", s.paths.model_dir),
        ("paths.log_dir", s.paths.log_dir),
    ]:
        try:
            p.mkdir(parents=True, exist_ok=True)
            out.append(CheckResult(name=label, ok=True, detail=f"writable at {p}"))
        except Exception as e:
            out.append(CheckResult(name=label, ok=False, detail=str(e)))
    return out


def run_checks() -> VerifyReport:
    report = VerifyReport()

    # 1. core ML stack
    for name, mod in [
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("opencv", "cv2"),
        ("ultralytics", "ultralytics"),
        ("numpy", "numpy"),
        ("Pillow", "PIL"),
        ("yaml", "yaml"),
        ("pydantic", "pydantic"),
        ("loguru", "loguru"),
        ("rich", "rich"),
    ]:
        ok, detail = _check_import(name, mod)
        report.add(f"import:{name}", ok, detail)

    # 2. compute device
    try:
        info = detect_device("auto")
        report.add("device.auto", True, f"selected={info.selected} cuda={info.cuda_available} mps={info.mps_available}")
    except Exception as e:  # pragma: no cover
        report.add("device.auto", False, str(e))

    # 3. filesystem
    for r in _check_paths():
        report.results.append(r)

    # 4. settings
    try:
        _ = describe()
        report.add("settings", True, "loaded successfully")
    except Exception as e:
        report.add("settings", False, str(e))

    return report


def verify_cli() -> int:
    """Console-script entry point: `skyguard-verify`."""
    setup_logging()
    log.info("Running SkyGuard environment verification")
    report = run_checks()
    print(report.pretty())
    print()
    print("Summary:", "PASS" if report.ok else "FAIL")
    return 0 if report.ok else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(verify_cli())
