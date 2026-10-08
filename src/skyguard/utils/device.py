"""
Compute device detection and selection.

Goal: be honest about what the host can run.

Priority:
  1. SKYGUARD_DEVICE env or config  (forces a specific device)
  2. CUDA available?                 (Linux/Windows with NVIDIA GPU)
  3. MPS available?                  (Apple Silicon, macOS 12.3+)
  4. CPU fallback                    (always available)

The selected device is exposed as a `torch.device` ready to use.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Optional

from skyguard.core.exceptions import DeviceError


@dataclass(frozen=True)
class DeviceInfo:
    """Snapshot of available compute backends."""

    requested: str               # raw user request ("auto", "cuda", ...)
    selected: str                # resolved device string ("cuda", "mps", "cpu")
    cuda_available: bool
    mps_available: bool
    cpu_count: int
    python_version: str
    torch_version: Optional[str]
    platform: str

    def to_dict(self) -> dict:
        return {
            "requested": self.requested,
            "selected": self.selected,
            "cuda_available": self.cuda_available,
            "mps_available": self.mps_available,
            "cpu_count": self.cpu_count,
            "python_version": self.python_version,
            "torch_version": self.torch_version,
            "platform": self.platform,
        }


def _try_import_torch():
    try:
        import torch  # type: ignore

        return torch, torch.__version__
    except Exception as e:  # pragma: no cover
        raise DeviceError(
            "PyTorch is not installed. Run `make install` or "
            "`pip install -r requirements.txt`."
        ) from e


def _detect_torch_backends() -> tuple[bool, bool, Optional[str]]:
    torch, ver = _try_import_torch()
    cuda_ok = bool(torch.cuda.is_available()) if hasattr(torch, "cuda") else False
    mps_ok = False
    if hasattr(torch, "backends") and hasattr(torch.backends, "mps"):
        try:
            mps_ok = bool(torch.backends.mps.is_available()) and bool(torch.backends.mps.is_built())
        except Exception:  # pragma: no cover - very old torch
            mps_ok = False
    return cuda_ok, mps_ok, ver


def detect_device(requested: str = "auto") -> DeviceInfo:
    """Return a DeviceInfo describing the resolved compute backend.

    Parameters
    ----------
    requested : {"auto","cpu","cuda","mps"}
        ``auto`` picks the best available backend.
    """
    requested = (requested or "auto").lower()
    if requested not in {"auto", "cpu", "cuda", "mps"}:
        raise DeviceError(f"Invalid device '{requested}'. Use one of: auto|cpu|cuda|mps")

    cuda_ok, mps_ok, torch_ver = _detect_torch_backends()
    cpu_count = 1
    try:
        import os

        cpu_count = os.cpu_count() or 1
    except Exception:  # pragma: no cover
        pass

    if requested == "cpu":
        selected = "cpu"
    elif requested == "cuda":
        if not cuda_ok:
            raise DeviceError("CUDA was requested but is not available on this host.")
        selected = "cuda"
    elif requested == "mps":
        if not mps_ok:
            raise DeviceError(
                "MPS was requested but is not available. "
                "Requires Apple Silicon (M1/M2/M3/M4) and macOS >= 12.3."
            )
        selected = "mps"
    else:  # auto
        if cuda_ok:
            selected = "cuda"
        elif mps_ok:
            selected = "mps"
        else:
            selected = "cpu"

    return DeviceInfo(
        requested=requested,
        selected=selected,
        cuda_available=cuda_ok,
        mps_available=mps_ok,
        cpu_count=cpu_count,
        python_version=sys.version.split()[0],
        torch_version=torch_ver,
        platform=sys.platform,
    )


def resolve_torch_device(requested: str = "auto"):
    """Return a ready-to-use ``torch.device`` for the selected backend."""
    info = detect_device(requested)
    import torch  # local import; heavy module

    if info.selected == "cuda":
        # Pin to first visible GPU by default; extend later for multi-GPU.
        return torch.device("cuda:0")
    if info.selected == "mps":
        return torch.device("mps")
    return torch.device("cpu")


def apply_torch_runtime_settings(deterministic: bool, benchmark: bool, matmul_precision: str) -> None:
    """Apply common torch runtime settings once at process start."""
    import torch

    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:  # pragma: no cover
            pass
    torch.backends.cudnn.benchmark = bool(benchmark)
    if hasattr(torch, "set_float32_matmul_precision"):
        try:
            torch.set_float32_matmul_precision(matmul_precision)
        except Exception:  # pragma: no cover
            pass


# --- CLI ---------------------------------------------------------------------


def _print_table(rows: list[tuple[str, str]]) -> None:
    w = max(len(k) for k, _ in rows)
    for k, v in rows:
        print(f"  {k.ljust(w)} : {v}")


def benchmark_cli() -> int:
    """Tiny entry point for `skyguard-bench`."""
    info = detect_device("auto")
    _print_table(
        [
            ("platform", info.platform),
            ("python", info.python_version),
            ("torch", str(info.torch_version)),
            ("cpu_count", str(info.cpu_count)),
            ("cuda", str(info.cuda_available)),
            ("mps", str(info.mps_available)),
            ("selected", info.selected),
        ]
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(benchmark_cli())
