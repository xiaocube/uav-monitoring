"""Jetson hardware detection and capability probing.

Detects Jetson device family (Nano / Xavier / Orin), GPU capabilities,
memory, CUDA cores, and TensorRT availability. Falls back gracefully
on non-Jetson hosts (returns is_jetson=False) so the rest of the
SkyGuard stack keeps working on dev machines.

Detection signals:
  1. /proc/device-tree/model (contains "NVIDIA Jetson")
  2. /etc/nv_tegra_release file existence
  3. jtop / jetson_stats Python package
  4. CUDA + aarch64 architecture (strong hint)
"""
from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from skyguard.core.logger import get_logger

log = get_logger(__name__)


# Known Jetson device families and their characteristics.
# Source: NVIDIA Jetson Orin / Xavier / Nano datasheets.
JETSON_MODELS: Dict[str, Dict[str, int]] = {
    "nano":   {"cuda_cores": 128,  "tensor_cores": 0,   "memory_mb": 4096,  "max_power_w": 10},
    "tx2":    {"cuda_cores": 256,  "tensor_cores": 0,   "memory_mb": 8192,  "max_power_w": 15},
    "xavier": {"cuda_cores": 512,  "tensor_cores": 64,  "memory_mb": 8192,  "max_power_w": 30},
    "orin":   {"cuda_cores": 2048, "tensor_cores": 64,  "memory_mb": 8192,  "max_power_w": 60},
    "orin-nx": {"cuda_cores": 1024, "tensor_cores": 32, "memory_mb": 8192,  "max_power_w": 25},
    "orin-nano": {"cuda_cores": 1024, "tensor_cores": 32, "memory_mb": 4096, "max_power_w": 15},
}


@dataclass(frozen=True)
class JetsonHardware:
    """Snapshot of Jetson hardware capabilities.

    On non-Jetson hosts, ``is_jetson`` is False and other fields are
    populated with best-effort values from the host OS.
    """
    is_jetson: bool
    model: str                    # e.g. "NVIDIA Jetson Orin NX 16GB"
    family: str                   # e.g. "orin-nx" (key into JETSON_MODELS)
    cuda_cores: int
    tensor_cores: int
    memory_mb: int
    max_power_w: int
    arch: str                     # e.g. "aarch64"
    cuda_available: bool
    cuda_version: Optional[str]   # e.g. "12.2"
    tensorrt_available: bool
    tensorrt_version: Optional[str]
    jetpack_version: Optional[str]
    gpu_name: Optional[str]

    def to_dict(self) -> dict:
        return {
            "is_jetson": self.is_jetson,
            "model": self.model,
            "family": self.family,
            "cuda_cores": self.cuda_cores,
            "tensor_cores": self.tensor_cores,
            "memory_mb": self.memory_mb,
            "max_power_w": self.max_power_w,
            "arch": self.arch,
            "cuda_available": self.cuda_available,
            "cuda_version": self.cuda_version,
            "tensorrt_available": self.tensorrt_available,
            "tensorrt_version": self.tensorrt_version,
            "jetpack_version": self.jetpack_version,
            "gpu_name": self.gpu_name,
        }

    @property
    def supports_fp16(self) -> bool:
        """All Jetson devices with Tensor Cores support FP16."""
        return self.tensor_cores > 0

    @property
    def supports_int8(self) -> bool:
        """Jetson Xavier / Orin support INT8; Nano / TX2 do not."""
        return self.family in {"xavier", "orin", "orin-nx"}


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def _read_device_tree_model() -> Optional[str]:
    """Read /proc/device-tree/model on Linux; None if not present."""
    path = Path("/proc/device-tree/model")
    if not path.exists():
        return None
    try:
        return path.read_text(errors="ignore").strip().rstrip("\x00")
    except OSError:
        return None


def _read_nv_tegra_release() -> Optional[str]:
    """Return JetPack version from /etc/nv_tegra_release."""
    path = Path("/etc/nv_tegra_release")
    if not path.exists():
        return None
    try:
        text = path.read_text(errors="ignore")
        # Format: "# R35 (release), REVISION: 3.1, ..."
        for line in text.splitlines():
            if line.startswith("# R") and "REVISION" in line:
                parts = line.split(",")
                if parts:
                    return parts[0].replace("# ", "").strip()
        return "unknown"
    except OSError:
        return None


def _detect_family(model_str: str) -> str:
    """Map model string to JETSON_MODELS key."""
    s = model_str.lower()
    if "orin nano" in s:
        return "orin-nano"
    if "orin nx" in s or "orin-nx" in s:
        return "orin-nx"
    if "orin" in s:
        return "orin"
    if "xavier" in s:
        return "xavier"
    if "tx2" in s:
        return "tx2"
    if "nano" in s:
        return "nano"
    return "unknown"


def _query_cuda() -> tuple[bool, Optional[str], Optional[str]]:
    """Return (available, version, gpu_name) via torch if importable."""
    try:
        import torch  # type: ignore
    except Exception:
        return False, None, None

    cuda_ok = bool(torch.cuda.is_available())
    version = None
    gpu_name = None
    if cuda_ok:
        try:
            version = torch.version.cuda
        except Exception:
            version = "unknown"
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = None
    return cuda_ok, version, gpu_name


def _query_tensorrt() -> tuple[bool, Optional[str]]:
    """Return (available, version) for TensorRT."""
    try:
        import tensorrt as trt  # type: ignore
        return True, trt.__version__
    except Exception:
        return False, None


def _try_jetson_stats() -> Optional[dict]:
    """Try jtop / jetson-stats for richer info; None if unavailable."""
    try:
        from jtop import jtop  # type: ignore
        with jtop() as jetson:
            return jetson.hardware
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_jetson() -> JetsonHardware:
    """Detect Jetson hardware capabilities.

    On macOS or non-Jetson Linux this returns a JetsonHardware with
    ``is_jetson=False`` but still populates CUDA / TensorRT availability
    so downstream code can decide what to do.
    """
    arch = platform.machine()
    model_str = _read_device_tree_model() or platform.processor() or "unknown"
    jetpack = _read_nv_tegra_release()

    # Strong signal: device-tree model contains "NVIDIA Jetson"
    is_jetson = "nvidia jetson" in model_str.lower() or jetpack is not None

    # Even on non-Jetson, query CUDA / TensorRT for dev-machine reporting
    cuda_ok, cuda_ver, gpu_name = _query_cuda()
    trt_ok, trt_ver = _query_tensorrt()

    family = _detect_family(model_str) if is_jetson else "unknown"
    specs = JETSON_MODELS.get(family, {})

    # Try jetson-stats for richer info (overrides defaults if present)
    jtop_hw = _try_jetson_stats()
    if jtop_hw:
        is_jetson = True
        if "model" in jtop_hw:
            model_str = jtop_hw["model"]
            family = _detect_family(model_str)
            specs = JETSON_MODELS.get(family, specs)

    return JetsonHardware(
        is_jetson=is_jetson,
        model=model_str,
        family=family,
        cuda_cores=specs.get("cuda_cores", 0),
        tensor_cores=specs.get("tensor_cores", 0),
        memory_mb=specs.get("memory_mb", 0),
        max_power_w=specs.get("max_power_w", 0),
        arch=arch,
        cuda_available=cuda_ok,
        cuda_version=cuda_ver,
        tensorrt_available=trt_ok,
        tensorrt_version=trt_ver,
        jetpack_version=jetpack,
        gpu_name=gpu_name,
    )


def print_hardware_info(hw: Optional[JetsonHardware] = None) -> None:
    """Pretty-print hardware info to stdout (used by CLI)."""
    hw = hw or detect_jetson()
    print(f"  Platform         : {hw.arch}")
    print(f"  Model            : {hw.model}")
    print(f"  Is Jetson        : {hw.is_jetson}")
    if hw.is_jetson:
        print(f"  Family           : {hw.family}")
        print(f"  CUDA Cores       : {hw.cuda_cores}")
        print(f"  Tensor Cores     : {hw.tensor_cores}")
        print(f"  Memory           : {hw.memory_mb} MB")
        print(f"  Max Power        : {hw.max_power_w} W")
        print(f"  JetPack          : {hw.jetpack_version or 'unknown'}")
        print(f"  Supports FP16    : {hw.supports_fp16}")
        print(f"  Supports INT8    : {hw.supports_int8}")
    print(f"  CUDA Available   : {hw.cuda_available}")
    print(f"  CUDA Version     : {hw.cuda_version or 'N/A'}")
    print(f"  TensorRT         : {hw.tensorrt_available}")
    print(f"  TensorRT Version : {hw.tensorrt_version or 'N/A'}")
    print(f"  GPU Name         : {hw.gpu_name or 'N/A'}")


if __name__ == "__main__":  # pragma: no cover
    print_hardware_info()
