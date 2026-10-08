"""Edge device health monitoring for Jetson.

Reads thermal zones, GPU/CPU utilization, memory, and power draw.
On Jetson, uses jtop (jetson-stats) for richer telemetry; on
non-Jetson hosts falls back to psutil for CPU/memory and returns
Nones for Jetson-specific fields.

The monitor is designed for periodic polling (every 1-5s) and the
EdgeHealth dataclass is JSON-serializable for the SkyGuard API.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import psutil

from skyguard.core.logger import get_logger
from skyguard.edge.hardware import JetsonHardware, detect_jetson

log = get_logger(__name__)


@dataclass
class EdgeHealth:
    """Snapshot of edge device health.

    All temperature fields are in degrees Celsius. Utilization fields
    are percentages (0-100). Power is in milliwatts. ``None`` means
    "not available on this host".
    """
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_total_mb: float
    disk_percent: float
    gpu_percent: Optional[float] = None
    gpu_temp: Optional[float] = None
    cpu_temp: Optional[float] = None
    thermal: Optional[float] = None           # board/PMIC thermal
    power_draw_mw: Optional[float] = None
    jetson_clocks_active: Optional[bool] = None
    power_mode: Optional[int] = None
    available: bool = True
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "memory_used_mb": self.memory_used_mb,
            "memory_total_mb": self.memory_total_mb,
            "disk_percent": self.disk_percent,
            "gpu_percent": self.gpu_percent,
            "gpu_temp": self.gpu_temp,
            "cpu_temp": self.cpu_temp,
            "thermal": self.thermal,
            "power_draw_mw": self.power_draw_mw,
            "jetson_clocks_active": self.jetson_clocks_active,
            "power_mode": self.power_mode,
            "available": self.available,
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------------
# Thermal zone readers (Linux sysfs)
# ---------------------------------------------------------------------------

_THERMAL_BASE = Path("/sys/class/thermal")


def _read_thermal_zone(name: str) -> Optional[float]:
    """Read a thermal zone temperature in Celsius. None if not present."""
    # Try common zone names: thermal_zone0 (CPU), thermal_zone1 (GPU), etc.
    for i in range(8):
        path = _THERMAL_BASE / f"thermal_zone{i}" / "temp"
        type_path = _THERMAL_BASE / f"thermal_zone{i}" / "type"
        if not path.exists():
            continue
        try:
            zone_type = type_path.read_text().strip().lower() if type_path.exists() else ""
            if name.lower() in zone_type or name == f"zone{i}":
                millideg = int(path.read_text().strip())
                return millideg / 1000.0
        except (OSError, ValueError):
            continue
    return None


def _read_hwmon_temp(label: str) -> Optional[float]:
    """Try to read a temperature from /sys/class/hwmon by label."""
    hwmon_base = Path("/sys/class/hwmon")
    if not hwmon_base.exists():
        return None
    for hwmon_dir in hwmon_base.iterdir():
        name_path = hwmon_dir / "name"
        if not name_path.exists():
            continue
        try:
            name = name_path.read_text().strip().lower()
            if label.lower() not in name:
                continue
            for temp_file in hwmon_dir.glob("temp*_input"):
                try:
                    millideg = int(temp_file.read_text().strip())
                    return millideg / 1000.0
                except (OSError, ValueError):
                    continue
        except OSError:
            continue
    return None


# ---------------------------------------------------------------------------
# jtop reader (Jetson-specific)
# ---------------------------------------------------------------------------

def _read_jtop() -> Optional[dict]:
    """Read a snapshot from jtop (jetson-stats). Returns None if unavailable."""
    try:
        from jtop import jtop  # type: ignore
        with jtop() as jetson:
            data: dict = {
                "gpu": None,
                "cpu": None,
                "temperatures": {},
                "power": None,
                "jetson_clocks": None,
                "nvpmodel": None,
            }
            try:
                if "gpu" in jetson:
                    data["gpu"] = jetson.gpu.get("val")
            except Exception:
                pass
            try:
                if "cpu" in jetson:
                    data["cpu"] = jetson.cpu.get("val")
            except Exception:
                pass
            try:
                if "temperatures" in jetson:
                    data["temperatures"] = dict(jetson.temperatures)
            except Exception:
                pass
            try:
                if "power" in jetson:
                    p = jetson.power
                    if isinstance(p, dict) and "power draw" in p:
                        data["power"] = p["power draw"]
            except Exception:
                pass
            try:
                if "jetson_clocks" in jetson:
                    data["jetson_clocks"] = bool(jetson.jetson_clocks)
            except Exception:
                pass
            try:
                if "nvpmodel" in jetson:
                    nm = jetson.nvpmodel
                    if isinstance(nm, dict) and "mode" in nm:
                        data["nvpmodel"] = nm["mode"]
            except Exception:
                pass
            return data
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Edge Monitor
# ---------------------------------------------------------------------------

class EdgeMonitor:
    """Periodic health monitor for Jetson edge devices.

    Usage:
        monitor = EdgeMonitor()
        health = monitor.snapshot()        # one-shot
        async for health in monitor.stream(interval=2.0):
            ...

    Parameters
    ----------
    hardware : JetsonHardware, optional
        Pre-detected hardware snapshot.
    thermal_warning_c : float
        Temperature threshold (°C) above which a warning is emitted.
    thermal_critical_c : float
        Temperature threshold above which a critical warning is emitted.
    """

    def __init__(
        self,
        hardware: Optional[JetsonHardware] = None,
        thermal_warning_c: float = 75.0,
        thermal_critical_c: float = 90.0,
    ) -> None:
        self._hardware = hardware
        self.thermal_warning_c = thermal_warning_c
        self.thermal_critical_c = thermal_critical_c

    @property
    def hardware(self) -> JetsonHardware:
        if self._hardware is None:
            self._hardware = detect_jetson()
        return self._hardware

    def snapshot(self) -> EdgeHealth:
        """Take a single health snapshot."""
        warnings: List[str] = []

        # CPU and memory (always available via psutil)
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        try:
            disk = psutil.disk_usage("/")
            disk_percent = disk.percent
        except Exception:
            disk_percent = 0.0

        gpu_percent: Optional[float] = None
        gpu_temp: Optional[float] = None
        cpu_temp: Optional[float] = None
        thermal: Optional[float] = None
        power_draw_mw: Optional[float] = None
        jc_active: Optional[bool] = None
        power_mode: Optional[int] = None

        # Try jtop first (richest source on Jetson)
        jtop_data = _read_jtop()
        if jtop_data:
            gpu_percent = jtop_data.get("gpu")
            temps = jtop_data.get("temperatures") or {}
            gpu_temp = temps.get("GPU") or temps.get("gpu")
            cpu_temp = temps.get("CPU") or temps.get("cpu")
            thermal = temps.get("thermal") or temps.get("PMIC")
            power_draw_mw = jtop_data.get("power")
            jc_active = jtop_data.get("jetson_clocks")
            power_mode = jtop_data.get("nvpmodel")

        # Fallback: sysfs thermal zones (works on any Linux)
        if gpu_temp is None:
            gpu_temp = _read_thermal_zone("gpu") or _read_hwmon_temp("gpu")
        if cpu_temp is None:
            cpu_temp = _read_thermal_zone("cpu") or _read_hwmon_temp("cpu")
        if thermal is None:
            thermal = _read_thermal_zone("thermal") or _read_thermal_zone("pmic")

        # Fallback: torch GPU utilization (if CUDA available)
        if gpu_percent is None and self.hardware.cuda_available:
            try:
                import torch  # type: ignore
                if torch.cuda.is_available():
                    # No direct util% in torch; use memory as a proxy signal.
                    # Real GPU% requires nvidia-ml-py (pynvml) or jtop.
                    pass
            except Exception:
                pass

        # Thermal warnings
        max_temp = max(
            t for t in [gpu_temp, cpu_temp, thermal] if t is not None
        ) if any([gpu_temp, cpu_temp, thermal]) else None
        if max_temp is not None:
            if max_temp >= self.thermal_critical_c:
                warnings.append(f"CRITICAL: temperature {max_temp:.1f}°C exceeds {self.thermal_critical_c}°C")
            elif max_temp >= self.thermal_warning_c:
                warnings.append(f"WARNING: temperature {max_temp:.1f}°C exceeds {self.thermal_warning_c}°C")

        # Memory warning
        if memory.percent > 90:
            warnings.append(f"WARNING: memory usage {memory.percent:.1f}% > 90%")

        return EdgeHealth(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_mb=memory.used / (1024 * 1024),
            memory_total_mb=memory.total / (1024 * 1024),
            disk_percent=disk_percent,
            gpu_percent=gpu_percent,
            gpu_temp=gpu_temp,
            cpu_temp=cpu_temp,
            thermal=thermal,
            power_draw_mw=power_draw_mw,
            jetson_clocks_active=jc_active,
            power_mode=power_mode,
            available=True,
            warnings=warnings,
        )

    async def stream(self, interval: float = 2.0):
        """Async generator yielding health snapshots at regular intervals.

        Parameters
        ----------
        interval : float
            Seconds between snapshots. Minimum 0.5s to avoid spamming.
        """
        interval = max(0.5, interval)
        while True:
            # Run psutil/jtop in a thread to avoid blocking the event loop
            health = await asyncio.to_thread(self.snapshot)
            yield health
            await asyncio.sleep(interval)


def print_health(health: Optional[EdgeHealth] = None) -> None:
    """Pretty-print a health snapshot (used by CLI)."""
    h = health or EdgeMonitor().snapshot()
    print(f"  CPU Usage         : {h.cpu_percent:.1f}%")
    print(f"  Memory            : {h.memory_percent:.1f}% ({h.memory_used_mb:.0f} / {h.memory_total_mb:.0f} MB)")
    print(f"  Disk              : {h.disk_percent:.1f}%")
    if h.gpu_percent is not None:
        print(f"  GPU Usage         : {h.gpu_percent:.1f}%")
    if h.gpu_temp is not None:
        print(f"  GPU Temp          : {h.gpu_temp:.1f}°C")
    if h.cpu_temp is not None:
        print(f"  CPU Temp          : {h.cpu_temp:.1f}°C")
    if h.thermal is not None:
        print(f"  Board Temp        : {h.thermal:.1f}°C")
    if h.power_draw_mw is not None:
        print(f"  Power Draw        : {h.power_draw_mw:.0f} mW")
    if h.jetson_clocks_active is not None:
        print(f"  Jetson Clocks     : {'ON' if h.jetson_clocks_active else 'OFF'}")
    if h.power_mode is not None:
        print(f"  Power Mode        : {h.power_mode}")
    if h.warnings:
        for w in h.warnings:
            print(f"  ! {w}")
