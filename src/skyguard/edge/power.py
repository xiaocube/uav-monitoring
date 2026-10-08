"""Power mode management for Jetson devices.

Wraps `nvpmodel` and `jetson_clocks` to control power/performance
profiles. On non-Jetson hosts all operations become no-ops with a
warning, so SkyGuard can still start on dev machines.

Typical Jetson power modes (varies by board):
  * 0 = MAXN          (max performance, max power)
  * 1 = MODE_10W      (Nano: 10W low-power)
  * 2 = MODE_15W      (Xavier: 15W)
  * 4 = MODE_30W_ALL  (Xavier: 6-core)
  * 8 = MODE_25W      (Orin NX)
"""
from __future__ import annotations

import enum
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional

from skyguard.core.logger import get_logger
from skyguard.edge.hardware import JetsonHardware, detect_jetson

log = get_logger(__name__)


class PowerMode(enum.IntEnum):
    """Common Jetson nvpmodel modes.

    Values are not universal across all Jetson boards but cover the
    most common ones. Use ``PowerMode.MAXN`` for max performance.
    """
    MAXN = 0             # Maximum performance
    MODE_10W = 1         # 10W (Nano low-power)
    MODE_15W = 2         # 15W (Xavier)
    MODE_30W_6CORE = 4   # 30W 6-core (Xavier)
    MODE_25W = 8         # 25W (Orin NX)
    MODE_20W = 15        # 20W (Orin Nano)


@dataclass(frozen=True)
class PowerState:
    """Current power state snapshot."""
    mode: int
    mode_name: str
    jetson_clocks_active: bool
    available: bool

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "mode_name": self.mode_name,
            "jetson_clocks_active": self.jetson_clocks_active,
            "available": self.available,
        }


def _run_cmd(cmd: list[str], check: bool = False) -> tuple[int, str]:
    """Run a shell command, return (returncode, combined output)."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
            check=check,
        )
        return result.returncode, (result.stdout + result.stderr).strip()
    except FileNotFoundError:
        return 127, f"command not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return 124, f"timeout: {' '.join(cmd)}"
    except Exception as e:
        return 1, str(e)


class PowerManager:
    """Manage Jetson power modes and clock frequencies.

    Parameters
    ----------
    hardware : JetsonHardware, optional
        Pre-detected hardware snapshot. If None, ``detect_jetson()``
        is called lazily on first use.
    dry_run : bool
        If True, commands are logged but not executed (useful for tests
        and planning).
    """

    def __init__(
        self,
        hardware: Optional[JetsonHardware] = None,
        dry_run: bool = False,
    ) -> None:
        self._hardware = hardware
        self._dry_run = dry_run

    @property
    def hardware(self) -> JetsonHardware:
        if self._hardware is None:
            self._hardware = detect_jetson()
        return self._hardware

    @property
    def available(self) -> bool:
        """True if nvpmodel is present on the host."""
        return self.hardware.is_jetson and shutil.which("nvpmodel") is not None

    def get_state(self) -> PowerState:
        """Read current power state. Returns available=False off-Jetson."""
        if not self.available:
            return PowerState(
                mode=-1,
                mode_name="unavailable",
                jetson_clocks_active=False,
                available=False,
            )
        rc, out = _run_cmd(["nvpmodel", "-q"])
        mode = -1
        mode_name = "unknown"
        if rc == 0:
            # Output: "NV Power Mode: MAXN"
            for line in out.splitlines():
                if "Power Mode" in line:
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        mode_name = parts[1].strip()
                    # Try to extract mode number
                    for tok in line.split():
                        if tok.isdigit():
                            mode = int(tok)
                            break
                    break

        # Check jetson_clocks via /home/nvidia/l4t_dfs.conf or sudo jetson_clocks --show
        rc2, out2 = _run_cmd(["jetson_clocks", "--show"])
        jc_active = rc2 == 0 and "ON" in out2.upper()

        return PowerState(
            mode=mode,
            mode_name=mode_name,
            jetson_clocks_active=jc_active,
            available=True,
        )

    def set_mode(self, mode: PowerMode) -> bool:
        """Set nvpmodel power mode. Requires sudo on most boards.

        Returns True if the command succeeded.
        """
        if not self.available:
            log.warning("nvpmodel not available - skipping set_mode({})", mode.name)
            return False

        cmd = ["sudo", "nvpmodel", "-m", str(int(mode))]
        log.info("Setting power mode: {} ({})", mode.name, " ".join(cmd))

        if self._dry_run:
            return True

        rc, out = _run_cmd(cmd)
        if rc != 0:
            log.error("nvpmodel failed: {}", out)
            return False
        return True

    def enable_max_performance(self) -> bool:
        """Convenience: MAXN mode + jetson_clocks (max freq on all cores)."""
        if not self.available:
            log.warning("Jetson power tools not available - skipping max performance")
            return False

        ok = self.set_mode(PowerMode.MAXN)
        if not ok:
            return False

        cmd = ["sudo", "jetson_clocks"]
        log.info("Enabling jetson_clocks (max frequency): {}", " ".join(cmd))

        if self._dry_run:
            return True

        rc, out = _run_cmd(cmd)
        if rc != 0:
            log.error("jetson_clocks failed: {}", out)
            return False
        return True

    def disable_max_performance(self) -> bool:
        """Revert jetson_clocks (allow frequency scaling)."""
        if not self.available:
            return False

        cmd = ["sudo", "jetson_clocks", "--fan"]
        log.info("Disabling jetson_clocks: {}", " ".join(cmd))

        if self._dry_run:
            return True

        rc, out = _run_cmd(cmd)
        return rc == 0

    def recommend_mode(self) -> PowerMode:
        """Recommend a power mode based on hardware family.

        Strategy:
          * Orin / Orin NX  -> MAXN (60W/25W, enough thermal headroom)
          * Xavier          -> MODE_30W_6CORE (balanced)
          * Orin Nano       -> MODE_20W
          * Nano            -> MODE_10W (thermal constrained)
          * Unknown         -> MAXN (let user decide)
        """
        family = self.hardware.family
        if family in {"orin", "orin-nx"}:
            return PowerMode.MAXN
        if family == "xavier":
            return PowerMode.MODE_30W_6CORE
        if family == "orin-nano":
            return PowerMode.MODE_20W
        if family == "nano":
            return PowerMode.MODE_10W
        return PowerMode.MAXN


def print_power_state(pm: Optional[PowerManager] = None) -> None:
    """Pretty-print current power state (used by CLI)."""
    pm = pm or PowerManager()
    state = pm.get_state()
    print(f"  Power Management : {'available' if state.available else 'unavailable'}")
    if state.available:
        print(f"  Current Mode     : {state.mode} ({state.mode_name})")
        print(f"  Jetson Clocks    : {'ON' if state.jetson_clocks_active else 'OFF'}")
        rec = pm.recommend_mode()
        print(f"  Recommended      : {int(rec)} ({rec.name})")
