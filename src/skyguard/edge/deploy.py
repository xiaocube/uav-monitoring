"""Edge deployment orchestration for Jetson.

Ties together hardware detection, power management, TensorRT engine
build, and the inference pipeline to provide a single deployment
manager that knows how to:

  1. Probe the target Jetson device
  2. Set the recommended power mode
  3. Build (or load) a TensorRT engine for the current model
  4. Start the SkyGuard service with the right config
  5. Report health and gracefully shut down

The manager is designed to be driven from a CLI command or from
the SkyGuard API. All operations are idempotent and safe to call
on non-Jetson hosts (they become no-ops with warnings).
"""
from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from skyguard.core.exceptions import ConfigurationError
from skyguard.core.logger import get_logger
from skyguard.edge.hardware import JetsonHardware, detect_jetson
from skyguard.edge.monitor import EdgeHealth, EdgeMonitor
from skyguard.edge.power import PowerManager, PowerMode
from skyguard.edge.trt_engine import (
    BuildResult,
    EngineBuildConfig,
    Precision,
    TRTEngineBuilder,
)

log = get_logger(__name__)


@dataclass
class DeploymentConfig:
    """Configuration for an edge deployment.

    Attributes
    ----------
    model_onnx_path : Path
        Source ONNX model to build the TensorRT engine from.
    model_engine_path : Path
        Where the built engine will be saved / loaded from.
    precision : Precision
        Target inference precision. Use FP16 for most Jetson devices.
    imgsz : int
        Model input size (square).
    max_batch_size : int
        Max batch size for the engine.
    auto_set_power_mode : bool
        If True, deploy() will set the recommended nvpmodel mode.
    auto_enable_jetson_clocks : bool
        If True, deploy() will enable jetson_clocks for max frequency.
    calib_data_dir : Optional[Path]
        INT8 calibration images directory. Required for INT8 precision.
    api_host : str
        Bind address for the SkyGuard API on the edge device.
    api_port : int
        API port.
    """
    model_onnx_path: Path
    model_engine_path: Path
    precision: Precision = Precision.FP16
    imgsz: int = 640
    max_batch_size: int = 1
    auto_set_power_mode: bool = True
    auto_enable_jetson_clocks: bool = True
    calib_data_dir: Optional[Path] = None
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    workspace_gb: int = 4

    def to_dict(self) -> dict:
        return {
            "model_onnx_path": str(self.model_onnx_path),
            "model_engine_path": str(self.model_engine_path),
            "precision": self.precision.value,
            "imgsz": self.imgsz,
            "max_batch_size": self.max_batch_size,
            "auto_set_power_mode": self.auto_set_power_mode,
            "auto_enable_jetson_clocks": self.auto_enable_jetson_clocks,
            "calib_data_dir": str(self.calib_data_dir) if self.calib_data_dir else None,
            "api_host": self.api_host,
            "api_port": self.api_port,
            "workspace_gb": self.workspace_gb,
        }


@dataclass
class DeploymentReport:
    """Result of a deployment operation."""
    hardware: JetsonHardware
    engine_build: Optional[BuildResult]
    power_mode_set: Optional[PowerMode]
    jetson_clocks_enabled: bool
    engine_reused: bool
    deploy_time_sec: float
    warnings: List[str] = field(default_factory=list)
    success: bool = True

    def to_dict(self) -> dict:
        return {
            "hardware": self.hardware.to_dict(),
            "engine_build": self.engine_build.to_dict() if self.engine_build else None,
            "power_mode_set": int(self.power_mode_set) if self.power_mode_set else None,
            "jetson_clocks_enabled": self.jetson_clocks_enabled,
            "engine_reused": self.engine_reused,
            "deploy_time_sec": self.deploy_time_sec,
            "warnings": self.warnings,
            "success": self.success,
        }


class EdgeDeploymentManager:
    """Single-entry-point deployer for SkyGuard on Jetson.

    Typical flow:
        config = DeploymentConfig(
            model_onnx_path=Path("models/best.onnx"),
            model_engine_path=Path("models/best_fp16.engine"),
        )
        manager = EdgeDeploymentManager(config)
        report = manager.deploy()
        if report.success:
            manager.start_api()
    """

    def __init__(
        self,
        config: DeploymentConfig,
        hardware: Optional[JetsonHardware] = None,
        dry_run: bool = False,
    ) -> None:
        self.config = config
        self._hardware = hardware
        self._dry_run = dry_run
        self._builder = TRTEngineBuilder(hardware=hardware)
        self._power = PowerManager(hardware=hardware, dry_run=dry_run)
        self._monitor = EdgeMonitor(hardware=hardware)

    @property
    def hardware(self) -> JetsonHardware:
        if self._hardware is None:
            self._hardware = detect_jetson()
        return self._hardware

    # ------------------------------------------------------------------
    # Pre-flight checks
    # ------------------------------------------------------------------

    def preflight(self) -> List[str]:
        """Run pre-flight checks; return a list of warnings (empty = OK).

        Checks:
          * ONNX source model exists
          * Hardware is Jetson (warning if not)
          * Precision is supported by hardware
          * INT8 has calibration data
          * Disk has enough free space for engine (2x model size)
        """
        warnings: List[str] = []

        if not self.config.model_onnx_path.exists():
            warnings.append(f"ONNX model not found: {self.config.model_onnx_path}")

        if not self.hardware.is_jetson:
            warnings.append(
                f"Host is not a Jetson device (detected: {self.hardware.model}). "
                "Deployment will run in CPU fallback mode."
            )

        if self.config.precision == Precision.INT8:
            if not self.hardware.supports_int8:
                warnings.append(
                    f"INT8 precision not supported on {self.hardware.family}. "
                    "Falling back to FP16."
                )
            elif self.config.calib_data_dir is None or not self.config.calib_data_dir.exists():
                warnings.append(
                    "INT8 precision requires calib_data_dir with calibration images."
                )

        # Disk space check
        try:
            import psutil
            disk = psutil.disk_usage(str(self.config.model_engine_path.parent))
            if disk.free < 2 * 1024 * 1024 * 1024:  # 2 GB
                warnings.append(
                    f"Low disk space: {disk.free / (1024**3):.1f} GB free. "
                    "Engine build may fail."
                )
        except Exception:
            pass

        return warnings

    # ------------------------------------------------------------------
    # Deploy
    # ------------------------------------------------------------------

    def deploy(self, force_rebuild: bool = False) -> DeploymentReport:
        """Execute the full deployment sequence.

        Parameters
        ----------
        force_rebuild : bool
            If True, rebuild the TensorRT engine even if it already exists.

        Returns
        -------
        DeploymentReport
            Full report of what was done.
        """
        t0 = time.perf_counter()
        warnings: List[str] = self.preflight()
        hw = self.hardware

        log.info(
            "Deploying SkyGuard to {} ({})",
            hw.model,
            hw.family if hw.is_jetson else "non-Jetson",
        )

        # 1. Power management (Jetson only)
        power_mode_set: Optional[PowerMode] = None
        jc_enabled = False
        if hw.is_jetson and self.config.auto_set_power_mode:
            power_mode_set = self._power.recommend_mode()
            if not self._power.set_mode(power_mode_set):
                warnings.append(f"Failed to set power mode {power_mode_set.name}")
            elif self.config.auto_enable_jetson_clocks:
                jc_enabled = self._power.enable_max_performance()
                if not jc_enabled:
                    warnings.append("Failed to enable jetson_clocks")

        # 2. Engine build / load
        engine_build: Optional[BuildResult] = None
        engine_reused = False

        if self.config.model_engine_path.exists() and not force_rebuild:
            log.info("Reusing existing engine: {}", self.config.model_engine_path)
            engine_reused = True
        elif self._builder.available:
            # Adjust precision if INT8 not supported
            precision = self.config.precision
            if precision == Precision.INT8 and not hw.supports_int8:
                precision = Precision.FP16
                warnings.append("INT8 unsupported on this hardware - using FP16")

            build_cfg = EngineBuildConfig(
                precision=precision,
                imgsz=self.config.imgsz,
                max_batch_size=self.config.max_batch_size,
                workspace_gb=self.config.workspace_gb,
                calib_data_dir=self.config.calib_data_dir,
            )
            try:
                engine_build = self._builder.build(
                    onnx_path=self.config.model_onnx_path,
                    output_path=self.config.model_engine_path,
                    config=build_cfg,
                )
            except Exception as e:
                warnings.append(f"Engine build failed: {e}")
                log.error("Engine build failed: {}", e)
        else:
            if not hw.is_jetson:
                warnings.append(
                    "TensorRT unavailable on non-Jetson host - skipping engine build. "
                    "Use ONNX or PyTorch backend instead."
                )

        success = engine_reused or engine_build is not None or not hw.is_jetson

        return DeploymentReport(
            hardware=hw,
            engine_build=engine_build,
            power_mode_set=power_mode_set,
            jetson_clocks_enabled=jc_enabled,
            engine_reused=engine_reused,
            deploy_time_sec=time.perf_counter() - t0,
            warnings=warnings,
            success=success,
        )

    # ------------------------------------------------------------------
    # API server
    # ------------------------------------------------------------------

    def start_api(self, block: bool = True) -> Optional[subprocess.Popen]:
        """Start the SkyGuard API server on the edge device.

        Parameters
        ----------
        block : bool
            If True, runs uvicorn in the current process (blocking).
            If False, spawns a subprocess and returns the Popen handle.

        Returns
        -------
        Optional[subprocess.Popen]
            None if blocking; Popen handle if non-blocking.
        """
        log.info(
            "Starting SkyGuard API on {}:{}", self.config.api_host, self.config.api_port
        )

        env = {
            **dict(__import__("os").environ),
            "SKYGUARD_WEB__HOST": self.config.api_host,
            "SKYGUARD_WEB__PORT": str(self.config.api_port),
            "SKYGUARD_EDGE__TARGET": "jetson" if self.hardware.is_jetson else "cpu",
        }

        if block:
            import uvicorn
            from skyguard.api.app import create_app
            app = create_app()
            uvicorn.run(
                app,
                host=self.config.api_host,
                port=self.config.api_port,
                log_level="info",
            )
            return None

        # Non-blocking: spawn subprocess
        cmd = [
            "python", "-m", "uvicorn",
            "skyguard.api.app:create_app",
            "--factory",
            "--host", self.config.api_host,
            "--port", str(self.config.api_port),
        ]
        return subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def health(self) -> EdgeHealth:
        """Convenience: take a health snapshot."""
        return self._monitor.snapshot()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Graceful shutdown: revert jetson_clocks, leave power mode as-is."""
        if self.hardware.is_jetson and self.config.auto_enable_jetson_clocks:
            log.info("Disabling jetson_clocks on shutdown")
            self._power.disable_max_performance()
        log.info("Edge deployment shut down")
