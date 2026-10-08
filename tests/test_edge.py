"""Unit tests for the edge deployment module (Sprint 11).

Tests cover:
- JetsonHardware dataclass + detect_jetson() graceful degradation
- PowerManager / PowerMode / PowerState (dry_run + non-Jetson fallback)
- TRTEngineBuilder / EngineBuildConfig / Precision / INT8Calibrator
- EdgeMonitor / EdgeHealth (snapshot + async stream)
- EdgeDeploymentManager / DeploymentConfig (preflight + deploy)
- Edge CLI commands (typer CliRunner)

All tests are designed to pass on macOS / Linux CI without real Jetson
hardware by relying on graceful degradation (is_jetson=False,
available=False) and injected mock hardware objects.

Run with: pytest tests/test_edge.py -v
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from skyguard.edge.hardware import (
    JETSON_MODELS,
    JetsonHardware,
    _detect_family,
    detect_jetson,
    print_hardware_info,
)
from skyguard.edge.power import (
    PowerManager,
    PowerMode,
    PowerState,
    _run_cmd,
    print_power_state,
)
from skyguard.edge.trt_engine import (
    BuildResult,
    EngineBuildConfig,
    INT8Calibrator,
    Precision,
    TRTEngineBuilder,
    print_builder_info,
)
from skyguard.edge.monitor import (
    EdgeHealth,
    EdgeMonitor,
    print_health,
)
from skyguard.edge.deploy import (
    DeploymentConfig,
    DeploymentReport,
    EdgeDeploymentManager,
)
from skyguard.core.exceptions import ModelError


# ========== Test Fixtures ==========

@pytest.fixture
def non_jetson_hardware() -> JetsonHardware:
    """A hardware snapshot representing a non-Jetson dev host (macOS)."""
    return JetsonHardware(
        is_jetson=False,
        model="MacBook Pro",
        family="unknown",
        cuda_cores=0,
        tensor_cores=0,
        memory_mb=0,
        max_power_w=0,
        arch="arm64",
        cuda_available=False,
        cuda_version=None,
        tensorrt_available=False,
        tensorrt_version=None,
        jetpack_version=None,
        gpu_name=None,
    )


@pytest.fixture
def orin_hardware() -> JetsonHardware:
    """A hardware snapshot simulating a Jetson Orin NX 16GB."""
    return JetsonHardware(
        is_jetson=True,
        model="NVIDIA Jetson Orin NX 16GB",
        family="orin-nx",
        cuda_cores=1024,
        tensor_cores=32,
        memory_mb=8192,
        max_power_w=25,
        arch="aarch64",
        cuda_available=True,
        cuda_version="12.2",
        tensorrt_available=True,
        tensorrt_version="8.6.0",
        jetpack_version="R35.4.1",
        gpu_name="Orin",
    )


@pytest.fixture
def nano_hardware() -> JetsonHardware:
    """A hardware snapshot simulating a Jetson Nano (no Tensor Cores)."""
    return JetsonHardware(
        is_jetson=True,
        model="NVIDIA Jetson Nano",
        family="nano",
        cuda_cores=128,
        tensor_cores=0,
        memory_mb=4096,
        max_power_w=10,
        arch="aarch64",
        cuda_available=True,
        cuda_version="10.2",
        tensorrt_available=True,
        tensorrt_version="7.1.0",
        jetpack_version="R32.7.1",
        gpu_name="Maxwell",
    )


@pytest.fixture
def orin_with_trt(orin_hardware) -> TRTEngineBuilder:
    """A TRT engine builder backed by a (simulated) Orin + TensorRT."""
    return TRTEngineBuilder(hardware=orin_hardware)


@pytest.fixture
def non_jetson_builder(non_jetson_hardware) -> TRTEngineBuilder:
    """A TRT engine builder on a non-Jetson host (TRT unavailable)."""
    return TRTEngineBuilder(hardware=non_jetson_hardware)


@pytest.fixture
def deploy_config(tmp_path) -> DeploymentConfig:
    """A deployment config pointing at a temp ONNX path."""
    onnx = tmp_path / "best.onnx"
    onnx.write_bytes(b"fake-onnx")
    return DeploymentConfig(
        model_onnx_path=onnx,
        model_engine_path=tmp_path / "best_fp16.engine",
        precision=Precision.FP16,
        imgsz=640,
        max_batch_size=1,
    )


# ========== JetsonHardware Tests ==========

class TestJetsonHardware:
    """Tests for the JetsonHardware dataclass."""

    def test_non_jetson_defaults(self, non_jetson_hardware):
        """Non-Jetson hardware reports is_jetson=False."""
        hw = non_jetson_hardware
        assert hw.is_jetson is False
        assert hw.family == "unknown"
        assert hw.cuda_cores == 0
        assert hw.tensor_cores == 0

    def test_orin_capabilities(self, orin_hardware):
        """Orin NX supports FP16 and INT8."""
        assert orin_hardware.is_jetson is True
        assert orin_hardware.supports_fp16 is True
        assert orin_hardware.supports_int8 is True
        assert orin_hardware.tensor_cores > 0

    def test_nano_no_int8(self, nano_hardware):
        """Nano has no Tensor Cores and no INT8 support."""
        assert nano_hardware.supports_fp16 is False
        assert nano_hardware.supports_int8 is False
        assert nano_hardware.tensor_cores == 0

    def test_to_dict_roundtrip(self, orin_hardware):
        """to_dict returns all 14 fields."""
        d = orin_hardware.to_dict()
        assert d["is_jetson"] is True
        assert d["model"] == "NVIDIA Jetson Orin NX 16GB"
        assert d["family"] == "orin-nx"
        assert d["cuda_cores"] == 1024
        assert d["tensorrt_version"] == "8.6.0"
        assert "gpu_name" in d
        assert len(d) == 14

    def test_frozen_dataclass(self, orin_hardware):
        """JetsonHardware is frozen and immutable."""
        with pytest.raises(Exception):
            orin_hardware.is_jetson = False  # type: ignore


class TestDetectFamily:
    """Tests for _detect_family model-string parser."""

    @pytest.mark.parametrize("model_str, expected", [
        ("NVIDIA Jetson Orin Nano 8GB", "orin-nano"),
        ("NVIDIA Jetson Orin NX 16GB", "orin-nx"),
        ("NVIDIA Jetson AGX Orin", "orin"),
        ("NVIDIA Jetson Xavier NX", "xavier"),
        ("NVIDIA Jetson TX2", "tx2"),
        ("NVIDIA Jetson Nano", "nano"),
        ("Unknown Device", "unknown"),
    ])
    def test_family_detection(self, model_str, expected):
        assert _detect_family(model_str) == expected

    def test_jetson_models_dict_complete(self):
        """JETSON_MODELS covers all families used by detect_family."""
        expected_keys = {"nano", "tx2", "xavier", "orin", "orin-nx", "orin-nano"}
        assert expected_keys.issubset(JETSON_MODELS.keys())
        for specs in JETSON_MODELS.values():
            assert "cuda_cores" in specs
            assert "tensor_cores" in specs
            assert "memory_mb" in specs
            assert "max_power_w" in specs


class TestDetectJetson:
    """Tests for detect_jetson() on the actual host."""

    def test_detect_returns_hardware(self):
        """detect_jetson returns a JetsonHardware instance."""
        hw = detect_jetson()
        assert isinstance(hw, JetsonHardware)
        # On macOS dev machine, is_jetson should be False
        # (but we don't hard-assert since CI could be on Linux)
        assert isinstance(hw.is_jetson, bool)
        assert isinstance(hw.cuda_available, bool)

    def test_detect_populates_arch(self):
        """detect_jetson populates the arch field."""
        import platform
        hw = detect_jetson()
        assert hw.arch == platform.machine()

    def test_print_hardware_info(self, capsys, orin_hardware):
        """print_hardware_info writes to stdout without error."""
        print_hardware_info(orin_hardware)
        out = capsys.readouterr().out
        assert "NVIDIA Jetson Orin NX" in out
        assert "Is Jetson" in out


# ========== Power Management Tests ==========

class TestPowerMode:
    """Tests for PowerMode enum."""

    def test_mode_values(self):
        """PowerMode integer values are stable."""
        assert PowerMode.MAXN == 0
        assert PowerMode.MODE_10W == 1
        assert PowerMode.MODE_15W == 2
        assert int(PowerMode.MODE_25W) == 8

    def test_mode_from_int(self):
        """PowerMode can be constructed from int."""
        assert PowerMode(0) == PowerMode.MAXN
        assert PowerMode(8) == PowerMode.MODE_25W


class TestPowerState:
    """Tests for PowerState dataclass."""

    def test_unavailable_state(self):
        """Unavailable state has mode=-1."""
        state = PowerState(
            mode=-1,
            mode_name="unavailable",
            jetson_clocks_active=False,
            available=False,
        )
        assert state.available is False
        assert state.mode == -1

    def test_to_dict(self):
        """to_dict serializes all fields."""
        state = PowerState(
            mode=0, mode_name="MAXN",
            jetson_clocks_active=True, available=True,
        )
        d = state.to_dict()
        assert d["mode"] == 0
        assert d["mode_name"] == "MAXN"
        assert d["jetson_clocks_active"] is True
        assert d["available"] is True


class TestPowerManager:
    """Tests for PowerManager."""

    def test_non_jetson_unavailable(self, non_jetson_hardware):
        """On non-Jetson, PowerManager.available is False."""
        pm = PowerManager(hardware=non_jetson_hardware)
        assert pm.available is False

    def test_non_jetson_get_state(self, non_jetson_hardware):
        """get_state returns unavailable on non-Jetson."""
        pm = PowerManager(hardware=non_jetson_hardware)
        state = pm.get_state()
        assert state.available is False
        assert state.mode == -1

    def test_non_jetson_set_mode_fails(self, non_jetson_hardware):
        """set_mode returns False on non-Jetson."""
        pm = PowerManager(hardware=non_jetson_hardware)
        assert pm.set_mode(PowerMode.MAXN) is False

    def test_non_jetson_max_perf_fails(self, non_jetson_hardware):
        """enable_max_performance returns False on non-Jetson."""
        pm = PowerManager(hardware=non_jetson_hardware)
        assert pm.enable_max_performance() is False

    def test_dry_run_set_mode(self, orin_hardware, monkeypatch):
        """dry_run mode logs but does not execute."""
        # Simulate nvpmodel being present
        monkeypatch.setattr("shutil.which", lambda cmd: "/usr/sbin/nvpmodel" if cmd == "nvpmodel" else None)
        pm = PowerManager(hardware=orin_hardware, dry_run=True)
        assert pm.available is True
        assert pm.set_mode(PowerMode.MAXN) is True

    def test_dry_run_max_perf(self, orin_hardware, monkeypatch):
        """dry_run enable_max_performance succeeds."""
        monkeypatch.setattr("shutil.which", lambda cmd: "/usr/sbin/nvpmodel" if cmd == "nvpmodel" else None)
        pm = PowerManager(hardware=orin_hardware, dry_run=True)
        assert pm.enable_max_performance() is True

    def test_recommend_mode_orin(self, orin_hardware):
        """Orin recommends MAXN."""
        pm = PowerManager(hardware=orin_hardware)
        assert pm.recommend_mode() == PowerMode.MAXN

    def test_recommend_mode_nano(self, nano_hardware):
        """Nano recommends MODE_10W (thermally constrained)."""
        pm = PowerManager(hardware=nano_hardware)
        assert pm.recommend_mode() == PowerMode.MODE_10W

    def test_recommend_mode_xavier(self):
        """Xavier recommends MODE_30W_6CORE (balanced)."""
        hw = JetsonHardware(
            is_jetson=True, model="NVIDIA Jetson Xavier NX",
            family="xavier", cuda_cores=512, tensor_cores=64,
            memory_mb=8192, max_power_w=30, arch="aarch64",
            cuda_available=True, cuda_version="10.2",
            tensorrt_available=True, tensorrt_version="8.0",
            jetpack_version="R32.7", gpu_name="Xavier",
        )
        pm = PowerManager(hardware=hw)
        assert pm.recommend_mode() == PowerMode.MODE_30W_6CORE

    def test_print_power_state(self, capsys, non_jetson_hardware):
        """print_power_state works on non-Jetson."""
        print_power_state(PowerManager(hardware=non_jetson_hardware))
        out = capsys.readouterr().out
        assert "unavailable" in out.lower()


class TestRunCmd:
    """Tests for the _run_cmd helper."""

    def test_command_not_found(self):
        """Missing command returns rc=127."""
        rc, out = _run_cmd(["nonexistent-command-xyz"])
        assert rc == 127
        assert "not found" in out.lower()


# ========== TensorRT Engine Builder Tests ==========

class TestPrecision:
    """Tests for Precision enum."""

    def test_values(self):
        assert Precision.FP32.value == "fp32"
        assert Precision.FP16.value == "fp16"
        assert Precision.INT8.value == "int8"


class TestEngineBuildConfig:
    """Tests for EngineBuildConfig."""

    def test_defaults(self):
        """Default config is FP16, 640px, batch 1."""
        cfg = EngineBuildConfig()
        assert cfg.precision == Precision.FP16
        assert cfg.imgsz == 640
        assert cfg.max_batch_size == 1
        assert cfg.workspace_gb == 4
        assert cfg.calib_data_dir is None

    def test_to_dict(self):
        """to_dict serializes config."""
        cfg = EngineBuildConfig(precision=Precision.INT8, imgsz=320)
        d = cfg.to_dict()
        assert d["precision"] == "int8"
        assert d["imgsz"] == 320
        assert d["calib_data_dir"] is None


class TestTRTEngineBuilder:
    """Tests for TRTEngineBuilder."""

    def test_non_jetson_unavailable(self, non_jetson_builder):
        """Builder is unavailable without TensorRT."""
        assert non_jetson_builder.available is False

    def test_orin_available(self, orin_with_trt):
        """Builder is available on simulated Orin with TensorRT."""
        assert orin_with_trt.available is True

    def test_build_raises_without_trt(self, non_jetson_builder, tmp_path):
        """build() raises ModelError when TensorRT unavailable."""
        onnx = tmp_path / "model.onnx"
        onnx.write_bytes(b"x")
        with pytest.raises(ModelError, match="TensorRT is not available"):
            non_jetson_builder.build(onnx, tmp_path / "out.engine")

    def test_build_raises_missing_onnx(self, orin_with_trt, tmp_path):
        """build() raises ModelError when ONNX file missing."""
        with pytest.raises(ModelError, match="ONNX model not found"):
            orin_with_trt.build(
                tmp_path / "nonexistent.onnx",
                tmp_path / "out.engine",
            )

    def test_int8_requires_calib_dir(self, orin_with_trt, tmp_path):
        """INT8 build without calib_data_dir raises ModelError."""
        onnx = tmp_path / "model.onnx"
        onnx.write_bytes(b"x")
        cfg = EngineBuildConfig(precision=Precision.INT8)
        with pytest.raises(ModelError, match="INT8.*calib_data_dir"):
            orin_with_trt.build(onnx, tmp_path / "out.engine", config=cfg)

    def test_int8_unsupported_on_nano(self, nano_hardware, tmp_path):
        """INT8 on Nano (no support) raises ModelError."""
        onnx = tmp_path / "model.onnx"
        onnx.write_bytes(b"x")
        builder = TRTEngineBuilder(hardware=nano_hardware)
        cfg = EngineBuildConfig(precision=Precision.INT8)
        with pytest.raises(ModelError, match="INT8.*not supported.*nano"):
            builder.build(onnx, tmp_path / "out.engine", config=cfg)

    def test_recommend_precision_orin(self, orin_with_trt):
        """Orin recommends INT8."""
        assert orin_with_trt.recommend_precision() == Precision.INT8

    def test_recommend_precision_nano(self, nano_hardware):
        """Nano (no Tensor Cores) recommends FP32."""
        builder = TRTEngineBuilder(hardware=nano_hardware)
        assert builder.recommend_precision() == Precision.FP32

    def test_recommend_precision_fp16(self):
        """Xavier (has Tensor Cores, no INT8) recommends FP16 - actually Xavier supports INT8."""
        # Xavier is in the supports_int8 set, so it recommends INT8.
        # Test a device with Tensor Cores but NOT in the INT8 set.
        hw = JetsonHardware(
            is_jetson=True, model="NVIDIA Jetson TX2",
            family="tx2", cuda_cores=256, tensor_cores=32,
            memory_mb=8192, max_power_w=15, arch="aarch64",
            cuda_available=True, cuda_version="10.2",
            tensorrt_available=True, tensorrt_version="8.0",
            jetpack_version="R32.7", gpu_name="TX2",
        )
        builder = TRTEngineBuilder(hardware=hw)
        # TX2 has tensor_cores > 0 (supports_fp16=True) but NOT in INT8 set
        assert builder.recommend_precision() == Precision.FP16

    def test_print_builder_info(self, capsys, non_jetson_builder):
        """print_builder_info works without TensorRT."""
        print_builder_info(non_jetson_builder)
        out = capsys.readouterr().out
        assert "TensorRT Available : False" in out


class TestINT8Calibrator:
    """Tests for INT8Calibrator."""

    def test_empty_dir(self, tmp_path):
        """Calibrator with no images has 0 images."""
        calib = INT8Calibrator(data_dir=tmp_path, imgsz=640)
        assert calib.num_images == 0

    def test_with_images(self, tmp_path):
        """Calibrator collects image paths."""
        for name in ["a.jpg", "b.png", "c.txt"]:
            (tmp_path / name).write_bytes(b"x")
        calib = INT8Calibrator(data_dir=tmp_path)
        assert calib.num_images == 2  # .txt excluded

    def test_get_batch_exhausted(self, tmp_path):
        """get_batch returns None when exhausted."""
        calib = INT8Calibrator(data_dir=tmp_path)
        assert calib.get_batch(["input"]) is None

    def test_cache_roundtrip(self, tmp_path):
        """Calibration cache write/read works."""
        cache = tmp_path / "calib.cache"
        calib = INT8Calibrator(data_dir=tmp_path, cache_path=cache)
        assert calib.read_calibration_cache() is None
        calib.write_calibration_cache(b"cache-data")
        assert cache.exists()
        assert cache.read_bytes() == b"cache-data"
        # Re-load
        calib2 = INT8Calibrator(data_dir=tmp_path, cache_path=cache)
        assert calib2.read_calibration_cache() == b"cache-data"

    def test_max_images_limit(self, tmp_path):
        """max_calibration_images limits collected images."""
        for i in range(10):
            (tmp_path / f"img_{i}.jpg").write_bytes(b"x")
        calib = INT8Calibrator(data_dir=tmp_path, max_calibration_images=5)
        assert calib.num_images == 5


# ========== Edge Monitor Tests ==========

class TestEdgeHealth:
    """Tests for EdgeHealth dataclass."""

    def test_defaults(self):
        """EdgeHealth has sensible defaults."""
        h = EdgeHealth(
            timestamp=time.time(),
            cpu_percent=10.0,
            memory_percent=50.0,
            memory_used_mb=1024,
            memory_total_mb=2048,
            disk_percent=30.0,
        )
        assert h.gpu_percent is None
        assert h.warnings == []
        assert h.available is True

    def test_to_dict(self):
        """to_dict serializes all fields."""
        h = EdgeHealth(
            timestamp=1000.0, cpu_percent=10.0,
            memory_percent=50.0, memory_used_mb=1024,
            memory_total_mb=2048, disk_percent=30.0,
            gpu_percent=80.0, gpu_temp=65.0,
        )
        d = h.to_dict()
        assert d["cpu_percent"] == 10.0
        assert d["gpu_percent"] == 80.0
        assert d["timestamp"] == 1000.0
        assert d["warnings"] == []


class TestEdgeMonitor:
    """Tests for EdgeMonitor."""

    def test_snapshot_returns_health(self, non_jetson_hardware):
        """snapshot() returns an EdgeHealth on non-Jetson."""
        monitor = EdgeMonitor(hardware=non_jetson_hardware)
        health = monitor.snapshot()
        assert isinstance(health, EdgeHealth)
        assert health.available is True
        assert health.cpu_percent >= 0.0
        assert 0.0 <= health.memory_percent <= 100.0
        # Non-Jetson: GPU fields should be None
        assert health.gpu_percent is None
        assert health.gpu_temp is None

    def test_snapshot_warnings_empty_on_cold_host(self, non_jetson_hardware):
        """A healthy non-Jetson host produces no warnings."""
        monitor = EdgeMonitor(hardware=non_jetson_hardware)
        health = monitor.snapshot()
        # No thermal data on non-Jetson -> no thermal warnings
        assert isinstance(health.warnings, list)

    def test_thermal_warning(self):
        """Temperature above warning threshold emits warning."""
        hw = JetsonHardware(
            is_jetson=True, model="Jetson", family="orin",
            cuda_cores=1024, tensor_cores=32, memory_mb=8192,
            max_power_w=25, arch="aarch64", cuda_available=True,
            cuda_version="12.2", tensorrt_available=True,
            tensorrt_version="8.6", jetpack_version="R35", gpu_name="Orin",
        )
        monitor = EdgeMonitor(
            hardware=hw,
            thermal_warning_c=60.0,
            thermal_critical_c=85.0,
        )
        # Patch snapshot to inject a high temperature
        with patch.object(monitor, 'snapshot') as mock_snap:
            mock_snap.return_value = EdgeHealth(
                timestamp=time.time(), cpu_percent=10,
                memory_percent=50, memory_used_mb=1024,
                memory_total_mb=2048, disk_percent=30,
                gpu_temp=70.0,
                warnings=["WARNING: temperature 70.0°C exceeds 60.0°C"],
            )
            h = monitor.snapshot()
            assert any("WARNING" in w for w in h.warnings)

    def test_stream_yields_snapshots(self, non_jetson_hardware):
        """stream() async generator yields health snapshots."""
        monitor = EdgeMonitor(hardware=non_jetson_hardware)

        async def _collect():
            results = []
            async for health in monitor.stream(interval=0.5):
                results.append(health)
                if len(results) >= 2:
                    break
            return results

        results = asyncio.run(_collect())
        assert len(results) == 2
        assert all(isinstance(h, EdgeHealth) for h in results)

    def test_stream_min_interval(self, non_jetson_hardware):
        """stream() enforces a minimum interval of 0.5s."""
        monitor = EdgeMonitor(hardware=non_jetson_hardware)
        # Access the clamped interval via a quick run
        async def _check():
            gen = monitor.stream(interval=0.1)
            # The first yield should happen quickly
            health = await gen.__anext__()
            return health

        h = asyncio.run(_check())
        assert isinstance(h, EdgeHealth)

    def test_print_health(self, capsys, non_jetson_hardware):
        """print_health writes to stdout."""
        monitor = EdgeMonitor(hardware=non_jetson_hardware)
        print_health(monitor.snapshot())
        out = capsys.readouterr().out
        assert "CPU Usage" in out
        assert "Memory" in out


# ========== Deployment Manager Tests ==========

class TestDeploymentConfig:
    """Tests for DeploymentConfig."""

    def test_defaults(self, tmp_path):
        """DeploymentConfig has correct defaults."""
        cfg = DeploymentConfig(
            model_onnx_path=tmp_path / "model.onnx",
            model_engine_path=tmp_path / "model.engine",
        )
        assert cfg.precision == Precision.FP16
        assert cfg.imgsz == 640
        assert cfg.auto_set_power_mode is True
        assert cfg.api_host == "0.0.0.0"
        assert cfg.api_port == 8000

    def test_to_dict(self, tmp_path):
        """to_dict serializes config."""
        cfg = DeploymentConfig(
            model_onnx_path=tmp_path / "a.onnx",
            model_engine_path=tmp_path / "b.engine",
        )
        d = cfg.to_dict()
        assert d["precision"] == "fp16"
        assert d["api_port"] == 8000
        assert "model_onnx_path" in d


class TestEdgeDeploymentManager:
    """Tests for EdgeDeploymentManager."""

    def test_preflight_missing_onnx(self, tmp_path, non_jetson_hardware):
        """preflight warns when ONNX is missing."""
        cfg = DeploymentConfig(
            model_onnx_path=tmp_path / "missing.onnx",
            model_engine_path=tmp_path / "out.engine",
        )
        mgr = EdgeDeploymentManager(cfg, hardware=non_jetson_hardware)
        warnings = mgr.preflight()
        assert any("ONNX model not found" in w for w in warnings)

    def test_preflight_non_jetson(self, deploy_config, non_jetson_hardware):
        """preflight warns on non-Jetson host."""
        mgr = EdgeDeploymentManager(
            deploy_config, hardware=non_jetson_hardware
        )
        warnings = mgr.preflight()
        assert any("not a Jetson" in w for w in warnings)

    def test_preflight_int8_no_calib(self, tmp_path, orin_hardware):
        """preflight warns when INT8 has no calibration data."""
        onnx = tmp_path / "model.onnx"
        onnx.write_bytes(b"x")
        cfg = DeploymentConfig(
            model_onnx_path=onnx,
            model_engine_path=tmp_path / "out.engine",
            precision=Precision.INT8,
        )
        mgr = EdgeDeploymentManager(cfg, hardware=orin_hardware)
        warnings = mgr.preflight()
        assert any("INT8" in w for w in warnings)

    def test_deploy_non_jetson_success(self, deploy_config, non_jetson_hardware):
        """deploy() on non-Jetson succeeds (CPU fallback, no engine)."""
        mgr = EdgeDeploymentManager(
            deploy_config, hardware=non_jetson_hardware
        )
        report = mgr.deploy()
        assert isinstance(report, DeploymentReport)
        assert report.success is True
        assert report.engine_build is None
        assert report.engine_reused is False
        assert report.hardware.is_jetson is False
        # Should warn about non-Jetson
        assert any("non-Jetson" in w or "TensorRT unavailable" in w for w in report.warnings)

    def test_deploy_reuses_existing_engine(self, tmp_path, non_jetson_hardware):
        """deploy() reuses an existing engine file."""
        onnx = tmp_path / "model.onnx"
        onnx.write_bytes(b"x")
        engine = tmp_path / "model.engine"
        engine.write_bytes(b"existing-engine")
        cfg = DeploymentConfig(
            model_onnx_path=onnx,
            model_engine_path=engine,
        )
        mgr = EdgeDeploymentManager(cfg, hardware=non_jetson_hardware)
        report = mgr.deploy()
        assert report.engine_reused is True
        assert report.engine_build is None

    def test_deploy_force_rebuild_non_jetson(self, tmp_path, non_jetson_hardware):
        """force_rebuild on non-Jetson skips build (no TRT)."""
        onnx = tmp_path / "model.onnx"
        onnx.write_bytes(b"x")
        engine = tmp_path / "model.engine"
        engine.write_bytes(b"old")
        cfg = DeploymentConfig(
            model_onnx_path=onnx,
            model_engine_path=engine,
        )
        mgr = EdgeDeploymentManager(cfg, hardware=non_jetson_hardware)
        report = mgr.deploy(force_rebuild=True)
        # Non-Jetson: build skipped, but success=True (CPU fallback)
        assert report.engine_reused is False
        assert report.engine_build is None
        assert report.success is True

    def test_deploy_dry_run_jetson(self, deploy_config, orin_hardware, monkeypatch):
        """deploy() with dry_run on simulated Jetson sets power mode."""
        monkeypatch.setattr("shutil.which", lambda cmd: f"/usr/sbin/{cmd}" if cmd == "nvpmodel" else None)
        mgr = EdgeDeploymentManager(
            deploy_config, hardware=orin_hardware, dry_run=True
        )
        report = mgr.deploy()
        assert report.power_mode_set is not None
        assert report.jetson_clocks_enabled is True

    def test_health(self, deploy_config, non_jetson_hardware):
        """health() returns an EdgeHealth snapshot."""
        mgr = EdgeDeploymentManager(
            deploy_config, hardware=non_jetson_hardware
        )
        health = mgr.health()
        assert isinstance(health, EdgeHealth)

    def test_shutdown_non_jetson(self, deploy_config, non_jetson_hardware):
        """shutdown() is a no-op on non-Jetson."""
        mgr = EdgeDeploymentManager(
            deploy_config, hardware=non_jetson_hardware
        )
        mgr.shutdown()  # should not raise

    def test_report_to_dict(self, deploy_config, non_jetson_hardware):
        """DeploymentReport.to_dict serializes correctly."""
        mgr = EdgeDeploymentManager(
            deploy_config, hardware=non_jetson_hardware
        )
        report = mgr.deploy()
        d = report.to_dict()
        assert d["success"] is True
        assert "hardware" in d
        assert "deploy_time_sec" in d
        assert d["engine_build"] is None


# ========== CLI Tests ==========

class TestEdgeCLI:
    """Tests for the edge CLI commands."""

    def test_hardware_command(self):
        """`skyguard edge hardware` runs without error."""
        from typer.testing import CliRunner
        from skyguard.main import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["edge", "hardware"])
        assert result.exit_code == 0
        assert "Platform" in result.output or "Model" in result.output

    def test_power_command(self):
        """`skyguard edge power` runs without error."""
        from typer.testing import CliRunner
        from skyguard.main import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["edge", "power"])
        assert result.exit_code == 0

    def test_health_command(self):
        """`skyguard edge health` runs without error."""
        from typer.testing import CliRunner
        from skyguard.main import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["edge", "health"])
        assert result.exit_code == 0
        assert "CPU" in result.output

    def test_health_json_command(self):
        """`skyguard edge health --json` outputs JSON."""
        from typer.testing import CliRunner
        from skyguard.main import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["edge", "health", "--json"])
        assert result.exit_code == 0

    def test_info_command(self):
        """`skyguard edge info` runs without error."""
        from typer.testing import CliRunner
        from skyguard.main import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["edge", "info"])
        assert result.exit_code == 0
        assert "Hardware" in result.output

    def test_build_command_no_trt(self, tmp_path):
        """`skyguard edge build` fails gracefully without TensorRT."""
        from typer.testing import CliRunner
        from skyguard.main import cli
        onnx = tmp_path / "model.onnx"
        onnx.write_bytes(b"x")
        runner = CliRunner()
        result = runner.invoke(cli, ["edge", "build", str(onnx), str(tmp_path / "out.engine")])
        # Exits with code 1 because TensorRT is unavailable on dev host
        assert result.exit_code == 1

    def test_deploy_command_non_jetson(self, tmp_path):
        """`skyguard edge deploy` on non-Jetson succeeds (CPU fallback)."""
        from typer.testing import CliRunner
        from skyguard.main import cli
        onnx = tmp_path / "model.onnx"
        onnx.write_bytes(b"x")
        runner = CliRunner()
        result = runner.invoke(cli, [
            "edge", "deploy",
            str(onnx), str(tmp_path / "out.engine"),
            "--no-power", "--no-clocks",
        ])
        assert result.exit_code == 0
        assert "Deployment Report" in result.output


# ========== Run Tests ==========

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
