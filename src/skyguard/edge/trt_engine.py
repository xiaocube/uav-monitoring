"""TensorRT engine builder for SkyGuard YOLO models.

Builds optimized TensorRT engines from ONNX models with FP32 / FP16 / INT8
precision. On non-Jetson hosts without TensorRT, the builder reports
``available=False`` and raises a clear error if invoked, so the rest of
the SkyGuard stack keeps working on dev machines.

Pipeline:
  1. Validate ONNX model path
  2. Create TensorRT builder + network + config
  3. Set precision (FP16 default, INT8 optional with calibrator)
  4. Build serialized engine (this can take minutes)
  5. Save to .engine / .plan file
  6. Return path for downstream TensorRTBackend

This module intentionally does NOT load the engine at build time -
that is the job of the inference backend. Separation of concerns
keeps memory usage predictable during long builds.
"""
from __future__ import annotations

import enum
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from skyguard.core.exceptions import ModelError
from skyguard.core.logger import get_logger
from skyguard.edge.hardware import JetsonHardware, detect_jetson

log = get_logger(__name__)


class Precision(enum.Enum):
    """TensorRT inference precision.

    FP16 is the default for Jetson (best accuracy/speed trade-off).
    INT8 requires a calibration dataset and is only supported on
    Xavier / Orin family devices.
    """
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"


@dataclass
class EngineBuildConfig:
    """Configuration for a TensorRT engine build.

    Attributes
    ----------
    precision : Precision
        Target precision. INT8 also requires ``calib_data_dir``.
    imgsz : int
        Input image size (square). Must match the ONNX export size.
    max_batch_size : int
        Maximum batch size the engine will accept at inference time.
    workspace_gb : int
        Builder workspace in GB. Larger = more optimization opportunities
        but uses more RAM during build.
    calib_data_dir : Optional[Path]
        Directory of calibration images (for INT8). Required when
        precision == INT8.
    calib_cache : Optional[Path]
        Path to cache file for calibration results (speeds up rebuilds).
    dynamic_batch : bool
        If True, engine accepts variable batch size up to max_batch_size.
        If False, engine is optimized for exactly max_batch_size (faster).
    """
    precision: Precision = Precision.FP16
    imgsz: int = 640
    max_batch_size: int = 1
    workspace_gb: int = 4
    calib_data_dir: Optional[Path] = None
    calib_cache: Optional[Path] = None
    dynamic_batch: bool = False

    def to_dict(self) -> dict:
        return {
            "precision": self.precision.value,
            "imgsz": self.imgsz,
            "max_batch_size": self.max_batch_size,
            "workspace_gb": self.workspace_gb,
            "calib_data_dir": str(self.calib_data_dir) if self.calib_data_dir else None,
            "calib_cache": str(self.calib_cache) if self.calib_cache else None,
            "dynamic_batch": self.dynamic_batch,
        }


@dataclass
class BuildResult:
    """Result of an engine build."""
    engine_path: Path
    precision: Precision
    input_shape: tuple
    build_time_sec: float
    engine_size_mb: float
    config: EngineBuildConfig

    def to_dict(self) -> dict:
        return {
            "engine_path": str(self.engine_path),
            "precision": self.precision.value,
            "input_shape": list(self.input_shape),
            "build_time_sec": self.build_time_sec,
            "engine_size_mb": self.engine_size_mb,
            "config": self.config.to_dict(),
        }


# ---------------------------------------------------------------------------
# INT8 Calibrator
# ---------------------------------------------------------------------------

class INT8Calibrator:
    """TensorRT INT8 calibrator using a directory of images.

    Implements the minimal IInt8Calibrator interface needed by
    TensorRT. Reads images from a directory, preprocesses them with
    the same letterbox logic as the inference backend, and feeds
    them to the calibration loop.

    On non-TensorRT hosts the class can still be instantiated for
    testing - it just cannot be passed to a real builder.
    """

    def __init__(
        self,
        data_dir: Path,
        cache_path: Optional[Path] = None,
        imgsz: int = 640,
        batch_size: int = 8,
        max_calibration_images: int = 500,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.cache_path = Path(cache_path) if cache_path else None
        self.imgsz = imgsz
        self.batch_size = batch_size
        self.max_images = max_calibration_images

        # Collect image paths
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        self.image_paths: List[Path] = [
            p for p in sorted(self.data_dir.rglob("*")) if p.suffix.lower() in exts
        ][:max_calibration_images]
        log.info(
            "INT8 calibrator: {} images from {}",
            len(self.image_paths),
            self.data_dir,
        )

        self._current_idx = 0
        self._cache_bytes: Optional[bytes] = None
        if self.cache_path and self.cache_path.exists():
            self._cache_bytes = self.cache_path.read_bytes()
            log.info("Loaded INT8 calibration cache: {}", self.cache_path)

    @property
    def num_images(self) -> int:
        return len(self.image_paths)

    def _preprocess(self, image_path: Path) -> Any:
        """Letterbox + normalize a single image for calibration."""
        import cv2
        import numpy as np

        img = cv2.imread(str(image_path))
        if img is None:
            # Return a gray placeholder if image can't be read
            return np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)

        h, w = img.shape[:2]
        scale = min(self.imgsz / w, self.imgsz / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        pad_w = (self.imgsz - new_w) // 2
        pad_h = (self.imgsz - new_h) // 2
        canvas[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

        # BGR -> RGB, normalize, HWC -> CHW
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        return normalized.transpose(2, 0, 1)

    def get_batch(self, names: List[str]) -> Optional[List[Any]]:
        """Return the next batch of calibration data, or None when exhausted."""
        if self._current_idx >= len(self.image_paths):
            return None

        import numpy as np

        end_idx = min(self._current_idx + self.batch_size, len(self.image_paths))
        batch_paths = self.image_paths[self._current_idx:end_idx]
        batch = np.stack([self._preprocess(p) for p in batch_paths], axis=0)
        self._current_idx = end_idx
        # TensorRT expects a list of device pointers in real usage;
        # here we just return the numpy batch - the trt wrapper handles
        # the copy to GPU memory.
        return [batch]

    def read_calibration_cache(self) -> Optional[bytes]:
        return self._cache_bytes

    def write_calibration_cache(self, cache: bytes) -> None:
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_bytes(cache)
            log.info("Wrote INT8 calibration cache: {}", self.cache_path)


# ---------------------------------------------------------------------------
# Engine Builder
# ---------------------------------------------------------------------------

class TRTEngineBuilder:
    """Build optimized TensorRT engines from ONNX models.

    Usage:
        builder = TRTEngineBuilder()
        if builder.available:
            result = builder.build(
                onnx_path=Path("models/best.onnx"),
                output_path=Path("models/best_fp16.engine"),
                config=EngineBuildConfig(precision=Precision.FP16),
            )
    """

    def __init__(self, hardware: Optional[JetsonHardware] = None) -> None:
        self._hardware = hardware

    @property
    def hardware(self) -> JetsonHardware:
        if self._hardware is None:
            self._hardware = detect_jetson()
        return self._hardware

    @property
    def available(self) -> bool:
        """True if TensorRT is importable on this host."""
        return self.hardware.tensorrt_available

    def _import_trt(self):
        """Import tensorrt lazily; raise ModelError if unavailable."""
        if not self.available:
            raise ModelError(
                "TensorRT is not available on this host. "
                "Engine build requires Jetson or an NVIDIA GPU host with TensorRT installed."
            )
        import tensorrt as trt  # type: ignore
        return trt

    def build(
        self,
        onnx_path: Path,
        output_path: Path,
        config: Optional[EngineBuildConfig] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> BuildResult:
        """Build a TensorRT engine from an ONNX model.

        Parameters
        ----------
        onnx_path : Path
            Path to the source ONNX model.
        output_path : Path
            Where to write the .engine file. Parent dirs are created.
        config : EngineBuildConfig, optional
            Build configuration. Defaults to FP16 precision.
        progress_callback : callable, optional
            Called with progress messages during the (long) build.

        Returns
        -------
        BuildResult
            Metadata about the built engine.

        Raises
        ------
        ModelError
            If TensorRT is unavailable, the ONNX file is missing,
            or INT8 is requested without a calib_data_dir.
        """
        config = config or EngineBuildConfig()
        onnx_path = Path(onnx_path)
        output_path = Path(output_path)

        if not onnx_path.exists():
            raise ModelError(f"ONNX model not found: {onnx_path}")

        if config.precision == Precision.INT8:
            if not self.hardware.supports_int8:
                raise ModelError(
                    f"INT8 precision not supported on {self.hardware.family}. "
                    "Requires Xavier or Orin family."
                )
            if config.calib_data_dir is None or not config.calib_data_dir.exists():
                raise ModelError(
                    "INT8 precision requires calib_data_dir pointing to a "
                    "directory of calibration images."
                )

        def _log(msg: str) -> None:
            log.info(msg)
            if progress_callback:
                progress_callback(msg)

        trt = self._import_trt()
        _log(f"TensorRT {trt.__version__} - building engine from {onnx_path.name}")

        # 1. Create logger, builder, network, config
        trt_logger = trt.Logger(trt.Logger.INFO)
        builder = trt.Builder(trt_logger)
        flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(flag)
        parser = trt.OnnxParser(network, trt_logger)

        # 2. Parse ONNX
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    log.error(parser.get_error(i))
                raise ModelError(f"Failed to parse ONNX: {onnx_path}")
        _log("ONNX parsed successfully")

        # 3. Configure builder
        builder_config = builder.create_builder_config()
        builder_config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE,
            config.workspace_gb * (1 << 30),
        )

        # Precision settings
        if config.precision == Precision.FP16:
            if not builder.platform_has_fast_fp16:
                _log("WARNING: FP16 not natively fast on this platform")
            builder_config.set_flag(trt.BuilderFlag.FP16)
            _log("Precision: FP16")
        elif config.precision == Precision.INT8:
            builder_config.set_flag(trt.BuilderFlag.INT8)
            builder_config.set_flag(trt.BuilderFlag.FP16)  # INT8 + FP16 fallback
            calibrator = INT8Calibrator(
                data_dir=config.calib_data_dir,
                cache_path=config.calib_cache,
                imgsz=config.imgsz,
            )
            builder_config.int8_calibrator = calibrator
            _log(f"Precision: INT8 (calibrator: {calibrator.num_images} images)")
        else:
            _log("Precision: FP32")

        # 4. Dynamic batch profile (optional)
        input_tensor = network.get_input(0)
        input_shape = input_tensor.shape
        if config.dynamic_batch:
            profile = builder.create_optimization_profile()
            min_shape = (1, 3, config.imgsz, config.imgsz)
            opt_shape = (max(1, config.max_batch_size // 2), 3, config.imgsz, config.imgsz)
            max_shape = (config.max_batch_size, 3, config.imgsz, config.imgsz)
            profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
            builder_config.add_optimization_profile(profile)
            _log(f"Dynamic batch: min={min_shape[0]} opt={opt_shape[0]} max={max_shape[0]}")
        else:
            input_shape = (config.max_batch_size, 3, config.imgsz, config.imgsz)
            _log(f"Fixed batch: {input_shape}")

        # 5. Build (this is the slow step)
        _log("Building engine... (this may take several minutes)")
        t0 = time.perf_counter()
        serialized = builder.build_serialized_network(network, builder_config)
        if serialized is None:
            raise ModelError(
                "TensorRT engine build failed. Check logs above for errors. "
                "Common causes: insufficient workspace, unsupported ops, "
                "or ONNX/tensorrt version mismatch."
            )
        build_time = time.perf_counter() - t0
        _log(f"Engine built in {build_time:.1f}s")

        # 6. Save engine
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(serialized)
        size_mb = output_path.stat().st_size / (1024 * 1024)
        _log(f"Engine saved: {output_path} ({size_mb:.1f} MB)")

        return BuildResult(
            engine_path=output_path,
            precision=config.precision,
            input_shape=tuple(input_shape),
            build_time_sec=build_time,
            engine_size_mb=size_mb,
            config=config,
        )

    def recommend_precision(self) -> Precision:
        """Recommend a precision based on hardware capabilities.

          * Orin / Xavier with INT8 support -> INT8 (3-4x faster than FP16)
          * Any Jetson with Tensor Cores    -> FP16 (2x faster than FP32)
          * Otherwise                       -> FP32
        """
        if self.hardware.supports_int8:
            return Precision.INT8
        if self.hardware.supports_fp16:
            return Precision.FP16
        return Precision.FP32


def print_builder_info(builder: Optional[TRTEngineBuilder] = None) -> None:
    """Pretty-print builder info (used by CLI)."""
    builder = builder or TRTEngineBuilder()
    hw = builder.hardware
    print(f"  TensorRT Available : {builder.available}")
    if builder.available:
        print(f"  TensorRT Version   : {hw.tensorrt_version}")
        rec = builder.recommend_precision()
        print(f"  Recommended Precision : {rec.value}")
        print(f"  Supports FP16      : {hw.supports_fp16}")
        print(f"  Supports INT8      : {hw.supports_int8}")
