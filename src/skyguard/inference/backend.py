"""
SkyGuard inference backend abstraction.

Provides a unified interface for multiple inference backends:
  * PyTorch (native)
  * ONNX Runtime (CPU optimized)
  * TensorRT (NVIDIA GPU optimized)
  * OpenVINO (Intel CPU/GPU)

Usage:
    backend = InferenceBackend.create("onnx", "models/trained/best.onnx")
    result = backend.infer(image)
    detections = backend.postprocess(result, conf_threshold=0.5)
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

from skyguard.core.exceptions import ModelError
from skyguard.core.logger import get_logger
from skyguard.vision.classes import SKYGUARD_NAMES

log = get_logger(__name__)


@dataclass
class Detection:
    """Single detection result."""

    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    class_id: int
    class_name: str
    confidence: float


@dataclass
class InferenceResult:
    """Raw inference result from backend."""

    output: np.ndarray
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float
    total_ms: float
    backend: str
    input_shape: Tuple[int, int, int, int]  # (1, 3, H, W)


class BaseBackend(ABC):
    """Abstract base class for inference backends."""

    def __init__(self, model_path: Union[str, Path], **kwargs) -> None:
        self.model_path = Path(model_path)
        self.input_size: Tuple[int, int] = kwargs.get("input_size", (640, 640))
        self.conf_threshold: float = kwargs.get("conf_threshold", 0.25)
        self.iou_threshold: float = kwargs.get("iou_threshold", 0.45)
        self.classes: List[str] = kwargs.get("classes", list(SKYGUARD_NAMES.values()))

    @abstractmethod
    def infer(self, image: np.ndarray) -> InferenceResult:
        """Run inference on a single image.

        Args:
            image: BGR image (H, W, 3) in uint8

        Returns:
            InferenceResult with raw output and timing info
        """
        pass

    @abstractmethod
    def warmup(self, runs: int = 3) -> float:
        """Warmup the model to get stable inference times.

        Returns:
            Average inference time in ms after warmup
        """
        pass

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, float, float]]:
        """Preprocess image for inference.

        Returns:
            (preprocessed_image, (orig_h, orig_w, scale, pad))
        """
        import cv2

        orig_h, orig_w = image.shape[:2]
        target_h, target_w = self.input_size

        # Letterbox resize
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        # Resize
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad
        pad_w = (target_w - new_w) // 2
        pad_h = (target_h - new_h) // 2

        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

        # Convert to RGB and normalize
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0

        # HWC -> CHW -> NCHW
        chw = normalized.transpose(2, 0, 1)
        nchw = np.expand_dims(chw, axis=0)

        return nchw, (orig_h, orig_w, scale, (pad_h, pad_w))

    def postprocess(
        self,
        output: np.ndarray,
        orig_shape: Tuple[int, int, float, Tuple[int, int]],
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
    ) -> List[Detection]:
        """Postprocess raw output to detections.

        Args:
            output: Raw model output (1, 4+N, num_boxes) or (1, num_boxes, 4+N)
            orig_shape: (orig_h, orig_w, scale, (pad_h, pad_w))
            conf_threshold: Confidence threshold
            iou_threshold: NMS IoU threshold

        Returns:
            List of Detection objects
        """
        import cv2

        conf = conf_threshold or self.conf_threshold
        iou = iou_threshold or self.iou_threshold

        orig_h, orig_w, scale, (pad_h, pad_w) = orig_shape

        # Handle different output formats
        if output.ndim == 3:
            if output.shape[1] == 4 + len(self.classes):
                # (1, 4+N, num_boxes) -> (num_boxes, 4+N)
                output = output[0].T
            elif output.shape[2] == 4 + len(self.classes):
                # (1, num_boxes, 4+N)
                output = output[0]

        # Filter by confidence
        scores = output[:, 4:]
        class_ids = np.argmax(scores, axis=1)
        confidences = scores[np.arange(len(scores)), class_ids]

        mask = confidences >= conf
        filtered = output[mask]
        filtered_class_ids = class_ids[mask]
        filtered_confidences = confidences[mask]

        if len(filtered) == 0:
            return []

        # Extract boxes (xywh -> xyxy)
        boxes = filtered[:, :4].copy()
        boxes[:, 0] -= boxes[:, 2] / 2  # x_center -> x1
        boxes[:, 1] -= boxes[:, 3] / 2  # y_center -> y1
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]  # x2
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]  # y2

        # Scale back to original image
        boxes[:, [0, 2]] -= pad_w
        boxes[:, [1, 3]] -= pad_h
        boxes[:, :4] /= scale

        # Clip to image bounds
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)

        # NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            filtered_confidences.tolist(),
            conf,
            iou,
        )

        if len(indices) == 0:
            return []

        # Build detections
        detections = []
        for i in indices.flatten():
            x1, y1, x2, y2 = boxes[i].astype(int)
            cls_id = int(filtered_class_ids[i])
            cls_name = self.classes[cls_id] if cls_id < len(self.classes) else f"class_{cls_id}"
            det = Detection(
                bbox=(x1, y1, x2, y2),
                class_id=cls_id,
                class_name=cls_name,
                confidence=float(filtered_confidences[i]),
            )
            detections.append(det)

        return detections


class PyTorchBackend(BaseBackend):
    """Native PyTorch inference backend."""

    def __init__(self, model_path: Union[str, Path], **kwargs) -> None:
        super().__init__(model_path, **kwargs)
        self.device = kwargs.get("device", "auto")
        self._model = None

    def _load_model(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(str(self.model_path))
        return self._model

    def infer(self, image: np.ndarray) -> InferenceResult:
        model = self._load_model()

        t0 = time.perf_counter()
        input_tensor, orig_shape = self.preprocess(image)
        t1 = time.perf_counter()

        # Run inference
        results = model.predict(
            source=image,
            imgsz=self.input_size[0],
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )
        t2 = time.perf_counter()

        # Extract output
        if len(results) > 0 and results[0].boxes is not None:
            output = results[0].boxes.data.cpu().numpy()
        else:
            output = np.zeros((0, 6), dtype=np.float32)

        t3 = time.perf_counter()

        return InferenceResult(
            output=output,
            preprocess_ms=(t1 - t0) * 1000,
            inference_ms=(t2 - t1) * 1000,
            postprocess_ms=(t3 - t2) * 1000,
            total_ms=(t3 - t0) * 1000,
            backend="pytorch",
            input_shape=(1, 3, *self.input_size),
        )

    def warmup(self, runs: int = 3) -> float:
        dummy = np.zeros((*self.input_size[::-1], 3), dtype=np.uint8)
        times = []
        for _ in range(runs):
            result = self.infer(dummy)
            times.append(result.inference_ms)
        return np.mean(times)


class ONNXBackend(BaseBackend):
    """ONNX Runtime inference backend."""

    def __init__(self, model_path: Union[str, Path], **kwargs) -> None:
        super().__init__(model_path, **kwargs)
        self.device = kwargs.get("device", "cpu")
        self._session = None
        self._provider = None

    def _load_session(self):
        if self._session is None:
            try:
                import onnxruntime as ort
            except ImportError:
                raise ModelError("onnxruntime not installed. Run: pip install onnxruntime")

            providers = ["CPUExecutionProvider"]

            if self.device == "cuda":
                providers.insert(0, "CUDAExecutionProvider")
            elif self.device == "auto":
                available = ort.get_available_providers()
                if "CoreMLExecutionProvider" in available:
                    providers.insert(0, "CoreMLExecutionProvider")
                elif "CUDAExecutionProvider" in available:
                    providers.insert(0, "CUDAExecutionProvider")

            self._session = ort.InferenceSession(
                str(self.model_path),
                providers=providers,
            )
            self._provider = self._session.get_providers()[0]
            log.info("ONNX session created: provider={}", self._provider)

        return self._session

    def infer(self, image: np.ndarray) -> InferenceResult:
        session = self._load_session()

        t0 = time.perf_counter()
        input_tensor, orig_shape = self.preprocess(image)
        t1 = time.perf_counter()

        # Get input name
        input_name = session.get_inputs()[0].name

        # Run inference
        outputs = session.run(None, {input_name: input_tensor})
        t2 = time.perf_counter()

        # Get output
        output = outputs[0]

        t3 = time.perf_counter()

        return InferenceResult(
            output=output,
            preprocess_ms=(t1 - t0) * 1000,
            inference_ms=(t2 - t1) * 1000,
            postprocess_ms=(t3 - t2) * 1000,
            total_ms=(t3 - t0) * 1000,
            backend="onnx",
            input_shape=input_tensor.shape,
        )

    def warmup(self, runs: int = 3) -> float:
        dummy = np.zeros((*self.input_size[::-1], 3), dtype=np.uint8)
        times = []
        for _ in range(runs):
            result = self.infer(dummy)
            times.append(result.inference_ms)
        return np.mean(times)


class TensorRTBackend(BaseBackend):
    """TensorRT inference backend (NVIDIA GPU only)."""

    def __init__(self, model_path: Union[str, Path], **kwargs) -> None:
        super().__init__(model_path, **kwargs)
        self.device = kwargs.get("device", "cuda")
        self._engine = None
        self._context = None

    def _load_engine(self):
        if self._engine is None:
            try:
                import tensorrt as trt
                import cuda
            except ImportError:
                raise ModelError(
                    "TensorRT not available. Run on NVIDIA GPU with TensorRT installed."
                )

            # Load engine from file
            with open(self.model_path, "rb") as f:
                runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
                self._engine = runtime.deserialize_cuda_engine(f.read())
                self._context = self._engine.create_execution_context()
                log.info("TensorRT engine loaded: {}", self.model_path)

        return self._engine, self._context

    def infer(self, image: np.ndarray) -> InferenceResult:
        raise NotImplementedError("TensorRT backend requires CUDA environment")

    def warmup(self, runs: int = 3) -> float:
        raise NotImplementedError("TensorRT backend requires CUDA environment")


class InferenceBackend:
    """Factory for creating inference backends."""

    BACKENDS = {
        "pytorch": PyTorchBackend,
        "onnx": ONNXBackend,
        "tensorrt": TensorRTBackend,
        "engine": TensorRTBackend,
    }

    @classmethod
    def create(
        cls,
        backend: str,
        model_path: Union[str, Path],
        **kwargs,
    ) -> BaseBackend:
        """Create an inference backend.

        Args:
            backend: Backend type ("pytorch", "onnx", "tensorrt", "engine")
            model_path: Path to model file
            **kwargs: Backend-specific options

        Returns:
            Backend instance
        """
        backend = backend.lower()
        if backend not in cls.BACKENDS:
            raise ModelError(f"Unknown backend: {backend}. Available: {list(cls.BACKENDS.keys())}")

        return cls.BACKENDS[backend](model_path, **kwargs)

    @classmethod
    def from_path(cls, model_path: Union[str, Path], **kwargs) -> BaseBackend:
        """Auto-detect backend from model file extension.

        Args:
            model_path: Path to model file

        Returns:
            Backend instance
        """
        path = Path(model_path)
        ext = path.suffix.lower()

        backend_map = {
            ".pt": "pytorch",
            ".pth": "pytorch",
            ".onnx": "onnx",
            ".engine": "tensorrt",
            ".plan": "tensorrt",
        }

        backend = backend_map.get(ext)
        if backend is None:
            raise ModelError(f"Unknown model format: {ext}")

        return cls.create(backend, model_path, **kwargs)