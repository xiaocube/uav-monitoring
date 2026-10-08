"""
SkyGuard optimized inference backends.

Sprint 5: High-performance inference with ONNX Runtime and TensorRT.

Backends:
  * PyTorch (.pt)      -- native PyTorch inference (development)
  * ONNX Runtime       -- cross-platform CPU inference (production)
  * TensorRT (.engine) -- NVIDIA GPU optimized (Jetson / CUDA servers)
  * OpenVINO           -- Intel CPU/GPU optimization (optional)
  * CoreML             -- Apple Silicon optimization (optional)
"""
from skyguard.inference.backend import InferenceBackend

__all__ = ["InferenceBackend"]