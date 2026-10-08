"""
Benchmark SkyGuard inference backends.

Compare performance across:
  * PyTorch (native)
  * ONNX Runtime (CPU)
  * TensorRT (NVIDIA GPU, if available)

Usage:
  python scripts/benchmark_inference.py --model models/trained/drone-v1-2/weights/best.pt
  python scripts/benchmark_inference.py --model models/trained/best.onnx --backend onnx
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for p in [str(SRC), str(ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np


def benchmark_backend(backend, image: np.ndarray, runs: int = 50) -> dict:
    """Run benchmark on a backend."""
    # Warmup
    backend.warmup(runs=5)

    # Benchmark
    times = []
    for _ in range(runs):
        result = backend.infer(image)
        times.append(result.inference_ms)

    return {
        "backend": result.backend,
        "runs": runs,
        "mean_ms": np.mean(times),
        "std_ms": np.std(times),
        "min_ms": np.min(times),
        "max_ms": np.max(times),
        "fps": 1000.0 / np.mean(times),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark SkyGuard inference backends")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--backend", type=str, default=None, help="Force specific backend")
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    # Create dummy image
    image = np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8)

    # Import backend
    from skyguard.inference.backend import InferenceBackend

    # Create backend
    if args.backend:
        backend = InferenceBackend.create(args.backend, args.model, input_size=(args.imgsz, args.imgsz))
    else:
        backend = InferenceBackend.from_path(args.model, input_size=(args.imgsz, args.imgsz))

    print(f"\nBenchmark: {args.model}")
    print(f"Backend: {backend.__class__.__name__}")
    print(f"Image size: {args.imgsz}x{args.imgsz}")
    print(f"Runs: {args.runs}\n")

    # Run benchmark
    results = benchmark_backend(backend, image, runs=args.runs)

    print("-" * 50)
    print(f"Backend:     {results['backend']}")
    print(f"Mean:        {results['mean_ms']:.2f} ms")
    print(f"Std:         {results['std_ms']:.2f} ms")
    print(f"Min:         {results['min_ms']:.2f} ms")
    print(f"Max:         {results['max_ms']:.2f} ms")
    print(f"FPS:         {results['fps']:.1f}")
    print("-" * 50)

    return 0


if __name__ == "__main__":
    sys.exit(main())