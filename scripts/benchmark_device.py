"""
Run a small torch matmul micro-benchmark on each available backend.

This is NOT a deep perf test — just enough to confirm the selected device
actually executes tensors and to give us a sane baseline for later sprints.

Usage:
  python scripts/benchmark_device.py [--size 1024] [--iters 50]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from skyguard.utils.device import detect_device  # noqa: E402


def _bench(device: str, size: int, iters: int) -> Optional[float]:
    try:
        import torch
    except Exception as e:
        print(f"  [skip {device}] torch unavailable: {e}")
        return None

    try:
        dev = torch.device(device)
    except Exception as e:
        print(f"  [skip {device}] invalid device: {e}")
        return None

    try:
        a = torch.randn(size, size, device=dev)
        b = torch.randn(size, size, device=dev)
    except Exception as e:
        print(f"  [skip {device}] tensor alloc failed: {e}")
        return None

    # Warm-up
    for _ in range(3):
        _ = a @ b
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        try:
            torch.mps.synchronize()
        except AttributeError:
            pass

    start = time.perf_counter()
    for _ in range(iters):
        c = a @ b
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        try:
            torch.mps.synchronize()
        except AttributeError:
            pass
    elapsed = time.perf_counter() - start
    ms_per_iter = (elapsed / iters) * 1000.0
    gflops = 2 * (size ** 3) / (elapsed / iters) / 1e9
    print(
        f"  [{device:<6}] size={size}x{size} iters={iters} "
        f"-> {ms_per_iter:8.2f} ms/iter  ~{gflops:7.2f} GFLOPS"
    )
    return ms_per_iter


def main() -> int:
    parser = argparse.ArgumentParser(description="SkyGuard device micro-benchmark")
    parser.add_argument("--size", type=int, default=1024, help="square matrix size")
    parser.add_argument("--iters", type=int, default=30, help="iterations")
    args = parser.parse_args()

    info = detect_device("auto")
    print(f"Selected device: {info.selected}")
    print(f"  torch       : {info.torch_version}")
    print(f"  cuda_avail  : {info.cuda_available}")
    print(f"  mps_avail   : {info.mps_available}")
    print(f"  cpu_count   : {info.cpu_count}")
    print()

    print("Running matmul benchmark:")
    if info.selected == "cuda":
        _bench("cuda:0", args.size, args.iters)
    elif info.selected == "mps":
        _bench("mps", args.size, args.iters)
    else:
        _bench("cpu", args.size, args.iters)
    return 0


if __name__ == "__main__":
    sys.exit(main())
