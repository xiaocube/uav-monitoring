#!/usr/bin/env python3
"""SkyGuard 模型性能压测脚本

测试不同分辨率下的推理性能：
- FPS / 单帧推理延迟
- 内存占用
- 模型大小对比
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import psutil
import torch

ROOT = Path(__file__).resolve().parent.parent

def get_memory_usage() -> dict:
    """获取当前进程内存占用"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return {
        "rss_mb": round(mem_info.rss / 1024 / 1024, 1),
        "vms_mb": round(mem_info.vms / 1024 / 1024, 1),
    }


def benchmark_model(model_path: str, imgsz_list: list, device: str, n_warmup: int = 5, n_runs: int = 30) -> list:
    """测试模型在不同分辨率下的推理性能"""
    from ultralytics import YOLO

    results = []
    model = YOLO(model_path)
    model_name = Path(model_path).name

    # 生成测试图片
    dummy_img = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

    for imgsz in imgsz_list:
        print(f"\n  测试 {model_name} @ imgsz={imgsz} ...")

        # Warmup
        for _ in range(n_warmup):
            model.predict(dummy_img, imgsz=imgsz, device=device, verbose=False)

        # Benchmark
        latencies = []
        mem_before = get_memory_usage()

        for _ in range(n_runs):
            t0 = time.perf_counter()
            model.predict(dummy_img, imgsz=imgsz, device=device, verbose=False)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)

        mem_after = get_memory_usage()

        latencies.sort()
        avg_ms = sum(latencies) / len(latencies)
        p50_ms = latencies[len(latencies) // 2]
        p95_ms = latencies[int(len(latencies) * 0.95)]
        p99_ms = latencies[int(len(latencies) * 0.99)]
        fps = 1000.0 / avg_ms

        result = {
            "model": model_name,
            "imgsz": imgsz,
            "avg_ms": round(avg_ms, 1),
            "p50_ms": round(p50_ms, 1),
            "p95_ms": round(p95_ms, 1),
            "p99_ms": round(p99_ms, 1),
            "fps": round(fps, 1),
            "mem_rss_mb": mem_after["rss_mb"],
            "mem_delta_mb": round(mem_after["rss_mb"] - mem_before["rss_mb"], 1),
        }
        results.append(result)
        print(f"    FPS: {fps:.1f} | Avg: {avg_ms:.1f}ms | P50: {p50_ms:.1f}ms | P95: {p95_ms:.1f}ms | Mem: {mem_after['rss_mb']}MB")

    return results


def main():
    parser = argparse.ArgumentParser(description="SkyGuard 模型性能压测")
    parser.add_argument("--model", type=str,
                        default=str(ROOT / "models/current/best_v3.pt"))
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--output", type=str,
                        default=str(ROOT / "runs/detect/train/uav-v4/perf_benchmark.json"))
    args = parser.parse_args()

    import json

    print("=" * 70)
    print("  SkyGuard 模型性能压测")
    print("=" * 70)
    print(f"模型: {args.model}")
    print(f"设备: {args.device}")
    print(f"PyTorch: {torch.__version__}")
    if torch.backends.mps.is_available():
        print("MPS: 可用 (Apple M4)")
    print()

    # 模型文件大小
    model_size_mb = Path(args.model).stat().st_size / 1024 / 1024
    print(f"模型文件大小: {model_size_mb:.1f} MB")
    print()

    # 测试不同分辨率
    imgsz_list = [320, 416, 640, 800, 960, 1280]
    print("[1/2] 测试 PyTorch (.pt) 模型...")
    pt_results = benchmark_model(args.model, imgsz_list, args.device)

    all_results = {
        "model_path": args.model,
        "model_size_mb": round(model_size_mb, 1),
        "device": args.device,
        "torch_version": torch.__version__,
        "pt_results": pt_results,
    }

    # 如果有 ONNX 模型也测试
    onnx_path = str(Path(args.model).with_suffix(".onnx"))
    if Path(onnx_path).exists():
        print(f"\n[2/2] 测试 ONNX 模型: {onnx_path}")
        try:
            onnx_results = benchmark_model(onnx_path, imgsz_list, "cpu")
            all_results["onnx_results"] = onnx_results
        except Exception as e:
            print(f"  ONNX 测试失败: {e}")
    else:
        print(f"\n[2/2] ONNX 模型不存在，跳过")

    # 保存结果
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # 打印汇总表
    print("\n" + "=" * 70)
    print("  性能压测汇总")
    print("=" * 70)
    print(f"{'模型':<20} {'分辨率':>6} {'FPS':>8} {'Avg(ms)':>8} {'P50(ms)':>8} {'P95(ms)':>8} {'内存(MB)':>10}")
    print("-" * 70)
    for r in pt_results:
        print(f"{r['model']:<20} {r['imgsz']:>6} {r['fps']:>8.1f} {r['avg_ms']:>8.1f} {r['p50_ms']:>8.1f} {r['p95_ms']:>8.1f} {r['mem_rss_mb']:>10.1f}")

    print(f"\n结果已保存到: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
