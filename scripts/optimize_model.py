"""
性能优化脚本: 导出模型到多种格式 + 推理基准测试.

用法:
  python scripts/optimize_model.py \
      --model models/trained/drone-v2-dronetrack/weights/best.pt \
      --formats onnx coreml torchscript \
      --imgsz 1280 \
      --benchmark
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in [str(SRC), str(ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from skyguard.core.logger import setup_logging
from skyguard.training.export import ModelExporter


def run_benchmark(model_path: Path, imgsz: int = 1280, n_runs: int = 50) -> None:
    """推理基准测试."""
    from ultralytics import YOLO
    import numpy as np

    print(f"\n{'='*60}")
    print(f"推理基准测试: {model_path.name}")
    print(f"{'='*60}")

    model = YOLO(str(model_path))
    dummy = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    # 预热
    for _ in range(5):
        model.predict(dummy, imgsz=imgsz, verbose=False)

    # 正式测试
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model.predict(dummy, imgsz=imgsz, verbose=False)
        times.append((time.perf_counter() - t0) * 1000)

    times = np.array(times)
    print(f"  推理次数: {n_runs}")
    print(f"  平均耗时: {np.mean(times):.1f} ms")
    print(f"  中位数:   {np.median(times):.1f} ms")
    print(f"  P95:      {np.percentile(times, 95):.1f} ms")
    print(f"  FPS:      {1000 / np.mean(times):.1f}")
    print(f"{'='*60}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="模型导出与性能优化")
    parser.add_argument("--model", type=Path, required=True, help="模型路径 (.pt)")
    parser.add_argument(
        "--formats", nargs="+", default=["onnx"],
        choices=["onnx", "coreml", "torchscript", "openvino", "engine"],
        help="导出格式",
    )
    parser.add_argument("--imgsz", type=int, default=1280, help="推理分辨率")
    parser.add_argument("--half", action="store_true", default=True, help="FP16 半精度")
    parser.add_argument("--int8", action="store_true", default=False, help="INT8 量化")
    parser.add_argument("--benchmark", action="store_true", help="运行推理基准测试")
    parser.add_argument("--benchmark-runs", type=int, default=50, help="基准测试次数")
    args = parser.parse_args()

    setup_logging()

    if not args.model.exists():
        print(f"错误: 模型不存在: {args.model}", file=sys.stderr)
        return 1

    # 导出
    print(f"\n导出模型: {args.model.name}")
    print(f"格式: {', '.join(args.formats)}")
    print(f"分辨率: {args.imgsz}, FP16: {args.half}, INT8: {args.int8}\n")

    exporter = ModelExporter({
        "formats": args.formats,
        "imgsz": args.imgsz,
        "half": args.half,
        "int8": args.int8,
        "simplify": True,
        "opset": 12,
    })

    exported = exporter.export(args.model)

    print(f"\n导出完成:")
    for p in exported:
        size_mb = p.stat().st_size / 1024 / 1024
        print(f"  {p.name:30s}  {size_mb:.1f} MB  →  {p}")

    # 基准测试
    if args.benchmark:
        run_benchmark(args.model, args.imgsz, args.benchmark_runs)
        # 如果有 ONNX 导出,也测一下
        onnx_path = args.model.with_suffix(".onnx")
        if onnx_path.exists():
            print("  [ONNX 格式基准测试]")
            try:
                import onnxruntime as ort
                import numpy as np

                sess = ort.InferenceSession(str(onnx_path))
                inp = sess.get_inputs()[0]
                dummy = np.random.randn(1, 3, args.imgsz, args.imgsz).astype(np.float32)

                # 预热
                for _ in range(5):
                    sess.run(None, {inp.name: dummy})

                times = []
                for _ in range(args.benchmark_runs):
                    t0 = time.perf_counter()
                    sess.run(None, {inp.name: dummy})
                    times.append((time.perf_counter() - t0) * 1000)

                times = np.array(times)
                print(f"  ONNX 平均: {np.mean(times):.1f} ms, FPS: {1000/np.mean(times):.1f}")
            except ImportError:
                print("  onnxruntime 未安装,跳过 ONNX 基准测试")

    return 0


if __name__ == "__main__":
    sys.exit(main())
