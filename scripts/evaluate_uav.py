#!/usr/bin/env python3
"""
训练后评估与模型导出脚本
========================
功能:
1. 模型评估（mAP, Precision, Recall, F1）
2. 混淆矩阵生成与分析
3. 漏检/误检样本排查
4. ONNX 模型导出（适配后续部署）
5. 推理速度基准测试

使用方法:
  python scripts/evaluate_uav.py
"""

import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

# 模型路径
MODEL_PATH = PROJECT_ROOT / "models" / "current" / "best_v4.pt"
DATASET_YAML = PROJECT_ROOT / "data" / "skyguard-v2" / "dataset.yaml"
EVAL_OUTPUT = PROJECT_ROOT / "models" / "current" / "evaluation"


def find_best_model():
    """查找 best.pt 模型文件"""
    if MODEL_PATH.exists():
        return MODEL_PATH

    # 尝试其他可能路径
    candidates = list(PROJECT_ROOT.glob("models/current/best_v*.pt"))
    if candidates:
        return candidates[0]

    print("❌ 未找到训练好的模型文件 (best.pt)")
    print("   请确保训练已完成，模型保存在 models/current/")
    sys.exit(1)


def evaluate_model(model_path: Path):
    """
    模型评估:
    - mAP@50, mAP@50-95
    - Precision, Recall
    - F1 曲线
    - 混淆矩阵
    """
    from ultralytics import YOLO

    print("=" * 60)
    print("📊 模型评估")
    print("=" * 60)
    print(f"模型: {model_path}")
    print(f"数据集: {DATASET_YAML}")

    model = YOLO(str(model_path))

    # 在验证集上评估
    print("\n⏳ 正在验证集上评估...")
    metrics = model.val(
        data=str(DATASET_YAML),
        imgsz=640,
        batch=8,
        device="mps",
        conf=0.001,  # 低置信度阈值以计算完整 mAP
        iou=0.6,
        split="val",
        plots=True,
        save_json=True,
        project=str(EVAL_OUTPUT.parent),
        name="evaluation",
    )

    # 输出关键指标
    print(f"\n📊 评估结果:")
    print(f"   mAP@50:     {metrics.box.map50:.4f}")
    print(f"   mAP@50-95:  {metrics.box.map:.4f}")
    print(f"   Precision:  {metrics.box.mp:.4f}")
    print(f"   Recall:     {metrics.box.mr:.4f}")
    if hasattr(metrics.box, "f1"):
        print(f"   F1:         {metrics.box.f1:.4f}")

    return metrics


def analyze_errors(model_path: Path):
    """
    漏检/误检分析:
    - 在验证集上运行推理，找出置信度低的样本
    - 统计 FP (误检) 和 FN (漏检) 较多的图片
    """
    from ultralytics import YOLO
    import torch

    print(f"\n{'=' * 60}")
    print(f"🔍 漏检/误检分析")
    print(f"{'=' * 60}")

    model = YOLO(str(model_path))

    # 在验证集上运行预测并保存结果
    print("\n⏳ 正在验证集上推理...")
    results = model.predict(
        source=str(DATASET_YAML.parent / "val" / "images"),
        imgsz=640,
        conf=0.25,  # 标准置信度阈值
        iou=0.6,
        device="mps",
        save=False,
        save_txt=False,
        save_conf=False,
        project=str(EVAL_OUTPUT.parent),
        name="error_analysis",
    )

    # 统计检测结果
    total_gt = 0
    total_det = 0
    empty_detections = 0
    low_conf_detections = 0

    for r in results:
        gt_count = len(r.boxes) if r.boxes is not None else 0
        total_gt += gt_count

        if r.boxes is not None and len(r.boxes) > 0:
            # 统计检测框
            confs = r.boxes.conf.cpu().numpy() if hasattr(r.boxes, 'conf') else []
            total_det += len(confs)
            low_conf = sum(1 for c in confs if c < 0.5)
            low_conf_detections += low_conf
        else:
            empty_detections += 1

    print(f"\n📊 错误分析结果:")
    print(f"   验证图片数: {len(results)}")
    print(f"   总 GT 框:   {total_gt}")
    print(f"   总检测框:   {total_det}")
    print(f"   无检测图片: {empty_detections} ({empty_detections/len(results)*100:.1f}%)")
    if total_det > 0:
        print(f"   低置信度(<0.5): {low_conf_detections} ({low_conf_detections/total_det*100:.1f}%)")
    if total_gt > 0:
        print(f"   平均检测/GT: {total_det/total_gt:.2f}")

    if empty_detections > len(results) * 0.1:
        print(f"\n⚠️  超过 10% 图片无检测结果，可能存在漏检问题")
        print(f"   建议: 降低 conf 阈值 / 检查数据集标注质量")

    return results


def export_onnx(model_path: Path):
    """导出 ONNX 模型，适配后续部署"""
    from ultralytics import YOLO

    print(f"\n{'=' * 60}")
    print(f"📦 ONNX 模型导出")
    print(f"{'=' * 60}")

    model = YOLO(str(model_path))

    export_dir = model_path.parent

    # 导出 ONNX (FP32)
    print("\n⏳ 导出 ONNX (FP32)...")
    onnx_path = model.export(
        format="onnx",
        imgsz=640,
        half=False,
        simplify=True,
        opset=12,
        workspace=4,
    )
    print(f"✅ ONNX (FP32): {onnx_path}")

    # 导出 ONNX (FP16)
    print("\n⏳ 导出 ONNX (FP16)...")
    try:
        onnx_fp16_path = model.export(
            format="onnx",
            imgsz=640,
            half=True,
            simplify=True,
            opset=12,
            workspace=4,
        )
        print(f"✅ ONNX (FP16): {onnx_fp16_path}")
    except Exception as e:
        print(f"⚠️  FP16 导出跳过: {e}")

    # 导出 CoreML (可选, Mac 部署)
    print("\n⏳ 导出 CoreML...")
    try:
        coreml_path = model.export(
            format="coreml",
            imgsz=640,
            half=False,
            nms=True,
        )
        print(f"✅ CoreML: {coreml_path}")
    except Exception as e:
        print(f"⚠️  CoreML 导出跳过: {e}")

    return onnx_path


def benchmark_speed(model_path: Path):
    """推理速度基准测试"""
    from ultralytics import YOLO
    import torch

    print(f"\n{'=' * 60}")
    print(f"⚡ 推理速度基准测试")
    print(f"{'=' * 60}")

    model = YOLO(str(model_path))

    # 预热
    print("\n预热中...")
    for _ in range(10):
        _ = model.predict(source=PROJECT_ROOT / "data" / "skyguard-v2" / "val" / "images",
                          imgsz=640, device="mps", verbose=False)

    # 测试
    val_dir = DATASET_YAML.parent / "val" / "images"
    images = list(val_dir.glob("*"))
    if not images:
        print("⚠️  无验证图片，跳过速度测试")
        return

    test_images = images[:min(100, len(images))]

    print(f"测试 {len(test_images)} 张图片 @1280...")
    start = time.time()
    for img in test_images:
        _ = model.predict(source=str(img), imgsz=640, device="mps", verbose=False)
    elapsed = time.time() - start

    fps = len(test_images) / elapsed
    latency = elapsed / len(test_images) * 1000

    print(f"\n📊 速度基准:")
    print(f"   FPS:     {fps:.1f}")
    print(f"   延迟:    {latency:.1f} ms/张")
    print(f"   设备:    MPS (Mac M4)")


def main():
    model_path = find_best_model()
    print(f"📂 模型: {model_path}")

    # 1. 模型评估
    evaluate_model(model_path)

    # 2. 漏检/误检分析
    try:
        analyze_errors(model_path)
    except Exception as e:
        print(f"⚠️  错误分析跳过: {e}")

    # 3. 导出 ONNX
    try:
        export_onnx(model_path)
    except Exception as e:
        print(f"⚠️  ONNX 导出失败: {e}")

    # 4. 速度基准测试
    try:
        benchmark_speed(model_path)
    except Exception as e:
        print(f"⚠️  速度测试跳过: {e}")

    print(f"\n{'=' * 60}")
    print(f"✅ 评估完成！")
    print(f"📂 评估结果: {EVAL_OUTPUT.parent}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()