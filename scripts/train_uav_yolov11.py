#!/usr/bin/env python3
"""
SkyGuard YOLOv11 训练脚本 - UAV 高空航拍小目标检测
=====================================================
运行环境: Mac M4 16GB (Metal MPS 加速)
框架: ultralytics YOLOv11

使用方法:
  # 标准模式（推荐）
  python scripts/train_uav_yolov11.py

  # 降级模式（内存吃紧时使用）
  python scripts/train_uav_yolov11.py --degraded

环境变量控制:
  PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.4  # 更保守的内存限制
  PYTORCH_ENABLE_MPS_FALLBACK=1         # 允许回退到CPU
"""

import os
import sys
import time
import warnings
import argparse
from pathlib import Path

# 抑制 MPS 相关警告
warnings.filterwarnings("ignore", category=UserWarning, message=".*MPS.*")

# 确保项目根目录在 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# 标准训练参数（Mac M4 16GB 优化）
# ============================================================
STANDARD_ARGS = {
    # 模型与数据
    "model": "yolo11s.pt",
    "data": "data/skyguard-v2/dataset.yaml",

    # 输出
    "project": "models/trained",
    "name": "skyguard-v2-uav",
    "exist_ok": False,

    # 图像
    "imgsz": 640,

    # 训练轮数
    "epochs": 100,
    "patience": 20,

    # M4 MPS 优化
    "batch": 8,
    "workers": 0,          # Mac 上设为 0 避免 dataloader 卡死
    "device": "mps",

    # 优化器
    "optimizer": "AdamW",
    "lr0": 0.001,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "cos_lr": True,
    "warmup_epochs": 3,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,

    # 数据增强（轻量化，避免 M4 内存溢出）
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 10.0,
    "translate": 0.1,
    "scale": 0.5,          # 多尺度训练（640下适度缩放）
    "shear": 2.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "mosaic": 1.0,         # 小目标增强核心
    "mixup": 0.1,
    "copy_paste": 0.0,     # 关闭：M4 16GB 下内存占用过高
    "close_mosaic": 10,

    # 验证
    "val": True,
    "iou": 0.6,
    "max_det": 300,

    # 保存
    "save": True,
    "save_period": 10,
    "pretrained": True,
    "resume": False,
    "seed": 42,
    "deterministic": True,

    # AMP
    "amp": True,

    # 其他
    "fraction": 1.0,
    "dropout": 0.0,
    "verbose": True,
    "plots": True,
    "single_cls": False,
    "rect": False,
}


# ============================================================
# 降级应急参数（内存持续吃紧时使用）
# ============================================================
DEGRADED_ARGS = {
    "model": "yolo11s.pt",
    "data": "data/skyguard-v2/dataset.yaml",
    "project": "models/trained",
    "name": "skyguard-v2-uav",
    "exist_ok": False,

    "imgsz": 576,          # 降低分辨率减少内存
    "epochs": 100,
    "patience": 20,

    "batch": 4,            # 减半 batch
    "workers": 0,
    "device": "mps",

    "optimizer": "AdamW",
    "lr0": 0.001,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "cos_lr": True,
    "warmup_epochs": 3,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,

    # 降级增强：关闭多尺度，仅保留基本增强
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 10.0,
    "translate": 0.1,
    "scale": 0.0,          # 关闭多尺度训练
    "shear": 2.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "mosaic": 1.0,
    "mixup": 0.0,          # 关闭 mixup
    "copy_paste": 0.0,
    "close_mosaic": 10,

    "val": True,
    "iou": 0.6,
    "max_det": 300,

    "save": True,
    "save_period": 10,
    "pretrained": True,
    "resume": False,
    "seed": 42,
    "deterministic": True,

    "amp": False,          # 关闭 AMP 避免不稳定
    "fraction": 1.0,
    "dropout": 0.0,
    "verbose": True,
    "plots": True,
    "single_cls": False,
    "rect": False,
}


def setup_mac_m4_environment(degraded=False):
    """配置 Mac M4 训练环境，避免常见 MPS 报错"""
    import torch

    print("=" * 60)
    print("🍎 Mac M4 环境检测")
    print("=" * 60)

    if torch.backends.mps.is_available():
        print("✅ MPS (Metal Performance Shaders) 可用")
        print(f"   PyTorch: {torch.__version__}")
    else:
        print("❌ MPS 不可用！将回退到 CPU 训练（速度极慢）")
        print("   请确认 PyTorch >= 2.1 已安装")

    # MPS 内存限制（PyTorch 2.8 下 HIGH_WATERMARK_RATIO 可能报错，先不设置）
    # os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", ratio)
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    print(f"   PYTORCH_ENABLE_MPS_FALLBACK = {os.environ['PYTORCH_ENABLE_MPS_FALLBACK']}")

    # 磁盘空间检查
    import shutil
    disk_usage = shutil.disk_usage(str(PROJECT_ROOT))
    free_gb = disk_usage.free / (1024**3)
    print(f"   磁盘可用空间: {free_gb:.1f} GB")
    if free_gb < 10:
        print("⚠️  磁盘空间不足 10GB，训练可能失败！")

    return torch.backends.mps.is_available()


def check_dataset():
    """检查数据集是否存在"""
    yaml_path = PROJECT_ROOT / "data" / "skyguard-v2" / "dataset.yaml"
    if not yaml_path.exists():
        print("\n❌ 数据集未准备！请先运行:")
        print("   python scripts/prepare_dataset.py")
        sys.exit(1)
    print(f"✅ 数据集配置: {yaml_path}")


def main():
    parser = argparse.ArgumentParser(description="SkyGuard YOLOv11 训练")
    parser.add_argument("--degraded", action="store_true",
                        help="降级模式：batch=4, imgsz=576, 关闭多尺度/AMP")
    args = parser.parse_args()

    setup_mac_m4_environment(degraded=args.degraded)
    check_dataset()

    from ultralytics import YOLO

    train_args = DEGRADED_ARGS if args.degraded else STANDARD_ARGS
    mode = "🟡 降级模式" if args.degraded else "🟢 标准模式"

    print(f"\n{'=' * 60}")
    print(f"🚁 SkyGuard YOLOv11 UAV 检测模型训练 - {mode}")
    print(f"{'=' * 60}")

    print(f"\n📋 训练参数:")
    for k, v in train_args.items():
        print(f"   {k}: {v}")

    imgsz = train_args["imgsz"]
    batch = train_args["batch"]
    print(f"\n⏳ 开始训练... (预计 8000 张 @{imgsz} batch={batch} 在 M4 上约 1-2 小时)")

    start_time = time.time()

    try:
        model = YOLO(train_args["model"])
        results = model.train(**{k: v for k, v in train_args.items() if k != "model"})

        elapsed = time.time() - start_time
        print(f"\n{'=' * 60}")
        print(f"✅ 训练完成！总耗时: {elapsed/60:.1f} 分钟")
        print(f"📂 模型保存路径: models/trained/skyguard-v2-uav/weights/")
        print(f"{'=' * 60}")

    except RuntimeError as e:
        error_msg = str(e)
        print(f"\n❌ 训练出错: {error_msg}")

        if "MPS" in error_msg or "out of memory" in error_msg.lower():
            print("\n🔧 MPS 内存溢出 → 请使用降级模式:")
            print("   python scripts/train_uav_yolov11.py --degraded")
        elif "dataloader" in error_msg.lower():
            print("\n🔧 DataLoader 问题:")
            print("   1. 确认 workers=0")
            print("   2. 运行 python scripts/validate_dataset.py 检查数据")
        raise


if __name__ == "__main__":
    main()