#!/bin/bash
# ============================================================
# SkyGuard UAV v4 训练启动脚本
# ============================================================
# 配置: config/training/uav_v4.yaml
# 输出: runs/detect/train/uav-v4/
#
# 安全停止: 在本终端按 Ctrl+C, 会自动保存 last.pt
# 监控: 在另一个终端运行 tensorboard --logdir runs/detect/train/uav-v4
# 续训: bash scripts/train/train_uav_v4.sh --resume
# ============================================================

set -e

cd /Users/panyanming/Desktop/无人机反制

# 激活虚拟环境
source .venv/bin/activate

# 检查配置文件
CONFIG="config/training/uav_v4.yaml"
if [ ! -f "$CONFIG" ]; then
    echo "错误: 配置文件 $CONFIG 不存在"
    exit 1
fi

# 检查数据集
DATASET="data/skyguard-v2/dataset.yaml"
if [ ! -f "$DATASET" ]; then
    echo "错误: 数据集配置 $DATASET 不存在"
    exit 1
fi

# 检查基础模型
if [ ! -f "yolo11s.pt" ]; then
    echo "下载 yolo11s.pt 预训练权重..."
    wget -q https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt
fi

echo "============================================================"
echo "SkyGuard UAV v4 训练"
echo "============================================================"
echo "配置: $CONFIG"
echo "数据集: $DATASET"
echo "输出: runs/detect/train/uav-v4/"
echo ""
echo "安全停止: 按 Ctrl+C (会自动保存 last.pt)"
echo "监控: tensorboard --logdir runs/detect/train/uav-v4"
echo "============================================================"
echo ""

# 续训模式
RESUME_FLAG=""
if [ "$1" = "--resume" ]; then
    RESUME_FLAG="resume=/Users/panyanming/Desktop/无人机反制/runs/detect/train/uav-v4/weights/last.pt"
    echo ">>> 续训模式: 从 last.pt 恢复"
    echo ""
fi

# 启动训练 (直接调用 ultralytics, 确保参数完整传递)
# 注意: project 用绝对路径避免 ultralytics 自动加 runs/detect/ 前缀
yolo train \
    model=yolo11s.pt \
    data=data/skyguard-v2/dataset.yaml \
    project=/Users/panyanming/Desktop/无人机反制/runs/detect/train \
    name=uav-v4 \
    exist_ok=True \
    epochs=50 \
    patience=15 \
    batch=16 \
    workers=0 \
    device=mps \
    imgsz=640 \
    optimizer=AdamW \
    lr0=0.001 \
    lrf=0.01 \
    momentum=0.937 \
    weight_decay=0.0005 \
    cos_lr=True \
    warmup_epochs=3 \
    warmup_momentum=0.8 \
    warmup_bias_lr=0.1 \
    hsv_h=0.015 \
    hsv_s=0.7 \
    hsv_v=0.4 \
    degrees=5.0 \
    translate=0.1 \
    scale=0.5 \
    shear=2.0 \
    perspective=0.0 \
    flipud=0.0 \
    fliplr=0.5 \
    mosaic=1.0 \
    mixup=0.1 \
    copy_paste=0.0 \
    close_mosaic=10 \
    val=True \
    iou=0.6 \
    max_det=300 \
    save=True \
    save_period=5 \
    pretrained=True \
    seed=42 \
    deterministic=True \
    amp=True \
    fraction=1.0 \
    dropout=0.0 \
    verbose=True \
    plots=True \
    $RESUME_FLAG

echo ""
echo "============================================================"
echo "训练完成!"
echo "============================================================"
echo "最佳权重: runs/detect/train/uav-v4/weights/best.pt"
echo "最终权重: runs/detect/train/uav-v4/weights/last.pt"
echo "训练曲线: runs/detect/train/uav-v4/results.csv"
echo "============================================================"
