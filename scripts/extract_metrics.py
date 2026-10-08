#!/usr/bin/env python3
"""提取所有模型版本的训练指标"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

models = [
    ("drone-v1 (YOLOv8n)", "archive/v1-dronetrack-20260720/训练结果/drone-v1/results.csv"),
    ("drone-v1-2 (YOLOv8n)", "archive/v1-dronetrack-20260720/训练结果/drone-v1-2/results.csv"),
    ("drone-v2-dronetrack (续训)", "archive/v1-dronetrack-20260720/训练结果/drone-v2-dronetrack/results.csv"),
    ("drone-v2-final (续训)", "archive/v1-dronetrack-20260720/训练结果/runs/models/trained/drone-v2-dronetrack-final/results.csv"),
    ("skyguard-v2-uav-3 (YOLOv11s)", "archive/v3-skyguard-uav-20260723/skyguard-v2-uav-3/results.csv"),
]

print("=" * 90)
print(f"{'模型':<30} {'Epochs':>7} {'mAP50':>8} {'mAP50-95':>10} {'P':>8} {'R':>8} {'数据来源':>15}")
print("=" * 90)

for name, rel_path in models:
    fpath = ROOT / rel_path
    if not fpath.exists():
        print(f"{name:<30} 文件不存在: {fpath}")
        continue
    with open(fpath) as fp:
        rows = list(csv.DictReader(fp))

    best_map50 = max(rows, key=lambda r: float(r['metrics/mAP50(B)']))
    best_map95 = max(rows, key=lambda r: float(r['metrics/mAP50-95(B)']))
    best_p = max(rows, key=lambda r: float(r['metrics/precision(B)']))
    best_r = max(rows, key=lambda r: float(r['metrics/recall(B)']))

    print(f"{name:<30} {len(rows):>7} {float(best_map50['metrics/mAP50(B)']):>8.4f} {float(best_map95['metrics/mAP50-95(B)']):>10.4f} {float(best_p['metrics/precision(B)']):>8.4f} {float(best_r['metrics/recall(B)']):>8.4f}")

print()
print("=== 各模型详细最佳指标 ===")
for name, rel_path in models:
    fpath = ROOT / rel_path
    if not fpath.exists():
        continue
    with open(fpath) as fp:
        rows = list(csv.DictReader(fp))
    best_map50 = max(rows, key=lambda r: float(r['metrics/mAP50(B)']))
    best_map95 = max(rows, key=lambda r: float(r['metrics/mAP50-95(B)']))
    best_p = max(rows, key=lambda r: float(r['metrics/precision(B)']))
    best_r = max(rows, key=lambda r: float(r['metrics/recall(B)']))
    last = rows[-1]

    print(f"\n--- {name} ---")
    print(f"  总Epoch: {len(rows)}")
    print(f"  最佳 mAP50:    {best_map50['metrics/mAP50(B)']} (Epoch {best_map50['epoch']})")
    print(f"  最佳 mAP50-95: {best_map95['metrics/mAP50-95(B)']} (Epoch {best_map95['epoch']})")
    print(f"  最佳 Precision: {best_p['metrics/precision(B)']} (Epoch {best_p['epoch']})")
    print(f"  最佳 Recall:    {best_r['metrics/recall(B)']} (Epoch {best_r['epoch']})")
    print(f"  末轮 mAP50:    {last['metrics/mAP50(B)']} (Epoch {last['epoch']})")
    print(f"  末轮 train/box_loss: {float(last['train/box_loss']):.4f}")
    print(f"  末轮 val/box_loss:   {float(last['val/box_loss']):.4f}")
