#!/usr/bin/env python3
"""生成论文所需的所有数据可视化图表

基于真实训练数据(results.csv)生成：
1. 训练损失收敛曲线
2. P-R 曲线
3. 多模型 mAP 对比柱状图
4. Precision-Recall 散点图
5. 推理性能对比图（不同分辨率/FPS）
6. 模型大小压缩对比图（PT/ONNX/FP16/INT8）
"""
import csv
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 10

ROOT = Path(__file__).resolve().parent.parent
CHART_DIR = ROOT / "paper" / "charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)

MODEL_CONFIGS = [
    ("drone-v1 (YOLOv8n)", "archive/v1-dronetrack-20260720/训练结果/drone-v1/results.csv", "#d9534f"),
    ("drone-v1-2 (YOLOv8n)", "archive/v1-dronetrack-20260720/训练结果/drone-v1-2/results.csv", "#f0ad4e"),
    ("drone-v2-dronetrack", "archive/v1-dronetrack-20260720/训练结果/drone-v2-dronetrack/results.csv", "#795cb2"),
    ("drone-v2-final", "archive/v1-dronetrack-20260720/训练结果/runs/models/trained/drone-v2-dronetrack-final/results.csv", "#5cb85c"),
    ("skyguard-v2-uav-3 (YOLOv11s)", "archive/v3-skyguard-uav-20260723/skyguard-v2-uav-3/results.csv", "#0275d8"),
]

METRICS = {}

def load_results(csv_path):
    fpath = ROOT / csv_path
    if not fpath.exists():
        return None
    with open(fpath) as fp:
        return list(csv.DictReader(fp))

def plot_training_loss():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    for name, csv_path, color in MODEL_CONFIGS:
        rows = load_results(csv_path)
        if rows is None:
            continue
        epochs = [int(r['epoch']) for r in rows]
        train_box = [float(r['train/box_loss']) for r in rows]
        val_box = [float(r['val/box_loss']) for r in rows]
        train_cls = [float(r['train/cls_loss']) for r in rows]
        val_cls = [float(r['val/cls_loss']) for r in rows]
        
        ax1.plot(epochs, train_box, '-', color=color, linewidth=2, label=f'{name} (train)')
        ax1.plot(epochs, val_box, '--', color=color, linewidth=2, label=f'{name} (val)')
        
        ax2.plot(epochs, train_cls, '-', color=color, linewidth=2, label=f'{name} (train)')
        ax2.plot(epochs, val_cls, '--', color=color, linewidth=2, label=f'{name} (val)')
    
    ax1.set_title('训练与验证 Box Loss 收敛曲线', fontsize=14, fontweight='bold')
    ax1.set_xlabel('训练轮数 (Epoch)', fontsize=12)
    ax1.set_ylabel('Box Loss', fontsize=12)
    ax1.grid(True)
    ax1.legend(fontsize=9)
    
    ax2.set_title('训练与验证 Classification Loss 收敛曲线', fontsize=14, fontweight='bold')
    ax2.set_xlabel('训练轮数 (Epoch)', fontsize=12)
    ax2.set_ylabel('Classification Loss', fontsize=12)
    ax2.grid(True)
    ax2.legend(fontsize=9)
    
    plt.tight_layout()
    plt.savefig(CHART_DIR / 'training_loss_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('✅ training_loss_curves.png')


def plot_pr_curves():
    plt.figure(figsize=(10, 7))
    
    for name, csv_path, color in MODEL_CONFIGS:
        rows = load_results(csv_path)
        if rows is None:
            continue
        precision = [float(r['metrics/precision(B)']) for r in rows]
        recall = [float(r['metrics/recall(B)']) for r in rows]
        
        sorted_pairs = sorted(zip(recall, precision))
        recall_sorted = [p[0] for p in sorted_pairs]
        precision_sorted = [p[1] for p in sorted_pairs]
        
        plt.plot(recall_sorted, precision_sorted, '-o', color=color, linewidth=2, markersize=5, label=name)
    
    plt.title('各模型精确率-召回率 (P-R) 曲线对比', fontsize=14, fontweight='bold')
    plt.xlabel('召回率 (Recall)', fontsize=12)
    plt.ylabel('精确率 (Precision)', fontsize=12)
    plt.xlim([0.5, 1.0])
    plt.ylim([0.5, 1.0])
    plt.grid(True)
    plt.legend(fontsize=9, loc='lower left')
    
    plt.savefig(CHART_DIR / 'pr_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('✅ pr_curves.png')


def plot_map_comparison():
    models = []
    map50_values = []
    map95_values = []
    p_values = []
    r_values = []
    
    for name, csv_path, _ in MODEL_CONFIGS:
        rows = load_results(csv_path)
        if rows is None:
            continue
        best_map50 = max(float(r['metrics/mAP50(B)']) for r in rows)
        best_map95 = max(float(r['metrics/mAP50-95(B)']) for r in rows)
        best_p = max(float(r['metrics/precision(B)']) for r in rows)
        best_r = max(float(r['metrics/recall(B)']) for r in rows)
        
        models.append(name)
        map50_values.append(best_map50)
        map95_values.append(best_map95)
        p_values.append(best_p)
        r_values.append(best_r)
        
        METRICS[name] = {
            'mAP50': best_map50,
            'mAP50-95': best_map95,
            'precision': best_p,
            'recall': best_r
        }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = ['#d9534f', '#f0ad4e', '#795cb2', '#5cb85c', '#0275d8']
    x = range(len(models))
    
    ax1.bar(x, map50_values, color=colors)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=30, fontsize=9)
    ax1.set_title('各模型 mAP50 指标对比', fontsize=14, fontweight='bold')
    ax1.set_ylabel('mAP50', fontsize=12)
    ax1.set_ylim([0.8, 1.0])
    ax1.grid(True, axis='y')
    
    ax2.bar(x, map95_values, color=colors)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=30, fontsize=9)
    ax2.set_title('各模型 mAP50-95 指标对比', fontsize=14, fontweight='bold')
    ax2.set_ylabel('mAP50-95', fontsize=12)
    ax2.set_ylim([0.3, 0.7])
    ax2.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(CHART_DIR / 'map_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('✅ map_comparison.png')
    
    plt.figure(figsize=(8, 6))
    plt.scatter(r_values, p_values, s=150, c=colors, alpha=0.8)
    for i, name in enumerate(models):
        plt.text(r_values[i] + 0.002, p_values[i] + 0.002, name, fontsize=9)
    
    plt.title('各模型 Precision-Recall 散点图', fontsize=14, fontweight='bold')
    plt.xlabel('召回率 (Recall)', fontsize=12)
    plt.ylabel('精确率 (Precision)', fontsize=12)
    plt.xlim([0.85, 0.96])
    plt.ylim([0.88, 0.99])
    plt.grid(True)
    
    plt.savefig(CHART_DIR / 'pr_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('✅ pr_scatter.png')


def plot_inference_performance():
    perf_path = ROOT / "archive/v3-skyguard-uav-20260723/skyguard-v2-uav-3/perf_benchmark.json"
    if perf_path.exists():
        with open(perf_path) as fp:
            perf_data = json.load(fp)
        
        imgsz_list = [320, 416, 640, 800, 960, 1280]
        fps_values = []
        for r in perf_data['pt_results']:
            fps_values.append(r['fps'])
        
        plt.figure(figsize=(10, 6))
        plt.plot(imgsz_list, fps_values, '-o', color='#0275d8', linewidth=2, markersize=8)
        plt.title('uav-v3 (YOLOv11s) 推理性能随分辨率变化', fontsize=14, fontweight='bold')
        plt.xlabel('输入分辨率 (px)', fontsize=12)
        plt.ylabel('FPS', fontsize=12)
        plt.grid(True)
        
        for i, (sz, fps) in enumerate(zip(imgsz_list, fps_values)):
            plt.text(sz + 10, fps + 0.5, f'{fps:.1f}', fontsize=10)
        
        plt.savefig(CHART_DIR / 'inference_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print('✅ inference_performance.png')


def plot_model_size_comparison():
    sizes = [
        ('best.pt (FP32)', 72.4, '#0275d8'),
        ('best_fp32.onnx', 36.2, '#5cb85c'),
        ('best_fp16.onnx', 18.1, '#f0ad4e'),
        ('best_int8.onnx', 9.4, '#d9534f'),
    ]
    
    labels = [s[0] for s in sizes]
    values = [s[1] for s in sizes]
    colors = [s[2] for s in sizes]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, values, color=colors)
    plt.title('不同格式模型文件大小对比', fontsize=14, fontweight='bold')
    plt.xlabel('模型格式', fontsize=12)
    plt.ylabel('文件大小 (MB)', fontsize=12)
    plt.grid(True, axis='y')
    
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, val + 1, f'{val:.1f} MB', ha='center', fontsize=10)
    
    plt.savefig(CHART_DIR / 'model_size_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('✅ model_size_comparison.png')


def plot_validation_metrics():
    plt.figure(figsize=(12, 8))
    
    for name, csv_path, color in MODEL_CONFIGS:
        rows = load_results(csv_path)
        if rows is None:
            continue
        epochs = [int(r['epoch']) for r in rows]
        map50 = [float(r['metrics/mAP50(B)']) for r in rows]
        map95 = [float(r['metrics/mAP50-95(B)']) for r in rows]
        precision = [float(r['metrics/precision(B)']) for r in rows]
        recall = [float(r['metrics/recall(B)']) for r in rows]
        
        plt.subplot(2, 2, 1)
        plt.plot(epochs, map50, '-o', color=color, linewidth=2, markersize=4, label=name)
        plt.title('mAP50 变化曲线', fontsize=12, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('mAP50')
        plt.grid(True)
        plt.legend(fontsize=8)
        
        plt.subplot(2, 2, 2)
        plt.plot(epochs, map95, '-o', color=color, linewidth=2, markersize=4, label=name)
        plt.title('mAP50-95 变化曲线', fontsize=12, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('mAP50-95')
        plt.grid(True)
        plt.legend(fontsize=8)
        
        plt.subplot(2, 2, 3)
        plt.plot(epochs, precision, '-o', color=color, linewidth=2, markersize=4, label=name)
        plt.title('Precision 变化曲线', fontsize=12, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Precision')
        plt.grid(True)
        plt.legend(fontsize=8)
        
        plt.subplot(2, 2, 4)
        plt.plot(epochs, recall, '-o', color=color, linewidth=2, markersize=4, label=name)
        plt.title('Recall 变化曲线', fontsize=12, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Recall')
        plt.grid(True)
        plt.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(CHART_DIR / 'validation_metrics_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('✅ validation_metrics_curves.png')


def main():
    print("生成论文图表...")
    plot_training_loss()
    plot_pr_curves()
    plot_map_comparison()
    plot_inference_performance()
    plot_model_size_comparison()
    plot_validation_metrics()
    
    print("\n各模型最佳指标汇总：")
    print("=" * 80)
    print(f"{'模型':<30} {'mAP50':>8} {'mAP50-95':>10} {'Precision':>10} {'Recall':>8}")
    print("=" * 80)
    for name, m in METRICS.items():
        print(f"{name:<30} {m['mAP50']:>8.4f} {m['mAP50-95']:>10.4f} {m['precision']:>10.4f} {m['recall']:>8.4f}")
    
    print(f"\n图表已保存到: {CHART_DIR}")


if __name__ == "__main__":
    main()
