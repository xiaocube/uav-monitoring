"""YOLOv11 无人机检测模型训练结果可视化"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

df = pd.read_csv('archive/v3-skyguard-uav-20260723/skyguard-v2-uav-3/results.csv')
df.columns = df.columns.str.strip()

epoch = df['epoch']
train_box = df['train/box_loss']
train_cls = df['train/cls_loss']
train_dfl = df['train/dfl_loss']
precision = df['metrics/precision(B)']
recall = df['metrics/recall(B)']
mAP50 = df['metrics/mAP50(B)']
mAP50_95 = df['metrics/mAP50-95(B)']
val_box = df['val/box_loss']
val_cls = df['val/cls_loss']
val_dfl = df['val/dfl_loss']

OUT = 'archive/v3-skyguard-uav-20260723/skyguard-v2-uav-3'

# ===== 图1: 综合仪表板 =====
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('SkyGuard AI - YOLOv11 无人机检测训练仪表板', fontsize=16, fontweight='bold')

# 1. 训练损失
ax = axes[0, 0]
ax.plot(epoch, train_box, label='Box', linewidth=2)
ax.plot(epoch, train_cls, label='Cls', linewidth=2)
ax.plot(epoch, train_dfl, label='DFL', linewidth=2)
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.set_title('训练损失 (Training Loss)')
ax.legend(); ax.grid(True, alpha=0.3)

# 2. 验证损失
ax = axes[0, 1]
ax.plot(epoch, val_box, label='Box', linewidth=2)
ax.plot(epoch, val_cls, label='Cls', linewidth=2)
ax.plot(epoch, val_dfl, label='DFL', linewidth=2)
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.set_title('验证损失 (Val Loss)')
ax.legend(); ax.grid(True, alpha=0.3)

# 3. Precision & Recall
ax = axes[1, 0]
ax.plot(epoch, precision, '-o', label='Precision', linewidth=2, markersize=3)
ax.plot(epoch, recall, '-s', label='Recall', linewidth=2, markersize=3)
ax.set_xlabel('Epoch'); ax.set_ylabel('Score')
ax.set_title('Precision & Recall')
ax.set_ylim(0.7, 1.0)
ax.legend(); ax.grid(True, alpha=0.3)

# 4. mAP + 最佳点标注
ax = axes[1, 1]
ax.plot(epoch, mAP50, '-o', label='mAP50', linewidth=2, markersize=3)
ax.plot(epoch, mAP50_95, '-s', label='mAP50-95', linewidth=2, markersize=3)

best_idx = mAP50.idxmax()
ax.annotate(f'Best mAP50: {mAP50[best_idx]:.3f}\n(Epoch {epoch[best_idx]})',
            xy=(epoch[best_idx], mAP50[best_idx]),
            xytext=(epoch[best_idx]+3, mAP50[best_idx]-0.08),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=9, color='red', fontweight='bold')

best_idx95 = mAP50_95.idxmax()
ax.annotate(f'Best mAP50-95: {mAP50_95[best_idx95]:.3f}\n(Epoch {epoch[best_idx95]})',
            xy=(epoch[best_idx95], mAP50_95[best_idx95]),
            xytext=(epoch[best_idx95]+3, mAP50_95[best_idx95]+0.05),
            arrowprops=dict(arrowstyle='->', color='orange'),
            fontsize=9, color='orange', fontweight='bold')

ax.set_xlabel('Epoch'); ax.set_ylabel('Score')
ax.set_title('mAP 指标 (mAP Metrics)')
ax.set_ylim(0.3, 1.0)
ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUT}/chart_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()

# ===== 图2: 训练损失曲线 =====
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('训练损失曲线 (Training Loss)', fontsize=14, fontweight='bold')

for ax, data, title in zip(axes, [train_box, train_cls, train_dfl], ['Box Loss', 'Cls Loss', 'DFL Loss']):
    ax.plot(epoch, data, linewidth=2, color='steelblue')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUT}/chart_training_loss.png', dpi=150, bbox_inches='tight')
plt.close()

# ===== 图3: 验证指标 =====
fig, axes = plt.subplots(2, 1, figsize=(10, 8))
fig.suptitle('验证指标曲线 (Validation Metrics)', fontsize=14, fontweight='bold')

ax = axes[0]
ax.plot(epoch, precision, '-o', label='Precision', linewidth=2, markersize=3)
ax.plot(epoch, recall, '-s', label='Recall', linewidth=2, markersize=3)
ax.set_xlabel('Epoch'); ax.set_ylabel('Score')
ax.set_title('Precision & Recall')
ax.set_ylim(0.7, 1.0)
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(epoch, mAP50, '-o', label='mAP50', linewidth=2, markersize=3, color='darkgreen')
ax.plot(epoch, mAP50_95, '-s', label='mAP50-95', linewidth=2, markersize=3, color='darkorange')
ax.set_xlabel('Epoch'); ax.set_ylabel('Score')
ax.set_title('mAP50 & mAP50-95')
ax.set_ylim(0.3, 1.0)
ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUT}/chart_validation_metrics.png', dpi=150, bbox_inches='tight')
plt.close()

# ===== 图4: 训练 vs 验证损失 =====
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('训练 vs 验证损失对比', fontsize=14, fontweight='bold')

for ax, tr, va, title in zip(axes, [train_box, train_cls, train_dfl], [val_box, val_cls, val_dfl], ['Box Loss', 'Cls Loss', 'DFL Loss']):
    ax.plot(epoch, tr, '-', label='Train', linewidth=2)
    ax.plot(epoch, va, '--', label='Val', linewidth=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUT}/chart_train_vs_val.png', dpi=150, bbox_inches='tight')
plt.close()

print("✅ 4张图表已生成:")
print(f"  {OUT}/chart_dashboard.png")
print(f"  {OUT}/chart_training_loss.png")
print(f"  {OUT}/chart_validation_metrics.png")
print(f"  {OUT}/chart_train_vs_val.png")
