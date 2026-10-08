# SkyGuard 无人机反制系统 - 项目上下文

## 项目概述

基于机器视觉的低空安防解决方案，使用 YOLOv11 进行无人机目标检测。
- **项目路径**：`/Users/panyanming/Desktop/无人机反制`
- **开发环境**：macOS Apple M4, Python 3.9.6, MPS (Metal Performance Shaders)
- **虚拟环境**：`.venv/`（不可同步至 iCloud，需在新设备重建）
- **框架**：YOLOv11s (Ultralytics 8.4.104), PyTorch 2.8.0

## 目录结构

```
无人机反制/
├── models/current/              # 当前生产模型
│   ├── best_v4.pt              # v4 模型 (18MB, yolo11s, mAP50=0.967)
│   ├── best_v3.pt              # v3 模型 (72MB, 备份)
│   └── best_ensemble.pt        # 权重平均模型 (实验性, 有BN冲突)
├── runs/detect/train/uav-v4/    # v4 训练输出
│   └── weights/{best,last}.pt
├── data/skyguard-v2/           # 当前训练数据集 (8184 张)
│   ├── train/ (5821 张)
│   ├── val/   (1664 张)
│   └── dataset.yaml
├── data/UAV.yolov11/           # Roboflow 原始数据集 (备份)
├── archive/                    # 历史归档
│   ├── v1-dronetrack-20260720/ # v1 系列 (drone-v1, v1-2, v2-dronetrack)
│   ├── v2-dronetrack-20260723/
│   └── v3-skyguard-uav-20260723/
├── config/training/uav_v4.yaml # v4 训练配置
├── scripts/                    # 脚本目录
│   ├── train/train_uav_v4.sh   # 训练启动脚本
│   ├── webcam_demo.py          # 摄像头实时检测 (已引用 v4)
│   ├── ensemble_wbf.py         # WBF 多模型集成推理 ⭐
│   ├── ensemble_weights.py     # 权重平均 (SWA, 实验性)
│   ├── evaluate_uav.py         # 模型评估
│   ├── inference_benchmark.py  # 推理性能测试
│   └── deploy_inference.py     # 部署推理
├── web-demo/                   # Web 演示应用
│   ├── src/App.jsx             # 前端主组件
│   ├── server/app.py           # Flask 后端 (已引用 v4)
│   └── start.sh                # 一键启动
├── paper/                      # 论文和汇报材料
│   ├── 论文_基于机器视觉的空域无人机目标识别监控方法研究.docx
│   ├── 无人机智能监控系统_项目汇报.pptx
│   └── charts/                 # 数据可视化图表
└── src/skyguard/               # 项目核心模块
    ├── core/{config,logger}.py
    ├── vision/{detector,annotate}.py
    └── utils/device.py
```

## 模型版本历史

| 版本 | 架构 | 数据集 | mAP50 | mAP50-95 | Precision | Recall | 状态 |
|------|------|--------|-------|----------|-----------|--------|------|
| v1-2 | YOLO11n (3M) | skyguard-v1 | 0.9768 | 0.6267 | 0.9768 | 0.9414 | 归档 |
| v3 | YOLO11s (9.4M) | skyguard-v2 | 0.9663 | 0.6268 | 0.9535 | 0.9560 | 归档 |
| **v4** | YOLO11s (9.4M) | skyguard-v2 | **0.9670** | 0.6222 | 0.9525 | **0.9564** | **当前** |
| **WBF集成** | v4+v3+v1-2 | — | — | — | — | **100%*** | 最佳 |

*WBF 集成在 200 张样本测试中达到 100% 识别率（vs 单模型 99.5%）

### v4 训练配置（关键修正）

相比 v3 的改进（恢复 v1-2 的成功配方）：
```yaml
lr0: 0.001          # v3 的 0.01 → 0.001 (降10倍)
optimizer: AdamW     # v3 的 auto → AdamW
degrees: 5.0         # v3 的 0 → 5.0 (恢复角度增强)
shear: 2.0           # v3 的 0 → 2.0 (恢复剪切增强)
mixup: 0.1           # v3 的 0 → 0.1 (恢复混合增强)
batch: 16            # v3 的 8 → 16
iou: 0.6             # v3 的 0.7 → 0.6
epochs: 50           # v4 实际训练 50 epoch
```

## 模型集成方案

### WBF (Weighted Box Fusion) - 推荐方案

- **脚本**：`scripts/ensemble_wbf.py`
- **原理**：多模型分别推理，对预测框进行加权融合
- **模型组合**：v4 (权重 1.0) + v3 (权重 1.0) + v1-2 (权重 0.8)
- **效果**：200 张样本识别率 100%，多模型一致率 82.3%
- **FPS**：12.8（三模型）vs 21.4（单 v4）
- **优势**：支持不同架构模型，精度最高

### 权重平均 (SWA) - 实验性

- **脚本**：`scripts/ensemble_weights.py`
- **问题**：YOLO inference mode 下 tensor 版本冲突，BN 统计量平均后失效
- **状态**：未成功，保留备用

## 使用指南

### 启动训练

```bash
cd /Users/panyanming/Desktop/无人机反制
bash scripts/train/train_uav_v4.sh           # 新训练
bash scripts/train/train_uav_v4.sh --resume  # 续训
```

**安全停止**：在训练终端按 `Ctrl+C`（自动保存 last.pt）
**监控**：`tensorboard --logdir runs/detect/train/uav-v4`

### 模型评估

```bash
.venv/bin/python scripts/evaluate_uav.py                    # 验证集评估
.venv/bin/python scripts/ensemble_wbf.py --eval             # 集成评估
.venv/bin/python scripts/ensemble_wbf.py --compare          # 对比单模型 vs 集成
```

### 实时检测

```bash
# 单模型 (v4)
.venv/bin/python scripts/webcam_demo.py --conf 0.2 --imgsz 640

# 三模型 WBF 集成 (推荐, 100% 识别率)
.venv/bin/python scripts/ensemble_wbf.py --webcam
.venv/bin/python scripts/ensemble_wbf.py --webcam --models v4 v3 v1-2

# 快捷键: +/- 调conf, [/] 调iou, ,/. 调wbf, i 切imgsz, t 切多模型一致, s 截图, q 退出
```

### Web Demo

```bash
cd /Users/panyanming/Desktop/无人机反制/web-demo
./start.sh                    # 一键启动
# 前端: http://localhost:5173
# 后端: http://localhost:5001
```

### 模型导出

```bash
.venv/bin/python scripts/deploy_inference.py  # 支持 ONNX, TorchScript, CoreML
```

## 硬约束 (必须遵守)

1. **训练数据集必须本地存储**（不可用 iCloud，防止文件 eviction 中断训练）
2. **配置文件中的绝对路径**在设备间迁移时必须更新
3. **虚拟环境 `.venv/` 不可同步 iCloud**，需在新设备重建
4. **新数据集训练不可复用旧数据集**
5. **Mac MPS 不支持进程暂停/恢复**，只能优雅终止 + 续训
6. **训练停止必须按 `Ctrl+C`**（自动保存 last.pt），禁止直接关闭终端
7. **训练、监控、对话必须在独立终端窗口**

## 工程规范

### 路径命名
- **生产模型**：`models/current/best_vX.pt`
- **训练输出**：`runs/detect/train/uav-vX/`
- **历史归档**：`archive/vX-名称-日期/`
- **数据集**：`data/数据集名/`
- **训练配置**：`config/training/uav_vX.yaml`

### 推理参数推荐

| 模式 | imgsz | conf | iou | FPS | 适用场景 |
|------|-------|------|-----|-----|----------|
| 均衡 | 640 | 0.2 | 0.6 | 15-20 | 日常使用 |
| 精度 | 800 | 0.15 | 0.6 | 7-10 | 远距离小目标 |
| 速度 | 640 | 0.25 | 0.5 | 20-30 | 实时性要求高 |
| 集成 | 640 | 0.2 | 0.6 | 12-15 | 最高精度 |

## 经验教训

1. **iCloud 同步会导致训练中断**（自动文件 eviction），必须本地存储
2. **`runs/` 目录包含 200+ 冗余中间验证结果**，可定期清理
3. **Ultralytics 默认 `tensorboard=False`**，需显式启用以生成 scalar 数据
4. **权重平均 (SWA) 在 YOLO 中不可行**：BN 统计量平均后失效，inference mode 下 tensor 版本冲突
5. **WBF 是多模型集成的最佳方案**：支持异构模型，精度提升明显
6. **v1-2 虽然单独识别率最低，但在集成中互补性最强**（不同数据集训练）
7. **低学习率 (0.001) + 数据增强 (degrees/shear/mixup) 是关键**：v4 超越 v3 的核心原因
8. **训练轮数并非越多越好**：v4 在 epoch 45 达到最佳，之后开始震荡

## iCloud 同步文件

以下文件已同步至 iCloud Drive (Documents 文件夹)：
- `论文_基于机器视觉的空域无人机目标识别监控方法研究.docx`
- `无人机智能监控系统_项目汇报.pptx`

iPhone 访问路径：文件 App → iCloud Drive → 文稿

## 关键技术决策

1. **选择 YOLOv11s 而非 YOLOv8**：v11 在小目标检测上更优
2. **选择 WBF 而非 SWA**：SWA 在 YOLO 中有 BN 冲突，WBF 支持异构模型
3. **保留 v1-2 模型**：虽架构不同，但在 WBF 集成中提供互补价值
4. **MPS 而非 CPU**：M4 GPU 加速推理 3-5 倍
5. **Flask + WebSocket**：web-demo 后端实时帧传输

## 下一步计划

- [ ] 将 WBF 集成集成到 web-demo 后端
- [ ] 导出 v4 为 ONNX/CoreML 用于边缘部署
- [ ] 添加红外摄像头支持（夜间检测）
- [ ] 优化移动端推理性能
- [ ] 论文答辩 PPT 准备
