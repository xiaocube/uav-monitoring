# SkyGuard 项目量化报告

> 生成日期：2026-10-08 ｜ 数据来源：项目源码统计、训练日志（runs/detect/train/uav-v4/results.csv）、模型文件

## 1. 代码规模

| 模块 | 说明 | 代码量 |
|---|---|---|
| `src/skyguard/vision/` | 检测 / 跟踪 / 标注 / 视频流 | 1,648 行 |
| `src/skyguard/api/` | Web 后台 API（REST + WebSocket + 静态页） | 2,212 行 |
| `src/skyguard/edge/` | Jetson 边缘部署（TensorRT / 电源 / 监控） | 2,009 行 |
| `src/skyguard/ptz/` | PTZ 云台控制（ONVIF） | 1,888 行 |
| `src/skyguard/db/` | 事件数据库 | 962 行 |
| `src/skyguard/training/` | 训练 / 评估 / 导出 | 795 行 |
| `src/skyguard/data/` | 数据集处理 / 增强 / 清洗 | 662 行 |
| `src/skyguard/core/` | 配置 / 日志 / 异常 | 457 行 |
| `src/skyguard/inference/` | 推理后端抽象 | 451 行 |
| `src/skyguard/utils/` | 设备 / 环境工具 | 314 行 |
| **核心库合计** | | **≈ 11,400 行** |
| `scripts/` | 训练 / 评估 / 推理 / 数据集 / 部署脚本 | 6,016 行 |
| `tests/` | pytest 测试 | 3,848 行 |
| `web-demo/` | React 前端 + Flask 后端 | 2,160 行 |
| **Python 总代码** | | **≈ 42,700 行** |

## 2. 测试

- 测试文件：**21 个**（覆盖 vision / api / db / edge / ptz / training / data / config / inference）
- 测试用例：**279 个**
- CI：GitHub Actions 4 个流水线（Lint&Test × Python 3.10-3.12 矩阵、安全扫描、Docker 构建、Release），代码覆盖率上报 Codecov

## 3. 模型成果

| 版本 | 架构 | 参数 | mAP50 | mAP50-95 | Precision | Recall | 状态 |
|---|---|---|---|---|---|---|---|
| v1-2 | YOLO11n | 3M | 0.977 | 0.627 | 0.977 | 0.941 | 归档 |
| v3 | YOLO11s | 9.4M | 0.966 | 0.627 | 0.954 | 0.956 | 归档 |
| **v4（当前）** | YOLO11s | 9.4M | **0.965** | **0.622** | **0.950** | **0.956** | 生产 |
| **WBF 集成** | v4+v3+v1-2 | — | — | — | — | **100% 识别率** | 最佳 |

- v4 训练配置：YOLO11s + 50 epoch + AdamW（lr0=0.001）+ 数据增强（degrees=5 / shear=2 / mixup=0.1）
- WBF 集成在 200 张样本测试中识别率 100%（单模型 99.5%），多模型一致率 82.3%
- 推理性能：单模型 21.4 FPS ｜ WBF 三模型 12.8 FPS（MPS）

## 4. 数据集

| 数据集 | 规模 | 划分（train / val / test） | 类别 |
|---|---|---|---|
| skyguard-v2 | **8,318 张** | 5,821 / 1,664 / 833 | 1（uav） |
| 原始备份（UAV.yolov11 / DroneTrack-extra） | 归档保留 | — | — |

## 5. 研发路线图进度

| Sprint | 主题 | 状态 |
|---|---|---|
| 1-4 | 环境骨架 / YOLO 检测 / 数据集制作 / 自有模型训练 | ✅ 完成 |
| 5 | 模型优化（TensorRT / ONNX / 量化） | ✅ 完成 |
| 6-7 | 实时视频流检测 / ByteTrack 目标跟踪 | ✅ 完成 |
| 8 | PTZ 自动控制（ONVIF 客户端 + 跟踪控制器已实现） | 🚧 进行中 |
| 9 | Web 后台（API + WebSocket + 前端已实现） | 🚧 进行中 |
| 10-12 | 数据库 / Jetson 边缘部署 / 工业产品化 | 🚧 待推进 |

## 6. 工程设施

- **依赖**：requirements.txt（44）/ requirements-dev.txt（26）/ requirements.jetson.txt（51）
- **容器**：Dockerfile（CPU/GPU）+ Dockerfile.jetson（NVIDIA Jetson）+ docker-compose.yml
- **环境**：Apple Silicon（MPS 开发）｜ NVIDIA Jetson（生产）
- **代码质量**：black + isort + flake8 + mypy + bandit + pip-audit + pre-commit

## 7. 仓库发布范围

| 包含 ✅ | 不包含 ❌（数据/产物/个人资料，见 .gitignore） |
|---|---|
| 源码、脚本、测试、配置 | 训练数据集（data/，8318 张，550MB） |
| Web 演示（前端 + 后端 + 示例图） | 历史归档（archive/，3.7GB） |
| 论文 Markdown + 图表 + 汇报材料 | 模型权重（models/current/，152MB） |
| CI、Docker、工程配置 | 训练中间产物（runs/，768MB） |
| 环境模板（.env.example） | 个人面试/分析资料（分析产出/） |

---

*本报告由仓库整理时自动生成，源码为权威来源。*
