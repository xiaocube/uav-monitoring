#!/usr/bin/env python3
"""
SkyGuard 模型权重平均 (SWA - Stochastic Weight Averaging)

将多个同架构 YOLO 模型的权重进行加权平均，生成一个融合模型。
融合后的模型不增加任何推理成本，但通常比单个模型精度更高、泛化更好。

适用条件:
  - 模型架构必须完全相同（同一预训练权重初始化）
  - 类别必须一致
  - 建议来自同一数据集的不同训练 run

Usage:
    # 平均 v3 + v4（默认权重各 0.5）
    python scripts/ensemble_weights.py

    # 自定义权重
    python scripts/ensemble_weights.py --weights 0.6 0.4

    # 指定输出路径
    python scripts/ensemble_weights.py --output models/current/best_ensemble.pt
"""
from __future__ import annotations
import argparse
import sys
import copy
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from ultralytics import YOLO


def average_weights(model_paths: list[Path], weights: list[float], output: Path):
    """对多个 YOLO 模型的权重进行加权平均。

    Args:
        model_paths: 模型文件路径列表
        weights: 各模型权重（会归一化）
        output: 输出路径
    """
    assert len(model_paths) == len(weights), "模型数与权重数不匹配"
    assert len(model_paths) >= 2, "至少需要 2 个模型"

    # 归一化权重
    total = sum(weights)
    weights = [w / total for w in weights]
    print(f"融合权重: {[f'{w:.3f}' for w in weights]}")

    # 加载第一个模型作为基础
    print(f"\n加载基础模型: {model_paths[0]}")
    base_model = YOLO(str(model_paths[0]))
    base_sd = base_model.model.state_dict()

    # 验证所有模型架构一致
    print("验证模型架构一致性...")
    for i, mp in enumerate(model_paths[1:], 1):
        m = YOLO(str(mp))
        sd = m.model.state_dict()
        if list(sd.keys()) != list(base_sd.keys()):
            raise ValueError(
                f"模型 {mp.name} 架构与基础模型不一致\n"
                f"  基础模型键数: {len(base_sd)}\n"
                f"  当前模型键数: {len(sd)}"
            )
        # 验证形状一致
        for k in base_sd:
            if sd[k].shape != base_sd[k].shape:
                raise ValueError(f"参数 {k} 形状不匹配: {base_sd[k].shape} vs {sd[k].shape}")
        print(f"  ✅ {mp.name}: 架构一致 ({len(sd)} 参数)")

    # 加权平均
    # 关键: BN 层的 running_mean/running_var 不能简单平均(会导致归一化失效)
    # 策略: 只平均训练参数(conv weight/bias, bn weight/bias),
    #       BN 的 running_mean/running_var 使用性能更好的模型(最后一个)
    print("\n执行权重平均...")
    print("  策略: 卷积权重平均, BN 统计量用 v4 (性能最佳模型)")

    # 加载最后一个模型(作为 BN 统计量来源)
    last_model = YOLO(str(model_paths[-1]))
    last_sd = last_model.model.state_dict()

    averaged_sd = copy.deepcopy(base_sd)
    for k in averaged_sd:
        if "running_mean" in k or "running_var" in k or "num_batches_tracked" in k:
            # BN 统计量: 使用最后一个模型(性能最佳)
            averaged_sd[k] = last_sd[k].clone()
        else:
            # 训练参数: 加权平均
            averaged_sd[k] = base_sd[k].float() * weights[0]
            for i, mp in enumerate(model_paths[1:], 1):
                m = YOLO(str(mp))
                sd = m.model.state_dict()
                averaged_sd[k] += sd[k].float() * weights[i]
                del m
            averaged_sd[k] = averaged_sd[k].to(base_sd[k].dtype)

    # 加载到基础模型
    base_model.model.load_state_dict(averaged_sd)
    base_model.model.eval()

    # 保存: 直接用 torch.save 保存完整模型对象
    print(f"\n保存融合模型到: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    # 方式 1: 用 YOLO 的 save (保存为 .pt 格式,可被 YOLO() 加载)
    # 但需要确保保存的是修改后的权重
    # YOLO save 内部会调用 torch.save,保存 model.model 的 state_dict
    import pickle
    ckpt = {
        "model": base_model.model,
        "train_args": {"model": "yolo11s.pt", "data": "data/skyguard-v2/dataset.yaml"},
        "date": None,
        "version": "8.4.104",
    }
    torch.save(ckpt, str(output), pickle_protocol=pickle.HIGHEST_PROTOCOL)

    # 验证保存成功
    print("验证融合模型...")
    verify = YOLO(str(output))
    params = sum(p.numel() for p in verify.model.parameters())
    print(f"  参数量: {params/1e6:.2f}M")
    print(f"  类别: {verify.names}")

    # 快速推理验证
    import os
    test_imgs = []
    for root, dirs, files in os.walk("data/skyguard-v2/val/images"):
        for f in files:
            if f.endswith((".jpg", ".png")):
                test_imgs.append(os.path.join(root, f))
                if len(test_imgs) >= 5:
                    break
        if len(test_imgs) >= 5:
            break
    r = verify(test_imgs[0], conf=0.2, verbose=False)
    print(f"  推理测试: 检出 {len(r[0].boxes)} 个目标 (测试图: {os.path.basename(test_imgs[0])})")

    return base_model


def main():
    parser = argparse.ArgumentParser(description="YOLO 模型权重平均")
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "models/current/best_v3.pt",
            "models/current/best_v4.pt",
        ],
        help="模型文件路径",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=[0.5, 0.5],
        help="各模型权重",
    )
    parser.add_argument(
        "--output",
        default="models/current/best_ensemble.pt",
        help="输出路径",
    )
    args = parser.parse_args()

    model_paths = [ROOT / p for p in args.models]
    for p in model_paths:
        if not p.exists():
            print(f"❌ 模型不存在: {p}")
            sys.exit(1)

    print("=" * 60)
    print("SkyGuard 模型权重平均 (SWA)")
    print("=" * 60)
    for p, w in zip(model_paths, args.weights):
        print(f"  {p.name}: 权重 {w}")
    print()

    average_weights(model_paths, args.weights, Path(args.output))

    print("\n✅ 融合完成！")
    print(f"输出: {args.output}")


if __name__ == "__main__":
    main()
