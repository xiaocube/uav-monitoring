"""
SWA (Stochastic Weight Averaging) 模型合并脚本.

将多个检查点的权重进行平均,提升模型泛化能力.
用法:
  python scripts/swa_merge.py \
      --models models/trained/drone-v2-dronetrack/weights/best.pt \
                models/trained/drone-v2-dronetrack/weights/last.pt \
                models/trained/drone-v2-dronetrack/weights/epoch10.pt \
      --output models/trained/drone-v2-dronetrack/weights/swa_merged.pt
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def merge_weights(model_paths: list[Path], output: Path) -> None:
    """对多个 YOLO 模型的权重进行 SWA 平均."""
    print(f"合并 {len(model_paths)} 个检查点:")
    for p in model_paths:
        print(f"  - {p}")

    # 加载第一个模型作为基准（ultralytics 的模型存储在 ema 键中）
    base_ckpt = torch.load(str(model_paths[0]), map_location="cpu", weights_only=False)
    base_model = base_ckpt.get("model") or base_ckpt.get("ema")
    if base_model is None:
        raise ValueError("无法从检查点中加载模型 (model/ema 均为 None)")

    # 获取 state_dict
    sd = base_model.state_dict() if hasattr(base_model, "state_dict") else dict(base_model.named_parameters())

    # 累加所有模型的权重
    avg_sd = {k: v.clone().float() for k, v in sd.items()}
    n_models = 1

    for path in model_paths[1:]:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        model = ckpt.get("model") or ckpt.get("ema")
        if model is None:
            print(f"  警告: 跳过 {path} (模型为 None)")
            continue
        other_sd = model.state_dict() if hasattr(model, "state_dict") else dict(model.named_parameters())

        for k in avg_sd:
            if k in other_sd:
                avg_sd[k] += other_sd[k].float()
                # 跟踪每个 key 被累加的次数
        n_models += 1

    # 平均
    for k in avg_sd:
        avg_sd[k] /= n_models

    # 将平均后的权重写回基准模型
    base_model.load_state_dict(avg_sd)

    # 保存（同时写入 model 和 ema 键确保兼容性）
    output.parent.mkdir(parents=True, exist_ok=True)
    base_ckpt["ema"] = base_model
    base_ckpt["model"] = base_model
    torch.save(base_ckpt, str(output))
    print(f"\n合并完成,保存到: {output}")
    print(f"平均了 {n_models} 个检查点的权重")


def main() -> int:
    parser = argparse.ArgumentParser(description="SWA 权重平均合并")
    parser.add_argument(
        "--models", nargs="+", required=True, type=Path,
        help="要合并的模型路径列表(2个或以上)",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="输出合并后的模型路径",
    )
    args = parser.parse_args()

    if len(args.models) < 2:
        print("错误: 至少需要 2 个模型才能进行 SWA 合并", file=sys.stderr)
        return 1

    for p in args.models:
        if not p.exists():
            print(f"错误: 模型文件不存在: {p}", file=sys.stderr)
            return 1

    merge_weights(args.models, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
