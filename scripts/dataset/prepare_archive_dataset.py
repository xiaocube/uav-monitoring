"""
将 archive/drone_dataset_yolo 数据集转换为 SkyGuard 训练格式。

数据集结构转换:
  archive/drone_dataset_yolo/dataset_txt/  (图片+标注混在一起)
  → data/skyguard-v1/processed/
      images/{train,val,test}/
      labels/{train,val,test}/
      data.yaml
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for p in [str(SRC), str(ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)


def prepare_dataset(
    source_dir: Path,
    target_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.15,
    test_ratio: float = 0.05,
    seed: int = 42,
) -> dict:
    """将数据集转换为 YOLO 训练格式。

    Args:
        source_dir: 原始数据目录（包含 .jpg 和 .txt）
        target_dir: 目标目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子

    Returns:
        统计信息字典
    """
    import json

    random.seed(seed)

    # 确保比例和为 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例和必须为 1"

    # 收集所有图片文件
    image_files = sorted([f for f in source_dir.glob("*.jpg")])
    print(f"发现 {len(image_files)} 张图片")

    # 随机打乱
    random.shuffle(image_files)

    # 划分数据集
    n = len(image_files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_files = image_files[:n_train]
    val_files = image_files[n_train:n_train + n_val]
    test_files = image_files[n_train + n_val:]

    print(f"划分: 训练 {len(train_files)} | 验证 {len(val_files)} | 测试 {len(test_files)}")

    # 创建目录结构
    for split in ["train", "val", "test"]:
        (target_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (target_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 复制文件
    def copy_files(files, split):
        for img_path in files:
            label_path = img_path.with_suffix(".txt")

            # 复制图片
            shutil.copy2(img_path, target_dir / "images" / split / img_path.name)

            # 复制标注（如果存在）
            if label_path.exists():
                shutil.copy2(label_path, target_dir / "labels" / split / label_path.name)

    copy_files(train_files, "train")
    copy_files(val_files, "val")
    copy_files(test_files, "test")

    # 生成 data.yaml
    data_yaml = {
        "path": str(target_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 1,
        "names": {0: "drone"},
    }

    yaml_path = target_dir / "data.yaml"
    with yaml_path.open("w", encoding="utf-8") as f:
        json.dump(data_yaml, f, indent=2, ensure_ascii=False)

    print(f"data.yaml 已生成: {yaml_path}")

    # 返回统计信息
    stats = {
        "total": n,
        "train": len(train_files),
        "val": len(val_files),
        "test": len(test_files),
        "classes": ["drone"],
    }

    # 保存统计信息
    stats_path = target_dir / "dataset_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    return stats


def main():
    parser = argparse.ArgumentParser(description="准备 archive 数据集用于训练")
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "archive/drone_dataset_yolo/dataset_txt",
        help="原始数据集目录",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=ROOT / "data/skyguard-v1/processed",
        help="目标目录",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.source.exists():
        print(f"错误: 源目录不存在: {args.source}")
        return 1

    stats = prepare_dataset(
        source_dir=args.source,
        target_dir=args.target,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    print("\n数据集准备完成!")
    print(f"  总计: {stats['total']} 张")
    print(f"  训练: {stats['train']} 张")
    print(f"  验证: {stats['val']} 张")
    print(f"  测试: {stats['test']} 张")
    print(f"  类别: {stats['classes']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())