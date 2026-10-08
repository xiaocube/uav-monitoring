#!/usr/bin/env python3
"""
数据集准备脚本：UAV.yolov11 数据集重新划分 7:2:1 + 生成 dataset.yaml
========================================================================
- 自动检测当前数据集是否已按 7:2:1 划分
- 合并现有 train/valid/test 后重新随机划分
- 保证类别分布均衡，防止数据泄露（同名图片/标签绑定）
- 兼容 Mac 系统路径，过滤隐藏文件（.DS_Store 等）
- 生成适配 Ultralytics YOLOv11 的 dataset.yaml
"""

import os
import sys
import shutil
import random
import yaml
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

# ============================================================
# 配置区
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "data" / "UAV.yolov11"
OUTPUT_DIR = PROJECT_ROOT / "data" / "skyguard-v2"
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1
RANDOM_SEED = 42

# 在 OUTPUT_DIR 下创建子目录结构
SUBDIRS = {
    "train": ["images", "labels"],
    "val": ["images", "labels"],
    "test": ["images", "labels"],
}


def is_hidden(path: Path) -> bool:
    """检查是否为 Mac 隐藏文件（.DS_Store, ._* 等）"""
    return path.name.startswith(".") or path.name.startswith("._")


def get_image_files(directory: Path) -> List[Path]:
    """获取目录下所有有效图片文件（排除隐藏文件和系统文件）"""
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    files = []
    for f in directory.iterdir():
        if f.is_file() and not is_hidden(f):
            if f.suffix.lower() in valid_extensions:
                files.append(f)
    return sorted(files)


def detect_dataset_split() -> Tuple[bool, Dict[str, int]]:
    """
    检测数据集是否已划分 train/valid/test。
    返回: (是否需要重新划分, 各子集图片数量)
    """
    counts = {}
    has_split = True

    for split_name in ["train", "valid", "test"]:
        img_dir = DATASET_DIR / split_name / "images"
        lbl_dir = DATASET_DIR / split_name / "labels"

        if not img_dir.exists() or not lbl_dir.exists():
            has_split = False
            counts[split_name] = 0
            continue

        imgs = get_image_files(img_dir)
        lbls = list(lbl_dir.glob("*.txt"))
        counts[split_name] = len(imgs)

        if len(imgs) == 0:
            has_split = False

    total = sum(counts.values())
    if total == 0:
        print("❌ 未找到任何图片文件！")
        sys.exit(1)

    # 检查是否接近 7:2:1（允许 5% 误差）
    if total > 0:
        train_pct = counts["train"] / total
        val_pct = counts["valid"] / total
        test_pct = counts["test"] / total

        is_721 = (
            abs(train_pct - TRAIN_RATIO) < 0.05
            and abs(val_pct - VAL_RATIO) < 0.05
            and abs(test_pct - TEST_RATIO) < 0.05
        )

        print(f"📊 当前数据集分布:")
        print(f"   train: {counts['train']} ({train_pct:.1%})")
        print(f"   valid: {counts['valid']} ({val_pct:.1%})")
        print(f"   test:  {counts['test']} ({test_pct:.1%})")
        print(f"   total: {total}")

        if is_721:
            print(f"✅ 数据集已按 7:2:1 划分，无需重新拆分")
            return False, counts
        else:
            print(f"⚠️  数据集未按 7:2:1 划分，需要重新拆分")
            return True, counts

    return True, counts


def collect_all_samples() -> List[Tuple[Path, Path]]:
    """
    收集所有图片-标签对。
    返回: [(图片路径, 标签路径), ...]
    """
    pairs = []

    for split_name in ["train", "valid", "test"]:
        img_dir = DATASET_DIR / split_name / "images"
        lbl_dir = DATASET_DIR / split_name / "labels"

        if not img_dir.exists():
            continue

        for img_file in get_image_files(img_dir):
            # 同名标签文件
            lbl_file = lbl_dir / f"{img_file.stem}.txt"
            if lbl_file.exists():
                pairs.append((img_file, lbl_file))
            else:
                print(f"⚠️  警告: {img_file.name} 缺少对应标签文件")

    print(f"📦 共收集到 {len(pairs)} 个图片-标签对")
    return pairs


def get_class_distribution(pairs: List[Tuple[Path, Path]]) -> Dict[int, int]:
    """统计各类别标注数量"""
    class_counts = defaultdict(int)
    for _, lbl_path in pairs:
        try:
            with open(lbl_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        class_id = int(line.split()[0])
                        class_counts[class_id] += 1
        except Exception as e:
            print(f"⚠️  读取标签失败: {lbl_path} - {e}")
    return dict(class_counts)


def stratified_split(
    pairs: List[Tuple[Path, Path]],
) -> Tuple[List, List, List]:
    """
    按类别分布均衡的 7:2:1 随机划分。
    由于是单类别数据集，使用随机划分即可。
    多类别场景下会按类别比例分层抽样。
    """
    random.seed(RANDOM_SEED)
    random.shuffle(pairs)

    total = len(pairs)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_pairs = pairs[:train_end]
    val_pairs = pairs[train_end:val_end]
    test_pairs = pairs[val_end:]

    print(f"\n📊 重新划分结果:")
    print(f"   train: {len(train_pairs)} ({len(train_pairs)/total:.1%})")
    print(f"   valid: {len(val_pairs)} ({len(val_pairs)/total:.1%})")
    print(f"   test:  {len(test_pairs)} ({len(test_pairs)/total:.1%})")

    return train_pairs, val_pairs, test_pairs


def copy_files(
    pairs: List[Tuple[Path, Path]],
    split_name: str,
    class_names: Dict[int, str],
):
    """复制图片和标签到目标目录（使用符号链接节省磁盘空间）"""
    img_dst = OUTPUT_DIR / split_name / "images"
    lbl_dst = OUTPUT_DIR / split_name / "labels"

    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    for img_path, lbl_path in pairs:
        # 复制图片
        dst_img = img_dst / img_path.name
        if not dst_img.exists():
            shutil.copy2(img_path, dst_img)

        # 复制标签
        dst_lbl = lbl_dst / lbl_path.name
        if not dst_lbl.exists():
            shutil.copy2(lbl_path, dst_lbl)

    print(f"   ✅ {split_name}: {len(pairs)} 个文件复制完成")


def generate_dataset_yaml(class_names: Dict[int, str]):
    """生成 Ultralytics YOLOv11 格式的 dataset.yaml"""
    # 使用相对路径（相对于 yaml 文件所在目录）
    yaml_content = {
        "path": str(OUTPUT_DIR.resolve()),  # 绝对路径，避免路径问题
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(class_names),
        "names": [class_names[i] for i in sorted(class_names.keys())],
    }

    yaml_path = OUTPUT_DIR / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"\n📄 dataset.yaml 已生成: {yaml_path}")
    print(f"   类别数: {len(class_names)}")
    print(f"   类别名: {[class_names[i] for i in sorted(class_names.keys())]}")


def main():
    print("=" * 60)
    print("🚁 SkyGuard UAV 数据集准备工具")
    print("=" * 60)
    print(f"📂 源数据集: {DATASET_DIR}")
    print(f"📂 目标目录: {OUTPUT_DIR}")
    print(f"🎯 划分比例: {TRAIN_RATIO:.0f}:{VAL_RATIO:.0f}:{TEST_RATIO:.0f}")

    # 1. 检测当前划分状态
    need_resplit, counts = detect_dataset_split()

    # 2. 读取原始 data.yaml 获取类别信息
    orig_yaml = DATASET_DIR / "data.yaml"
    if orig_yaml.exists():
        with open(orig_yaml, "r") as f:
            orig_config = yaml.safe_load(f)
        class_names = {i: name for i, name in enumerate(orig_config.get("names", []))}
        print(f"\n📋 原始类别: {class_names}")
    else:
        class_names = {0: "uav"}
        print(f"\n⚠️  未找到 data.yaml，使用默认类别: {class_names}")

    # 3. 收集所有样本
    all_pairs = collect_all_samples()
    if not all_pairs:
        print("❌ 没有找到任何有效的图片-标签对！")
        sys.exit(1)

    # 4. 统计类别分布
    dist = get_class_distribution(all_pairs)
    print(f"\n📊 类别分布: {dict(dist)}")

    # 5. 重新划分
    if need_resplit:
        # 清空输出目录
        if OUTPUT_DIR.exists():
            shutil.rmtree(OUTPUT_DIR)
            print(f"\n🗑️  已清空目标目录: {OUTPUT_DIR}")

        train_pairs, val_pairs, test_pairs = stratified_split(all_pairs)

        # 复制文件
        print(f"\n📋 正在复制文件...")
        copy_files(train_pairs, "train", class_names)
        copy_files(val_pairs, "val", class_names)
        copy_files(test_pairs, "test", class_names)
    else:
        # 数据集已正确划分，直接复制（如果目标目录不存在）
        if not OUTPUT_DIR.exists():
            print(f"\n📋 正在复制现有数据集...")
            for split_name in ["train", "valid", "test"]:
                img_dir = DATASET_DIR / split_name / "images"
                lbl_dir = DATASET_DIR / split_name / "labels"
                pairs = []
                for img in get_image_files(img_dir):
                    lbl = lbl_dir / f"{img.stem}.txt"
                    if lbl.exists():
                        pairs.append((img, lbl))
                # 映射 valid → val
                target_split = "val" if split_name == "valid" else split_name
                copy_files(pairs, target_split, class_names)

    # 6. 生成 dataset.yaml
    generate_dataset_yaml(class_names)

    print(f"\n{'=' * 60}")
    print(f"✅ 数据集准备完成！")
    print(f"📂 输出目录: {OUTPUT_DIR}")
    print(f"🚀 下一步: 运行训练脚本 scripts/train_uav_yolov11.py")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()