#!/usr/bin/env python3
"""
训练前置校验脚本
==================
批量检查：
1. 图片与标签一一对应
2. 空标签文件
3. 标签坐标越界（归一化坐标应在 [0,1] 范围内）
4. 损坏/无法读取的图片
5. 类别 ID 合法性
6. 图片尺寸异常（0x0 等）
"""

import os
import sys
from pathlib import Path
from collections import defaultdict
from PIL import Image

# ============================================================
# 配置
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "data" / "skyguard-v2"
REPORT_FILE = PROJECT_ROOT / "data" / "dataset_validation_report.txt"

# 如果 skyguard-v2 还没生成，检查 UAV.yolov11
if not DATASET_DIR.exists():
    DATASET_DIR = PROJECT_ROOT / "data" / "UAV.yolov11"
    print(f"⚠️  skyguard-v2 不存在，校验原始数据集: {DATASET_DIR}")

SPLITS = ["train", "val", "test"]
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def is_hidden(path: Path) -> bool:
    return path.name.startswith(".") or path.name.startswith("._")


def check_image(img_path: Path) -> bool:
    """检查图片是否可正常读取"""
    try:
        with Image.open(img_path) as img:
            img.verify()
        # 部分格式 verify 后需要重新打开
        with Image.open(img_path) as img:
            w, h = img.size
            if w == 0 or h == 0:
                return False
        return True
    except Exception:
        return False


def check_label(lbl_path: Path, num_classes: int) -> list:
    """
    检查标签文件合法性，返回异常列表。
    检查项：空文件、坐标越界、类别ID越界、格式错误
    """
    errors = []
    try:
        with open(lbl_path, "r") as f:
            lines = f.readlines()
    except Exception as e:
        return [f"无法读取文件: {e}"]

    if len(lines) == 0:
        errors.append("空标签文件（无标注）")
        return errors

    for line_no, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) != 5:
            errors.append(f"第{line_no}行格式错误（期望5列，实际{len(parts)}列）: {line}")
            continue

        try:
            class_id = int(parts[0])
            x, y, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        except ValueError:
            errors.append(f"第{line_no}行数字解析失败: {line}")
            continue

        # 类别ID检查
        if class_id < 0 or class_id >= num_classes:
            errors.append(f"第{line_no}行类别ID越界: {class_id} (期望 0-{num_classes-1})")

        # 坐标越界检查（YOLO归一化坐标应在 [0,1]）
        if x < 0 or x > 1 or y < 0 or y > 1:
            errors.append(f"第{line_no}行中心坐标越界: ({x}, {y})")
        if w <= 0 or w > 1 or h <= 0 or h > 1:
            errors.append(f"第{line_no}行宽高异常: ({w}, {h})")
        # 边界框超出图像范围
        if x - w/2 < -0.01 or x + w/2 > 1.01 or y - h/2 < -0.01 or y + h/2 > 1.01:
            errors.append(f"第{line_no}行边界框超出图像范围: x={x}, y={y}, w={w}, h={h}")

    return errors


def main():
    print("=" * 60)
    print("🔍 SkyGuard 数据集前置校验")
    print("=" * 60)
    print(f"📂 数据集路径: {DATASET_DIR}")

    # 读取类别数
    yaml_path = DATASET_DIR / "dataset.yaml"
    if not yaml_path.exists():
        # 尝试原始 data.yaml
        yaml_path = DATASET_DIR / "data.yaml"
    if yaml_path.exists():
        import yaml
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)
        num_classes = config.get("nc", 1)
    else:
        num_classes = 1

    all_issues = defaultdict(list)  # 按类型分组的问题
    total_images = 0
    total_labels = 0
    missing_labels = []
    missing_images = []
    corrupt_images = []
    empty_labels = []
    coord_errors = []
    total_annotations = 0

    for split_name in SPLITS:
        img_dir = DATASET_DIR / split_name / "images"
        lbl_dir = DATASET_DIR / split_name / "labels"

        if not img_dir.exists():
            print(f"⚠️  {split_name}/images 不存在，跳过")
            continue

        print(f"\n📁 检查 {split_name}/ ...")

        # 收集图片文件
        img_files = {}
        for f in img_dir.iterdir():
            if f.is_file() and not is_hidden(f) and f.suffix.lower() in VALID_EXTENSIONS:
                img_files[f.stem] = f

        # 收集标签文件
        lbl_files = {}
        if lbl_dir.exists():
            for f in lbl_dir.iterdir():
                if f.is_file() and f.suffix == ".txt" and not is_hidden(f):
                    lbl_files[f.stem] = f

        total_images += len(img_files)
        total_labels += len(lbl_files)

        # 检查图片-标签对应关系
        for stem in img_files:
            if stem not in lbl_files:
                missing_labels.append((split_name, img_files[stem].name))
        for stem in lbl_files:
            if stem not in img_files:
                missing_images.append((split_name, lbl_files[stem].name))

        # 检查图片完整性
        for stem, img_path in img_files.items():
            if not check_image(img_path):
                corrupt_images.append((split_name, img_path.name))

        # 检查标签内容
        for stem, lbl_path in lbl_files.items():
            if stem in img_files:  # 只检查有对应图片的标签
                errors = check_label(lbl_path, num_classes)
                if errors:
                    for err in errors:
                        if "空标签" in err:
                            empty_labels.append((split_name, lbl_path.name))
                        elif "坐标越界" in err or "宽高异常" in err or "边界框超出" in err:
                            coord_errors.append((split_name, lbl_path.name, err))
                        else:
                            all_issues["其他"].append((split_name, lbl_path.name, err))

        # 统计标注数
        for stem, lbl_path in lbl_files.items():
            try:
                with open(lbl_path, "r") as f:
                    lines = [l.strip() for l in f if l.strip()]
                    total_annotations += len(lines)
            except Exception:
                pass

    # 输出报告
    print(f"\n{'=' * 60}")
    print(f"📊 校验结果汇总")
    print(f"{'=' * 60}")
    print(f"  总图片数: {total_images}")
    print(f"  总标签数: {total_labels}")
    print(f"  总标注框: {total_annotations}")
    print(f"  类别数:   {num_classes}")

    issues_found = False

    if missing_labels:
        issues_found = True
        print(f"\n❌ 缺少标签文件的图片 ({len(missing_labels)} 个):")
        for split, name in missing_labels[:20]:
            print(f"   [{split}] {name}")
        if len(missing_labels) > 20:
            print(f"   ... 还有 {len(missing_labels) - 20} 个")

    if missing_images:
        issues_found = True
        print(f"\n❌ 缺少图片的标签 ({len(missing_images)} 个):")
        for split, name in missing_images[:20]:
            print(f"   [{split}] {name}")
        if len(missing_images) > 20:
            print(f"   ... 还有 {len(missing_images) - 20} 个")

    if corrupt_images:
        issues_found = True
        print(f"\n❌ 损坏/无法读取的图片 ({len(corrupt_images)} 个):")
        for split, name in corrupt_images:
            print(f"   [{split}] {name}")

    if empty_labels:
        issues_found = True
        print(f"\n⚠️  空标签文件（无标注）({len(empty_labels)} 个):")
        for split, name in empty_labels[:20]:
            print(f"   [{split}] {name}")
        if len(empty_labels) > 20:
            print(f"   ... 还有 {len(empty_labels) - 20} 个")

    if coord_errors:
        issues_found = True
        print(f"\n❌ 坐标异常 ({len(coord_errors)} 个):")
        for split, name, err in coord_errors[:20]:
            print(f"   [{split}] {name}: {err}")
        if len(coord_errors) > 20:
            print(f"   ... 还有 {len(coord_errors) - 20} 个")

    if not issues_found:
        print(f"\n✅ 数据集校验通过！所有文件正常。")

    # 写入详细报告
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("SkyGuard 数据集校验报告\n")
        f.write("=" * 60 + "\n")
        f.write(f"数据集路径: {DATASET_DIR}\n")
        f.write(f"总图片数: {total_images}\n")
        f.write(f"总标签数: {total_labels}\n")
        f.write(f"总标注框: {total_annotations}\n\n")

        if missing_labels:
            f.write(f"\n缺少标签的图片 ({len(missing_labels)}):\n")
            for split, name in missing_labels:
                f.write(f"  [{split}] {name}\n")
        if missing_images:
            f.write(f"\n缺少图片的标签 ({len(missing_images)}):\n")
            for split, name in missing_images:
                f.write(f"  [{split}] {name}\n")
        if corrupt_images:
            f.write(f"\n损坏图片 ({len(corrupt_images)}):\n")
            for split, name in corrupt_images:
                f.write(f"  [{split}] {name}\n")
        if empty_labels:
            f.write(f"\n空标签 ({len(empty_labels)}):\n")
            for split, name in empty_labels:
                f.write(f"  [{split}] {name}\n")
        if coord_errors:
            f.write(f"\n坐标异常 ({len(coord_errors)}):\n")
            for split, name, err in coord_errors:
                f.write(f"  [{split}] {name}: {err}\n")

    print(f"\n📄 详细报告已保存: {REPORT_FILE}")

    return 0 if not issues_found else 1


if __name__ == "__main__":
    sys.exit(main())