#!/usr/bin/env python3
"""
SkyGuard WBF (Weighted Box Fusion) 模型集成推理

将多个 YOLO 模型的预测结果进行加权框融合 (WBF)，生成比单一模型精度更高的检测结果。
不同于 NMS（只保留最佳框），WBF 会对重叠框进行加权平均，充分利用所有模型的信息。

支持的模型:
  YOLOv11 系列 (当前生产):
    - v4: YOLO11s, skyguard-v2 数据集, mAP50=0.967
    - v3: YOLO11s, skyguard-v2 数据集, mAP50=0.966
    - v1-2: YOLO11n, skyguard-v1 数据集, mAP50=0.977
  YOLOv8 系列 (历史归档):
    - v8-v1: YOLOv8n, skyguard-v1 数据集, mAP50=0.964
    - v8-v2: YOLOv8n, DroneTrack 数据集 (bird+uav), mAP50=0.740

Usage:
    # 默认: v4 + v3 融合
    python scripts/ensemble_wbf.py

    # 三模型融合 (v4 + v3 + v1-2)
    python scripts/ensemble_wbf.py --models v4 v3 v1-2

    # 全模型融合 (YOLOv11 + YOLOv8)
    python scripts/ensemble_wbf.py --models v4 v3 v1-2 v8-v1 v8-v2

    # 评估模式
    python scripts/ensemble_wbf.py --eval

    # 单图推理
    python scripts/ensemble_wbf.py --image path/to/image.jpg

    # 摄像头实时检测
    python scripts/ensemble_wbf.py --webcam
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# 模型注册表
# ============================================================

MODEL_REGISTRY = {
    # YOLOv11 系列 (当前生产)
    "v4": {
        "path": ROOT / "models/current/best_v4.pt",
        "weight": 1.0,
        "class_map": {0: "uav"},
    },
    "v3": {
        "path": ROOT / "models/current/best_v3.pt",
        "weight": 1.0,
        "class_map": {0: "uav"},
    },
    "v1-2": {
        "path": ROOT / "archive/v1-dronetrack-20260720/训练结果/drone-v1-2/weights/best.pt",
        "weight": 0.8,
        "class_map": {0: "uav"},  # drone -> uav 统一
    },
    # YOLOv8 系列 (历史归档)
    "v8-v1": {
        "path": ROOT / "archive/v1-dronetrack-20260720/训练结果/drone-v1/weights/best.pt",
        "weight": 0.5,  # YOLOv8 初代模型
        "class_map": {0: "uav"},  # drone -> uav 统一
    },
    "v8-v2": {
        "path": ROOT / "archive/v1-dronetrack-20260720/训练结果/drone-v2-dronetrack/weights/best.pt",
        "weight": 0.4,  # YOLOv8, 2类模型, mAP较低
        "class_map": {0: "bird", 1: "uav"},  # bird 会被过滤,仅保留 uav
        "filter_classes": ["bird"],  # 过滤 bird 类
    },
}


# ============================================================
# WBF 核心算法
# ============================================================

def weighted_box_fusion(
    detections_list: list[list[dict]],
    weights: list[float] | None = None,
    iou_thr: float = 0.55,
    conf_type: str = "avg",
):
    """加权框融合 (Weighted Box Fusion)

    Args:
        detections_list: 每个模型的检测列表,每个检测是 {x,y,w,h,conf,class}
        weights: 各模型权重
        iou_thr: IoU 阈值,超过此值认为同一目标
        conf_type: 置信度融合方式 ('avg'/'max'/'box_and_model_avg')

    Returns:
        融合后的检测列表
    """
    if weights is None:
        weights = [1.0] * len(detections_list)

    # 归一化权重
    total_w = sum(weights)
    weights = [w / total_w for w in weights]

    # 收集所有检测框,标注来源模型
    all_boxes = []
    for model_idx, dets in enumerate(detections_list):
        for d in dets:
            all_boxes.append({
                "x1": d["x"],
                "y1": d["y"],
                "x2": d["x"] + d["w"],
                "y2": d["y"] + d["h"],
                "conf": d["conf"],
                "class": d.get("class", "uav"),
                "model_idx": model_idx,
                "weight": weights[model_idx],
            })

    if not all_boxes:
        return []

    # 按置信度降序排列
    all_boxes.sort(key=lambda b: b["conf"], reverse=True)

    # 聚类: 将重叠的框分组
    clusters = []
    used = [False] * len(all_boxes)

    for i, box in enumerate(all_boxes):
        if used[i]:
            continue
        cluster = [box]
        used[i] = True

        for j in range(i + 1, len(all_boxes)):
            if used[j]:
                continue
            if all_boxes[j]["class"] != box["class"]:
                continue
            iou = _box_iou(box, all_boxes[j])
            if iou >= iou_thr:
                cluster.append(all_boxes[j])
                used[j] = True

        clusters.append(cluster)

    # 对每个簇进行加权融合
    fused = []
    for cluster in clusters:
        if len(cluster) == 1:
            # 单框,直接使用
            b = cluster[0]
            fused.append({
                "x": b["x1"],
                "y": b["y1"],
                "w": b["x2"] - b["x1"],
                "h": b["y2"] - b["y1"],
                "conf": b["conf"],
                "class": b["class"],
                "model_count": 1,
            })
            continue

        # 多框融合: 加权平均坐标
        total_weight = sum(b["weight"] for b in cluster)
        fused_x1 = sum(b["x1"] * b["weight"] for b in cluster) / total_weight
        fused_y1 = sum(b["y1"] * b["weight"] for b in cluster) / total_weight
        fused_x2 = sum(b["x2"] * b["weight"] for b in cluster) / total_weight
        fused_y2 = sum(b["y2"] * b["weight"] for b in cluster) / total_weight

        # 置信度融合
        if conf_type == "avg":
            fused_conf = sum(b["conf"] * b["weight"] for b in cluster) / total_weight
        elif conf_type == "max":
            fused_conf = max(b["conf"] for b in cluster)
        else:  # box_and_model_avg
            # 融合模型数越多,置信度越高
            model_count = len(set(b["model_idx"] for b in cluster))
            base_conf = sum(b["conf"] * b["weight"] for b in cluster) / total_weight
            fused_conf = base_conf * (0.5 + 0.5 * model_count / len(weights))

        fused.append({
            "x": fused_x1,
            "y": fused_y1,
            "w": fused_x2 - fused_x1,
            "h": fused_y2 - fused_y1,
            "conf": fused_conf,
            "class": cluster[0]["class"],
            "model_count": len(set(b["model_idx"] for b in cluster)),
        })

    # 按置信度排序
    fused.sort(key=lambda d: d["conf"], reverse=True)
    return fused


def _box_iou(a: dict, b: dict) -> float:
    """计算两个框的 IoU"""
    x1 = max(a["x1"], b["x1"])
    y1 = max(a["y1"], b["y1"])
    x2 = min(a["x2"], b["x2"])
    y2 = min(a["y2"], b["y2"])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
    area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
    union = area_a + area_b - inter
    return inter / max(union, 1e-6)


# ============================================================
# 集成推理器
# ============================================================

class EnsembleDetector:
    """多模型 WBF 集成检测器"""

    def __init__(self, model_names: list[str], conf: float = 0.2, iou: float = 0.6, imgsz: int = 640):
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.models = {}
        self.weights = {}
        self.class_maps = {}
        self.filter_classes = {}

        print(f"加载 {len(model_names)} 个模型...")
        for name in model_names:
            if name not in MODEL_REGISTRY:
                raise ValueError(f"未知模型: {name}")
            info = MODEL_REGISTRY[name]
            if not info["path"].exists():
                raise FileNotFoundError(f"模型文件不存在: {info['path']}")
            print(f"  {name}: {info['path'].name}")
            self.models[name] = YOLO(str(info["path"]))
            self.weights[name] = info["weight"]
            self.class_maps[name] = info.get("class_map", {0: "uav"})
            self.filter_classes[name] = info.get("filter_classes", [])

        print(f"模型权重: {self.weights}")
        print()

    def _get_detections(self, name: str, model, image) -> list[dict]:
        """从单个模型获取检测结果，支持类别映射和过滤"""
        results = model(
            image,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device="mps",
            verbose=False,
        )
        class_map = self.class_maps[name]
        filter_set = set(self.filter_classes[name])
        dets = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            orig_class = class_map.get(cls_id, "uav")
            if orig_class in filter_set:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            dets.append({
                "x": x1,
                "y": y1,
                "w": x2 - x1,
                "h": y2 - y1,
                "conf": float(box.conf[0]),
                "class": "uav",
            })
        return dets

    def detect(self, image, wbf_iou: float = 0.55) -> list[dict]:
        """对单张图片进行集成检测"""
        all_detections = []
        for name, model in self.models.items():
            dets = self._get_detections(name, model, image)
            all_detections.append(dets)

        weights = [self.weights[name] for name in self.models]
        fused = weighted_box_fusion(all_detections, weights, iou_thr=wbf_iou)
        return fused

    def detect_with_timing(self, image) -> tuple[list[dict], dict]:
        """检测并返回各模型耗时"""
        timings = {}
        all_detections = []

        for name, model in self.models.items():
            t0 = time.time()
            dets = self._get_detections(name, model, image)
            elapsed = (time.time() - t0) * 1000
            timings[name] = elapsed
            all_detections.append(dets)

        t0 = time.time()
        weights = [self.weights[name] for name in self.models]
        fused = weighted_box_fusion(all_detections, weights)
        timings["wbf"] = (time.time() - t0) * 1000
        timings["total"] = sum(timings.values())

        return fused, timings


# ============================================================
# 评估函数
# ============================================================

def evaluate_ensemble(detector: EnsembleDetector, num_samples: int = 200):
    """在验证集上评估集成模型"""
    import random
    random.seed(42)

    val_imgs = []
    for root, dirs, files in os.walk("data/skyguard-v2/val/images"):
        for f in files:
            if f.endswith((".jpg", ".png")):
                val_imgs.append(os.path.join(root, f))

    sample = random.sample(val_imgs, min(num_samples, len(val_imgs)))
    print(f"评估样本: {len(sample)} 张")

    # warmup
    detector.detect(sample[0])

    t0 = time.time()
    total_dets = 0
    correct = 0
    multi_model_dets = 0

    for img_path in sample:
        fused = detector.detect(img_path)
        total_dets += len(fused)

        # 统计多模型一致检测
        for d in fused:
            if d.get("model_count", 1) >= 2:
                multi_model_dets += 1

        # 检查是否有 GT 且有检出
        label_path = img_path.replace("/images/", "/labels/").replace(".jpg", ".txt").replace(".png", ".txt")
        if os.path.exists(label_path):
            with open(label_path) as f:
                gt_count = len([l for l in f.readlines() if l.strip()])
            if gt_count > 0 and len(fused) > 0:
                correct += 1

    elapsed = time.time() - t0
    fps = len(sample) / elapsed

    print(f"\n=== 集成评估结果 ===")
    print(f"  总检出:     {total_dets}")
    print(f"  识别率:     {correct}/{len(sample)} ({correct/len(sample)*100:.1f}%)")
    print(f"  多模型一致: {multi_model_dets}/{total_dets} ({multi_model_dets/max(total_dets,1)*100:.1f}%)")
    print(f"  耗时:       {elapsed:.1f}s")
    print(f"  FPS:        {fps:.1f}")

    return correct, total_dets, fps


def compare_models(num_samples: int = 200):
    """对比单模型 vs 集成模型 (包含 YOLOv8 和 YOLOv11)"""
    import random
    random.seed(42)

    val_imgs = []
    for root, dirs, files in os.walk("data/skyguard-v2/val/images"):
        for f in files:
            if f.endswith((".jpg", ".png")):
                val_imgs.append(os.path.join(root, f))
    sample = random.sample(val_imgs, min(num_samples, len(val_imgs)))

    print(f"对比测试: {len(sample)} 张验证集图片\n")

    # 所有可用模型
    all_models = list(MODEL_REGISTRY.keys())

    # 单模型评估
    results = {}
    for name in all_models:
        info = MODEL_REGISTRY[name]
        if not info["path"].exists():
            print(f"  {name:10s}: 文件不存在, 跳过")
            continue
        model = YOLO(str(info["path"]))
        _ = model(sample[0], verbose=False)

        class_map = info.get("class_map", {0: "uav"})
        filter_set = set(info.get("filter_classes", []))

        t0 = time.time()
        dets = 0
        correct = 0
        for img_path in sample:
            r = model(img_path, conf=0.2, iou=0.6, imgsz=640, device="mps", verbose=False)[0]
            valid_boxes = 0
            for box in r.boxes:
                cls_id = int(box.cls[0])
                orig_class = class_map.get(cls_id, "uav")
                if orig_class not in filter_set:
                    valid_boxes += 1
            dets += valid_boxes
            label_path = img_path.replace("/images/", "/labels/").replace(".jpg", ".txt").replace(".png", ".txt")
            if os.path.exists(label_path):
                with open(label_path) as f:
                    gt_count = len([l for l in f.readlines() if l.strip()])
                if gt_count > 0 and valid_boxes > 0:
                    correct += 1
        elapsed = time.time() - t0
        results[name] = {"dets": dets, "correct": correct, "fps": len(sample) / elapsed}
        print(f"  {name:10s}: 检出={dets:4d}  识别率={correct}/{len(sample)}  FPS={len(sample)/elapsed:.1f}")

    # 多种集成组合
    ensembles = [
        ("YOLOv11-2模型", ["v4", "v3"]),
        ("YOLOv11-3模型", ["v4", "v3", "v1-2"]),
        ("YOLOv11+YOLOv8", ["v4", "v3", "v1-2", "v8-v1", "v8-v2"]),
    ]

    print()
    for ens_name, ens_models in ensembles:
        available = [m for m in ens_models if m in results or MODEL_REGISTRY[m]["path"].exists()]
        if len(available) < len(ens_models):
            missing = [m for m in ens_models if m not in available]
            print(f"  {ens_name}: 跳过 (缺少 {missing})")
            continue
        print(f"  {ens_name} ({available}):")
        detector = EnsembleDetector(available, conf=0.2, iou=0.6)
        correct_ens, dets_ens, fps_ens = evaluate_ensemble(detector, num_samples)
        results[ens_name] = {"dets": dets_ens, "correct": correct_ens, "fps": fps_ens}

    # 汇总
    print(f"\n=== 对比汇总 ===")
    print(f"{'模型':>15}  {'检出数':>6}  {'识别率':>8}  {'FPS':>6}")
    print("-" * 55)
    for name, r in results.items():
        print(f"{name:>15}  {r['dets']:>6}  {r['correct']}/{num_samples}  {r['fps']:>6.1f}")


# ============================================================
# 可视化
# ============================================================

def draw_detections(image, detections, color=(0, 255, 0)):
    """在图片上绘制检测框"""
    for d in detections:
        x, y, w, h = int(d["x"]), int(d["y"]), int(d["w"]), int(d["h"])
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

        conf = d["conf"]
        model_count = d.get("model_count", 1)
        label = f"uav {conf:.0%}"
        if model_count >= 2:
            label += f" [{model_count}M]"  # 多模型一致标记

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x, y - th - 10), (x + tw + 10, y), color, -1)
        cv2.putText(image, label, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return image


# ============================================================
# 主函数
# ============================================================

import os


def main():
    parser = argparse.ArgumentParser(description="SkyGuard WBF 模型集成推理")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["v4", "v3"],
        choices=list(MODEL_REGISTRY.keys()),
        help="使用的模型",
    )
    parser.add_argument("--conf", type=float, default=0.2, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.6, help="NMS IOU 阈值")
    parser.add_argument("--imgsz", type=int, default=640, help="推理尺寸")
    parser.add_argument("--wbf-iou", type=float, default=0.55, help="WBF 融合 IOU 阈值")
    parser.add_argument("--eval", action="store_true", help="评估模式")
    parser.add_argument("--compare", action="store_true", help="对比模式")
    parser.add_argument("--image", type=str, help="单图推理")
    parser.add_argument("--webcam", action="store_true", help="摄像头实时检测")
    parser.add_argument("--num-samples", type=int, default=200, help="评估样本数")
    args = parser.parse_args()

    if args.compare:
        compare_models(args.num_samples)
        return

    detector = EnsembleDetector(args.models, conf=args.conf, iou=args.iou, imgsz=args.imgsz)

    if args.eval:
        evaluate_ensemble(detector, args.num_samples)
        return

    if args.image:
        print(f"\n推理: {args.image}")
        image = cv2.imread(args.image)
        if image is None:
            print(f"❌ 无法读取图片: {args.image}")
            return

        fused, timings = detector.detect_with_timing(args.image)
        print(f"检出: {len(fused)} 个目标")
        for d in fused:
            mc = d.get("model_count", 1)
            print(f"  conf={d['conf']:.3f}  size={d['w']:.0f}x{d['h']:.0f}  models={mc}")
        print(f"耗时: {timings}")

        image = draw_detections(image, fused)
        output_path = ROOT / "models/current/ensemble_result.jpg"
        cv2.imwrite(str(output_path), image)
        print(f"结果已保存: {output_path}")
        return

    if args.webcam:
        print("\n启动摄像头实时检测")
        print("快捷键:")
        print("  q/ESC   退出")
        print("  +/-     调整置信度 (步长 0.05)")
        print("  [/]     调整 NMS IOU (步长 0.05)")
        print("  ,/.     调整 WBF IOU (步长 0.05)")
        print("  i       切换推理尺寸 (640/800/960)")
        print("  t       切换仅显示多模型一致目标")
        print("  s       截图保存到 data/samples/")
        print("  h       切换参数显示")
        print()

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ 无法打开摄像头")
            return

        # 可调参数
        conf = args.conf
        iou = args.iou
        wbf_iou = args.wbf_iou
        imgsz = args.imgsz
        show_multi_only = False
        show_help = True

        frame_count = 0
        last_fps_time = time.time()
        fps = 0
        save_dir = ROOT / "data/samples"
        save_dir.mkdir(parents=True, exist_ok=True)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            # 动态更新检测器参数
            detector.conf = conf
            detector.iou = iou
            detector.imgsz = imgsz

            fused, timings = detector.detect_with_timing(frame)

            # 过滤: 仅显示多模型一致目标
            if show_multi_only:
                fused = [d for d in fused if d.get("model_count", 1) >= 2]

            frame = draw_detections(frame, fused)

            # FPS 统计
            frame_count += 1
            now = time.time()
            if now - last_fps_time >= 1:
                fps = frame_count
                frame_count = 0
                last_fps_time = now

            # 顶部状态栏
            multi_count = sum(1 for d in fused if d.get("model_count", 1) >= 2)
            info = f"FPS:{fps} Dets:{len(fused)} Multi:{multi_count} {timings.get('total', 0):.0f}ms"
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
            cv2.putText(frame, info, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 参数面板
            if show_help:
                panel = [
                    f"conf={conf:.2f}  iou={iou:.2f}  wbf={wbf_iou:.2f}  imgsz={imgsz}",
                    f"models={list(detector.models.keys())}  multi_only={show_multi_only}",
                    "[+/-]conf  [/[ ]iou  [,.]wbf  [i]imgsz  [t]multi  [h]hide  [s]save  [q]quit",
                ]
                y = frame.shape[0] - 60
                cv2.rectangle(frame, (0, y - 5), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
                for i, line in enumerate(panel):
                    cv2.putText(frame, line, (10, y + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            cv2.imshow("SkyGuard Ensemble", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # q or ESC
                break
            elif key == ord("+") or key == ord("="):
                conf = min(0.9, conf + 0.05)
                print(f"conf -> {conf:.2f}")
            elif key == ord("-") or key == ord("_"):
                conf = max(0.05, conf - 0.05)
                print(f"conf -> {conf:.2f}")
            elif key == ord("["):
                iou = max(0.1, iou - 0.05)
                print(f"iou -> {iou:.2f}")
            elif key == ord("]"):
                iou = min(0.9, iou + 0.05)
                print(f"iou -> {iou:.2f}")
            elif key == ord(","):
                wbf_iou = max(0.1, wbf_iou - 0.05)
                print(f"wbf_iou -> {wbf_iou:.2f}")
            elif key == ord("."):
                wbf_iou = min(0.9, wbf_iou + 0.05)
                print(f"wbf_iou -> {wbf_iou:.2f}")
            elif key == ord("i"):
                imgsz = 800 if imgsz == 640 else (960 if imgsz == 800 else 640)
                print(f"imgsz -> {imgsz}")
            elif key == ord("t"):
                show_multi_only = not show_multi_only
                print(f"multi_only -> {show_multi_only}")
            elif key == ord("h"):
                show_help = not show_help
            elif key == ord("s"):
                fname = f"ensemble_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(str(save_dir / fname), frame)
                print(f"截图已保存: {save_dir / fname}")

        cap.release()
        cv2.destroyAllWindows()
        return

    # 默认: 评估模式
    print("\n请指定运行模式:")
    print("  --eval     验证集评估")
    print("  --compare  对比单模型 vs 集成")
    print("  --image    单图推理")
    print("  --webcam   摄像头实时检测")


if __name__ == "__main__":
    main()
