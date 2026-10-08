#!/usr/bin/env python3
"""SkyGuard 无人机检测 — 机载部署推理脚本

适配无人机机载视觉实时识别场景，支持：
- 摄像头实时输入
- 本地视频文件输入
- 画面框选目标，输出类别、像素坐标、置信度
- 轻量化设计，支持 ONNX/CoreML 推理

Usage:
    # 摄像头实时检测
    python scripts/deploy_inference.py --source camera --camera 0

    # 本地视频检测
    python scripts/deploy_inference.py --source video --input video.mp4

    # 使用 ONNX 模型（嵌入式部署）
    python scripts/deploy_inference.py --source camera --model best.onnx --device cpu

    # 输出检测结果到 JSON
    python scripts/deploy_inference.py --source video --input video.mp4 --output-json results.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODEL = ROOT / "models/current/best_v3.pt"
CLASS_NAMES = {0: "UAV"}


def detect_frame(model, frame: np.ndarray, conf: float, iou: float, imgsz: int, device: str) -> list:
    """单帧检测，返回检测结果列表"""
    results = model.predict(frame, conf=conf, iou=iou, imgsz=imgsz, device=device, verbose=False)
    detections = []
    for r in results:
        boxes = r.boxes
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
            cls_id = int(boxes.cls[i].cpu().numpy())
            score = float(boxes.conf[i].cpu().numpy())
            detections.append({
                "class_id": cls_id,
                "class_name": CLASS_NAMES.get(cls_id, f"class_{cls_id}"),
                "confidence": round(score, 3),
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "center_xy": [int((x1 + x2) / 2), int((y1 + y2) / 2)],
                "width": int(x2 - x1),
                "height": int(y2 - y1),
            })
    return detections


def draw_detections(frame: np.ndarray, detections: list) -> np.ndarray:
    """在画面上绘制检测结果"""
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        score = det["confidence"]
        cls_name = det["class_name"]

        # 框选目标
        color = (0, 255, 0) if score > 0.5 else (0, 200, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # 标签
        label = f"{cls_name} {score:.2f} [{det['center_xy'][0]},{det['center_xy'][1]}]"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 6), (x1 + label_size[0] + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return frame


def draw_hud(frame: np.ndarray, fps: float, n_det: int, model_name: str, imgsz: int, device: str) -> np.ndarray:
    """绘制 HUD 信息"""
    lines = [
        f"SkyGuard Deploy | {model_name} | imgsz={imgsz} | {device}",
        f"FPS: {fps:.1f} | Targets: {n_det}",
    ]
    for i, line in enumerate(lines):
        cv2.rectangle(frame, (8, 8 + i * 24), (8 + 400, 8 + (i + 1) * 24), (0, 0, 0), -1)
        cv2.putText(frame, line, (14, 24 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return frame


def main():
    parser = argparse.ArgumentParser(description="SkyGuard 机载部署推理脚本")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL), help="模型路径(.pt/.onnx)")
    parser.add_argument("--source", type=str, default="camera", choices=["camera", "video"], help="输入源")
    parser.add_argument("--camera", type=int, default=0, help="摄像头索引")
    parser.add_argument("--input", type=str, help="视频文件路径(source=video时必填)")
    parser.add_argument("--conf", type=float, default=0.2, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU 阈值")
    parser.add_argument("--imgsz", type=int, default=640, help="推理分辨率")
    parser.add_argument("--device", type=str, default="auto", help="推理设备: auto/cpu/mps/cuda")
    parser.add_argument("--output-json", type=str, help="输出检测结果 JSON 文件路径")
    parser.add_argument("--no-display", action="store_true", help="不显示画面（无头模式）")
    args = parser.parse_args()

    # 加载模型
    from ultralytics import YOLO
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[ERROR] 模型不存在: {model_path}")
        return 1

    print(f"[*] 加载模型: {model_path.name}")
    model = YOLO(str(model_path))

    # 打开输入源
    if args.source == "camera":
        cap = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            print(f"[ERROR] 无法打开摄像头 {args.camera}")
            return 1
        print(f"[*] 摄像头: {args.camera}")
    else:
        if not args.input:
            print("[ERROR] video 模式需要 --input 参数")
            return 1
        cap = cv2.VideoCapture(args.input)
        if not cap.isOpened():
            print(f"[ERROR] 无法打开视频: {args.input}")
            return 1
        print(f"[*] 视频: {args.input}")

    # 输出 JSON
    all_detections = [] if args.output_json else None
    fps_history = []
    frame_idx = 0

    print(f"[*] conf={args.conf} iou={args.iou} imgsz={args.imgsz} device={args.device}")
    print("[*] 按 q/ESC 退出")

    try:
        while True:
            t0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            detections = detect_frame(model, frame, args.conf, args.iou, args.imgsz, args.device)
            t1 = time.perf_counter()

            fps = 1.0 / max(t1 - t0, 1e-6)
            fps_history.append(fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = sum(fps_history) / len(fps_history)

            # 记录检测结果
            if all_detections is not None:
                all_detections.append({
                    "frame": frame_idx,
                    "timestamp": round(frame_idx / max(avg_fps, 1), 3),
                    "fps": round(fps, 1),
                    "detections": detections,
                })

            # 绘制
            if not args.no_display:
                frame = draw_detections(frame, detections)
                frame = draw_hud(frame, avg_fps, len(detections), model_path.name, args.imgsz, args.device)
                cv2.imshow("SkyGuard Deploy", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"  [Frame {frame_idx}] FPS: {avg_fps:.1f} | 检测到 {len(detections)} 个目标")

    except KeyboardInterrupt:
        print("\n[*] Ctrl+C 退出")
    finally:
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    print(f"[*] 共处理 {frame_idx} 帧 | 平均 FPS: {sum(fps_history)/max(len(fps_history),1):.1f}")

    if all_detections is not None and args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(all_detections, f, indent=2, ensure_ascii=False)
        print(f"[*] 检测结果已保存: {args.output_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
