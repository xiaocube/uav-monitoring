#!/usr/bin/env python3
"""
SkyGuard 摄像头识别成功率测试 Demo

用 Mac 摄像头实时检测，统计识别成功率，生成测试报告。

功能：
  - 实时检测 + 画面标注
  - 统计总帧数、检测帧数、各类别检出率
  - 实时显示置信度分布
  - 退出后生成测试报告

Usage:
    # 基础测试（默认配置）
    python scripts/accuracy_test_demo.py

    # 指定模型 + 高分辨率 + 低阈值
    python scripts/accuracy_test_demo.py \
        --model models/trained/drone-v2-dronetrack/weights/best_final.pt \
        --conf 0.15 --imgsz 1280

    # 录制测试视频 + 生成报告
    python scripts/accuracy_test_demo.py --record --report

Controls:
    q / ESC  - 退出并生成报告
    r        - 开始/停止录制
    +/-      - 调整置信度阈值
    space    - 暂停/继续
"""
from __future__ import annotations

import argparse
import sys
import time
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent

import cv2
import numpy as np
from ultralytics import YOLO


def draw_hud(frame, stats, conf, fps, paused, recording):
    """在画面上绘制 HUD 信息"""
    h, w = frame.shape[:2]
    # 半透明背景
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (420, 200), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    lines = [
        f"FPS: {fps:.1f}  |  Conf: {conf:.2f}  |  imgsz: {stats['imgsz']}",
        f"Total Frames: {stats['total']}",
        f"Detected Frames: {stats['detected']} ({100*stats['detected']/max(1,stats['total']):.1f}%)",
        f"--- Class Stats ---",
    ]

    for cls_name, data in stats['classes'].items():
        pct = 100 * data['count'] / max(1, stats['total'])
        avg_conf = np.mean(data['confs']) if data['confs'] else 0
        lines.append(f"  {cls_name}: {data['count']} ({pct:.1f}%) avg={avg_conf:.2f}")

    lines.append(f"--- Avg Confidence: {stats['avg_conf']:.2f} ---")

    if paused:
        lines.append("[ PAUSED ]")
    if recording:
        lines.append("[ REC ]")

    for i, line in enumerate(lines):
        color = (0, 0, 255) if (recording and i == 0) else (255, 255, 255)
        cv2.putText(frame, line, (16, 28 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    # 右下角提示
    hint = "q:quit  r:rec  +/-:conf  space:pause"
    cv2.putText(frame, hint, (w - 320, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)

    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description="SkyGuard 摄像头识别成功率测试")
    parser.add_argument("--model", type=str,
                        default="models/trained/drone-v2-dronetrack/weights/best_final.pt",
                        help="模型路径")
    parser.add_argument("--camera", type=int, default=0, help="摄像头索引")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.50, help="NMS IoU 阈值")
    parser.add_argument("--imgsz", type=int, default=1280, help="推理分辨率")
    parser.add_argument("--device", type=str, default="mps", help="推理设备: mps/cpu/cuda")
    parser.add_argument("--width", type=int, default=1280, help="采集宽")
    parser.add_argument("--height", type=int, default=720, help="采集高")
    parser.add_argument("--record", action="store_true", help="录制测试视频")
    parser.add_argument("--report", action="store_true", help="退出后生成 JSON 报告")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[ERROR] 模型不存在: {model_path}")
        return 1

    # 加载模型
    print(f"[*] 加载模型: {model_path.name}")
    model = YOLO(str(model_path))
    class_names = model.names
    print(f"[*] 类别: {class_names}")
    print(f"[*] 设备: {args.device}  imgsz: {args.imgsz}  conf: {args.conf}")

    # 打开摄像头
    print(f"[*] 打开摄像头 {args.camera} ...")
    cap = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("[ERROR] 无法打开摄像头")
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[*] 分辨率: {actual_w}x{actual_h}")

    # 统计数据
    stats = {
        'total': 0,
        'detected': 0,
        'classes': defaultdict(lambda: {'count': 0, 'confs': []}),
        'avg_conf': 0.0,
        'imgsz': args.imgsz,
        'all_confs': [],
    }

    # 录制
    writer = None
    recording = False
    if args.record:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        rec_path = ROOT / "data" / "samples" / f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        rec_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(rec_path), fourcc, 25, (actual_w, actual_h))
        recording = True
        print(f"[*] 录制到: {rec_path}")

    print()
    print("=" * 60)
    print("  SkyGuard 识别成功率测试")
    print("  q/ESC: 退出并生成报告 | r: 录制 | +/-: 置信度")
    print("  space: 暂停/继续")
    print("=" * 60)
    print()

    cur_conf = args.conf
    paused = False
    fps_history = []
    start_time = time.time()

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] 读取帧失败")
                break

            # 推理
            t0 = time.perf_counter()
            results = model.predict(
                frame, imgsz=args.imgsz, conf=cur_conf, iou=args.iou,
                device=args.device, verbose=False
            )
            dt = (time.perf_counter() - t0) * 1000
            fps_history.append(1000 / dt if dt > 0 else 0)
            if len(fps_history) > 60:
                fps_history.pop(0)

            # 统计
            stats['total'] += 1
            result = results[0]
            has_det = len(result.boxes) > 0

            if has_det:
                stats['detected'] += 1
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf_val = float(box.conf[0])
                    cls_name = class_names.get(cls_id, str(cls_id))
                    stats['classes'][cls_name]['count'] += 1
                    stats['classes'][cls_name]['confs'].append(conf_val)
                    stats['all_confs'].append(conf_val)

            # 画框
            annotated = result.plot()

            # 计算 FPS
            cur_fps = np.mean(fps_history) if fps_history else 0

            # 更新统计
            stats['avg_conf'] = np.mean(stats['all_confs']) if stats['all_confs'] else 0

            # HUD
            annotated = draw_hud(annotated, stats, cur_conf, cur_fps, paused, recording)

            if recording and writer:
                writer.write(annotated)

            cv2.imshow("SkyGuard Accuracy Test", annotated)

        # 按键
        key = cv2.waitKey(1 if not paused else 50) & 0xFF
        if key in (ord('q'), 27):  # q or ESC
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('r'):
            recording = not recording
            print(f"[*] 录制: {'ON' if recording else 'OFF'}")
        elif key in (ord('+'), ord('=')):
            cur_conf = min(0.95, cur_conf + 0.05)
            print(f"[*] 置信度: {cur_conf:.2f}")
        elif key == ord('-'):
            cur_conf = max(0.05, cur_conf - 0.05)
            print(f"[*] 置信度: {cur_conf:.2f}")

    # 清理
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    # 生成报告
    elapsed = time.time() - start_time
    print()
    print("=" * 60)
    print("  测试报告")
    print("=" * 60)
    print(f"  测试时长: {elapsed:.1f} 秒")
    print(f"  总帧数: {stats['total']}")
    print(f"  检测到目标的帧: {stats['detected']} ({100*stats['detected']/max(1,stats['total']):.1f}%)")
    print(f"  平均 FPS: {np.mean(fps_history):.1f}" if fps_history else "  FPS: N/A")
    print(f"  平均置信度: {stats['avg_conf']:.3f}" if stats['all_confs'] else "  平均置信度: N/A")
    print()
    print("  类别统计:")
    for cls_name, data in sorted(stats['classes'].items()):
        pct = 100 * data['count'] / max(1, stats['total'])
        avg_c = np.mean(data['confs']) if data['confs'] else 0
        max_c = max(data['confs']) if data['confs'] else 0
        min_c = min(data['confs']) if data['confs'] else 0
        print(f"    {cls_name}: 检出 {data['count']} 次 ({pct:.1f}%)  "
              f"置信度 avg={avg_c:.3f} min={min_c:.3f} max={max_c:.3f}")
    print("=" * 60)

    # JSON 报告
    if args.report:
        report = {
            'timestamp': datetime.now().isoformat(),
            'model': str(model_path),
            'device': args.device,
            'imgsz': args.imgsz,
            'conf_threshold': cur_conf,
            'duration_sec': round(elapsed, 1),
            'total_frames': stats['total'],
            'detected_frames': stats['detected'],
            'detection_rate': round(100 * stats['detected'] / max(1, stats['total']), 2),
            'avg_fps': round(np.mean(fps_history), 1) if fps_history else 0,
            'avg_confidence': round(stats['avg_conf'], 4) if stats['all_confs'] else 0,
            'classes': {
                name: {
                    'count': d['count'],
                    'rate_pct': round(100 * d['count'] / max(1, stats['total']), 2),
                    'avg_conf': round(float(np.mean(d['confs'])), 4) if d['confs'] else 0,
                    'min_conf': round(float(min(d['confs'])), 4) if d['confs'] else 0,
                    'max_conf': round(float(max(d['confs'])), 4) if d['confs'] else 0,
                }
                for name, d in sorted(stats['classes'].items())
            }
        }
        report_path = ROOT / "data" / f"accuracy_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n[*] 报告已保存: {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
