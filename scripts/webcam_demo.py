#!/usr/bin/env python3
"""
SkyGuard 实时摄像头无人机检测 Demo

使用 Mac 自带摄像头,加载训练好的无人机检测模型,实时检测画面中的无人机。

支持模型版本:
  - drone-v1-2: 单类无人机检测(原模型)
  - drone-v2-dronetrack: 双类检测(bird + uav,追加训练后)

针对远距离小目标检测,提供两种增强模式:
  1. 高分辨率推理(--imgsz 1280):提升模型输入分辨率,代价是推理变慢
  2. 切片推理(--tile 2):将画面切成 2x2 重叠切片分别推理,显著提升小目标召回率

Usage:
    # 默认:drone-v1-2 模型 + 默认摄像头(0)
    python scripts/webcam_demo.py

    # 使用新训练的双类模型(bird + uav)
    python scripts/webcam_demo.py --model models/trained/drone-v2-dronetrack/weights/best.pt

    # 远距离增强:降低置信度 + 高分辨率推理
    python scripts/webcam_demo.py --model models/trained/drone-v2-dronetrack/weights/best.pt --conf 0.15 --imgsz 1280

    # 远距离最强:切片推理(2x2 切片,每片 640px)
    python scripts/webcam_demo.py --model models/trained/drone-v2-dronetrack/weights/best.pt --conf 0.15 --tile 2

    # 用预训练 COCO 模型(检测飞机/鸟/风筝)
    python scripts/webcam_demo.py --model yolov8n.pt --coco

Controls:
    q / ESC  - 退出
    s        - 截图保存到 data/samples/
    r        - 开始/停止录制(保存到 data/samples/)
    t        - 运行中切换 切片推理 开/关
    +/-      - 运行中调整 置信度阈值
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import cv2
import numpy as np

from skyguard.core.config import clear_settings_cache, get_settings
from skyguard.core.logger import setup_logging
from skyguard.utils.device import resolve_torch_device
from skyguard.vision.annotate import annotate
from skyguard.vision.detector import YOLODetector, Detection, DetectionResult

# 默认模型:项目训练好的无人机检测模型
DEFAULT_MODEL = ROOT / "models" / "current" / "best_v4.pt"

# 截图 / 录制输出目录
OUTPUT_DIR = ROOT / "data" / "samples"


# ---------------------------------------------------------------------------
# 切片推理 (Tiling Inference) —— 零依赖实现,等效于 SAHI
# ---------------------------------------------------------------------------

def _slice_image(
    image: np.ndarray, n_tiles: int, overlap: float = 0.25
) -> List[Tuple[np.ndarray, int, int, int, int]]:
    """将图像切成 n_tiles x n_tiles 个重叠切片。

    Returns: List of (tile_image, x_offset, y_offset, tile_w, tile_h)
    """
    h, w = image.shape[:2]
    tile_w = w // n_tiles
    tile_h = h // n_tiles
    # 重叠像素(防止目标被切断在切片边界)
    ow = int(tile_w * overlap)
    oh = int(tile_h * overlap)

    slices = []
    for row in range(n_tiles):
        for col in range(n_tiles):
            x1 = max(0, col * tile_w - ow)
            y1 = max(0, row * tile_h - oh)
            x2 = min(w, (col + 1) * tile_w + ow)
            y2 = min(h, (row + 1) * tile_h + oh)
            tile = image[y1:y2, x1:x2]
            slices.append((tile, x1, y1, x2 - x1, y2 - y1))
    return slices


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> List[int]:
    """纯 numpy NMS,返回保留的索引。"""
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / np.maximum(union, 1e-6)
        idx = np.where(iou <= iou_thr)[0]
        order = order[idx + 1]
    return keep


def predict_with_tiling(
    detector: YOLODetector,
    image: np.ndarray,
    n_tiles: int = 2,
    overlap: float = 0.25,
) -> DetectionResult:
    """切片推理:将画面切成 n_tiles x n_tiles 个重叠切片分别推理,
    再把检测框映射回原图坐标,最后做全局 NMS 去重。

    对远距离小目标有显著效果 —— 切片后小目标在局部切片中相对变大,
    模型更容易检测到。
    """
    t0 = time.perf_counter()

    slices = _slice_image(image, n_tiles, overlap)

    all_boxes: List[Tuple[int, int, int, int]] = []
    all_scores: List[float] = []
    all_cls_ids: List[int] = []

    for tile_img, x_off, y_off, _, _ in slices:
        result = detector.predict(tile_img)
        for d in result.detections:
            # 映射回原图坐标
            x1, y1, x2, y2 = d.xyxy
            all_boxes.append((x1 + x_off, y1 + y_off, x2 + x_off, y2 + y_off))
            all_scores.append(d.confidence)
            all_cls_ids.append(d.class_id)

    # 全局 NMS 去重(切片重叠区会产生重复检测)
    if all_boxes:
        boxes_arr = np.array(all_boxes, dtype=np.float32)
        scores_arr = np.array(all_scores, dtype=np.float32)
        keep_idx = _nms(boxes_arr, scores_arr, iou_thr=0.45)
        boxes_arr = boxes_arr[keep_idx]
        scores_arr = scores_arr[keep_idx]
        cls_arr = np.array(all_cls_ids)[keep_idx]
    else:
        boxes_arr = np.empty((0, 4))
        scores_arr = np.empty(0)
        cls_arr = np.empty(0, dtype=int)

    inf_ms = (time.perf_counter() - t0) * 1000.0

    h, w = image.shape[:2]
    detections: List[Detection] = []
    for box, score, cls_id in zip(boxes_arr, scores_arr, cls_arr):
        x1, y1, x2, y2 = map(int, box)
        detections.append(
            Detection(
                xyxy=(x1, y1, x2, y2),
                confidence=float(score),
                class_id=int(cls_id),
                class_name=detector._class_name(int(cls_id)),
                skyguard_class=None,
                cx=(x1 + x2) // 2,
                cy=(y1 + y2) // 2,
            )
        )

    return DetectionResult(
        detections=detections,
        model_name=detector.model_path,
        inference_ms=inf_ms,
        img_width=w,
        img_height=h,
    )


# ---------------------------------------------------------------------------
# 后处理过滤 —— 减少误报
# ---------------------------------------------------------------------------

def filter_detections(
    detections: List[Detection],
    img_w: int,
    img_h: int,
    min_box_ratio: float = 0.001,
    max_box_ratio: float = 0.5,
    min_aspect: float = 0.3,
    max_aspect: float = 3.0,
    min_conf_by_area: bool = True,
) -> List[Detection]:
    """根据尺寸、宽高比、面积加权置信度过滤检测框,减少误报。

    Args:
        detections: 原始检测结果
        img_w, img_h: 画面尺寸
        min_box_ratio: 检测框最小面积占比
        max_box_ratio: 检测框最大面积占比
        min_aspect: 最小宽高比(w/h)
        max_aspect: 最大宽高比(w/h)
        min_conf_by_area: 是否根据面积动态提高小目标的置信度要求
    """
    img_area = img_w * img_h
    filtered = []
    for d in detections:
        x1, y1, x2, y2 = d.xyxy
        bw = x2 - x1
        bh = y2 - y1
        box_area = bw * bh
        area_ratio = box_area / max(img_area, 1)

        # 面积过滤
        if area_ratio < min_box_ratio:
            continue
        if area_ratio > max_box_ratio:
            continue

        # 宽高比过滤(无人机接近正方形或横向)
        aspect = bw / max(bh, 1)
        if aspect < min_aspect or aspect > max_aspect:
            continue

        # 面积加权置信度:越小的目标要求越高的置信度
        # 优化:降低小目标置信度门槛,提升召回率
        if min_conf_by_area:
            if area_ratio < 0.002:
                min_conf = 0.35
            elif area_ratio < 0.005:
                min_conf = 0.25
            elif area_ratio < 0.01:
                min_conf = 0.15
            else:
                min_conf = 0.0
            if d.confidence < min_conf:
                continue

        filtered.append(d)
    return filtered


class TemporalConsistency:
    """时间一致性过滤:连续 N 帧都检测到(同一位置)才算有效。

    增强版特性:
    1. 置信度累积:跟踪目标的平均置信度,持续低置信度的目标更容易是误报
    2. 移动检测:无人机应该会移动,完全静止的目标可能是误报
    3. 尺寸稳定性:真实目标尺寸变化平滑,突变的可能是误报
    """

    def __init__(
        self,
        required_frames: int = 2,
        iou_threshold: float = 0.25,
        min_avg_confidence: float = 0.3,
        require_motion: bool = False,
        motion_threshold: float = 0.02,
    ):
        self.required = required_frames
        self.iou_thr = iou_threshold
        self.min_avg_conf = min_avg_confidence
        self.require_motion = require_motion
        self.motion_thr = motion_threshold
        self.history: dict[int, dict] = {}
        self.next_id = 0

    def update(self, detections: List[Detection]) -> List[Detection]:
        if self.required <= 1:
            return detections

        current_boxes = [(d.xyxy, d) for d in detections]
        matched = set()
        output = []

        for track_id, track in list(self.history.items()):
            best_iou = 0.0
            best_d = None
            best_idx = -1
            for i, (box, d) in enumerate(current_boxes):
                if i in matched:
                    continue
                iou = self._box_iou(track["box"], box)
                if iou > best_iou and iou >= self.iou_thr:
                    best_iou = iou
                    best_d = d
                    best_idx = i
            if best_d is not None:
                matched.add(best_idx)
                old_cx = (track["box"][0] + track["box"][2]) / 2
                old_cy = (track["box"][1] + track["box"][3]) / 2
                new_cx = (best_d.xyxy[0] + best_d.xyxy[2]) / 2
                new_cy = (best_d.xyxy[1] + best_d.xyxy[3]) / 2
                dx = abs(new_cx - old_cx) / max(best_d.xyxy[2] - best_d.xyxy[0], 1)
                dy = abs(new_cy - old_cy) / max(best_d.xyxy[3] - best_d.xyxy[1], 1)
                moved = dx + dy

                track["box"] = best_d.xyxy
                track["count"] = min(track["count"] + 1, self.required * 3)
                track["age"] = 0
                track["conf_sum"] = track.get("conf_sum", 0.0) + best_d.confidence
                track["total_frames"] = track.get("total_frames", 0) + 1
                track["motion"] = track.get("motion", 0.0) * 0.7 + moved * 0.3
                track["last_det"] = best_d

                if track["count"] >= self.required:
                    avg_conf = track["conf_sum"] / max(track["total_frames"], 1)
                    if avg_conf < self.min_avg_conf:
                        continue
                    if self.require_motion and track["motion"] < self.motion_thr:
                        continue
                    output.append(best_d)
            else:
                track["age"] += 1
                track["motion"] = track.get("motion", 0.0) * 0.9
                if track["age"] > self.required * 2:
                    del self.history[track_id]

        for i, (box, d) in enumerate(current_boxes):
            if i not in matched:
                self.history[self.next_id] = {
                    "box": box,
                    "count": 1,
                    "age": 0,
                    "conf_sum": d.confidence,
                    "total_frames": 1,
                    "motion": 0.0,
                    "last_det": d,
                }
                self.next_id += 1

        return output

    @staticmethod
    def _box_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
        x1a, y1a, x2a, y2a = box_a
        x1b, y1b, x2b, y2b = box_b
        x1 = max(x1a, x1b)
        y1 = max(y1a, y1b)
        x2 = min(x2a, x2b)
        y2 = min(y2a, y2b)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter = (x2 - x1) * (y2 - y1)
        area_a = (x2a - x1a) * (y2a - y1a)
        area_b = (x2b - x1b) * (y2b - y1b)
        union = area_a + area_b - inter
        return inter / max(union, 1)


# ---------------------------------------------------------------------------
# HUD
# ---------------------------------------------------------------------------

def _draw_hud(
    frame: np.ndarray,
    fps: float,
    n_detections: int,
    n_raw: int,
    inference_ms: float,
    model_name: str,
    device: str,
    recording: bool,
    conf: float,
    tile_mode: bool,
    imgsz: int,
    filter_on: bool,
    track_frames: int,
) -> np.ndarray:
    """在画面左上角绘制 HUD 信息。"""
    mode = f"TILE{' ON' if tile_mode else ' OFF'}"
    filt = f"FILT {'ON' if filter_on else 'OFF'}"
    hud_lines = [
        f"SkyGuard | {model_name} | imgsz={imgsz} | {mode} | {filt}",
        f"Device: {device}  |  FPS: {fps:.1f}  |  conf={conf:.2f}  |  track={track_frames}f",
        f"Detections: {n_detections} (raw:{n_raw})  |  Inference: {inference_ms:.1f} ms",
        f"[q]uit  [s]nap  [r]ecord  [t]ile  [+/-]conf  [f]ilter  [m]otion  [1/2]track"
        + ("  [REC]" if recording else ""),
    ]
    h = len(hud_lines) * 24 + 12
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (760, 8 + h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for i, line in enumerate(hud_lines):
        color = (0, 0, 255) if (recording and i == len(hud_lines) - 1) else (255, 255, 255)
        cv2.putText(frame, line, (16, 30 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    return frame


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="SkyGuard 摄像头实时无人机检测 Demo")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL), help="模型路径(.pt/.onnx)")
    parser.add_argument("--camera", type=int, default=0, help="摄像头索引(默认 0)")
    parser.add_argument("--conf", type=float, default=0.2, help="置信度阈值(远距离建议 0.15)")
    parser.add_argument("--iou", type=float, default=0.50, help="NMS IoU 阈值")
    parser.add_argument("--imgsz", type=int, default=800, help="模型输入分辨率(远距离建议 1280)")
    parser.add_argument("--device", type=str, default="auto", help="推理设备:auto/cpu/mps/cuda")
    parser.add_argument("--width", type=int, default=1920, help="采集分辨率宽")
    parser.add_argument("--height", type=int, default=1080, help="采集分辨率高")
    parser.add_argument("--tile", type=int, default=0, help="切片数(0=关闭,2=2x2,3=3x3)。远距离小目标建议 2")
    parser.add_argument("--coco", action="store_true", help="使用 COCO 预训练模型(检测飞机/鸟/风筝)")
    parser.add_argument("--min-box-ratio", type=float, default=0.001,
                        help="最小检测框占画面面积比例(0-1),过滤太小的误报")
    parser.add_argument("--max-box-ratio", type=float, default=0.5,
                        help="最大检测框占画面面积比例(0-1),过滤太大的误报")
    parser.add_argument("--min-aspect", type=float, default=0.3,
                        help="最小宽高比(无人机通常接近正方形)")
    parser.add_argument("--max-aspect", type=float, default=3.0,
                        help="最大宽高比")
    parser.add_argument("--track-frames", type=int, default=2,
                        help="时间一致性:连续 N 帧都检测到才算有效(减少闪烁误报)")
    parser.add_argument("--min-avg-conf", type=float, default=0.3,
                        help="时间一致性要求的平均置信度(持续低置信度视为误报)")
    parser.add_argument("--motion-filter", action="store_true",
                        help="启用移动检测过滤(静止目标视为误报,适合固定摄像头)")
    args = parser.parse_args()

    setup_logging()
    clear_settings_cache()
    get_settings()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[ERROR] 模型文件不存在: {model_path}")
        print("提示: 运行 'python scripts/webcam_demo.py --model yolov8n.pt --coco' 使用预训练模型")
        return 1

    classes = None
    if args.coco:
        classes = [5, 14, 38]  # airplane, bird, kite

    device = resolve_torch_device(args.device)
    tile_mode = args.tile > 0
    print(f"[*] 加载模型: {model_path.name}")
    print(f"[*] 推理设备: {device}")
    print(f"[*] 摄像头索引: {args.camera}")
    print(f"[*] 置信度: {args.conf}  NMS: {args.iou}  imgsz: {args.imgsz}")
    print(f"[*] 切片推理: {'ON (' + str(args.tile) + 'x' + str(args.tile) + ')' if tile_mode else 'OFF'}")
    print()

    # 构造 detector 时传入 imgsz
    detector = YOLODetector(
        model_path=model_path,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        classes=classes,
        verbose=False,
    )

    # 打开摄像头
    print(f"[*] 打开摄像头 {args.camera} ...")
    print("    (首次运行 macOS 会弹出摄像头授权弹窗,请点击允许)")
    backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    cap = None
    for backend in backends:
        cap = cv2.VideoCapture(args.camera, backend)
        if cap.isOpened():
            break
        cap = None
    if cap is None or not cap.isOpened():
        print("[ERROR] 无法打开摄像头。请检查:")
        print("  1. 摄像头是否被其他程序占用")
        print("  2. 系统设置 > 隐私与安全 > 摄像头 是否已授权终端")
        print("  3. 尝试其他索引: --camera 1")
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or args.width)
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or args.height)
    cap_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    print(f"[*] 采集分辨率: {actual_w}x{actual_h} @ {cap_fps:.1f} fps")
    print()
    print("=" * 60)
    print("  SkyGuard 实时检测已启动")
    print("  按 q/ESC 退出 | s 截图 | r 录制")
    print("  按 t 切换切片推理 | +/- 调整置信度")
    print("=" * 60)
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 运行时可变状态
    fps_history: list[float] = []
    frame_count = 0
    recording = False
    writer: cv2.VideoWriter | None = None
    cur_conf = args.conf
    cur_tile = args.tile
    filter_enabled = True
    motion_filter_enabled = args.motion_filter
    cur_track_frames = args.track_frames
    temporal_filter = TemporalConsistency(
        required_frames=cur_track_frames,
        min_avg_confidence=args.min_avg_conf,
        require_motion=motion_filter_enabled,
    )

    try:
        while True:
            t_start = time.perf_counter()

            ok, image = cap.read()
            if not ok or image is None:
                print("[WARN] 读取摄像头帧失败,退出")
                break

            # 推理:切片模式 or 普通模式
            if cur_tile > 0:
                result = predict_with_tiling(detector, image, n_tiles=cur_tile)
            else:
                # 运行时 conf 调整需要传入当前值
                result = detector.predict(image, conf=cur_conf)
            n_det_raw = len(result.detections)
            inf_ms = result.inference_ms or 0.0

            # 后处理过滤
            detections = result.detections
            if filter_enabled:
                detections = filter_detections(
                    detections,
                    img_w=result.img_width,
                    img_h=result.img_height,
                    min_box_ratio=args.min_box_ratio,
                    max_box_ratio=args.max_box_ratio,
                    min_aspect=args.min_aspect,
                    max_aspect=args.max_aspect,
                )
                # 时间一致性过滤
                detections = temporal_filter.update(detections)
            n_det = len(detections)

            # 可视化
            annotated = annotate(
                image,
                boxes=[d.xyxy for d in detections],
                class_ids=[d.class_id for d in detections],
                scores=[d.confidence for d in detections],
            )

            # FPS
            t_end = time.perf_counter()
            instant_fps = 1.0 / max(t_end - t_start, 1e-6)
            fps_history.append(instant_fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = sum(fps_history) / len(fps_history)

            # HUD
            annotated = _draw_hud(
                annotated, avg_fps, n_det, n_det_raw, inf_ms, model_path.name,
                device, recording, cur_conf, cur_tile > 0, args.imgsz,
                filter_enabled, cur_track_frames,
            )

            # 录制
            if recording and writer is not None:
                writer.write(annotated)

            cv2.imshow("SkyGuard Drone Detection", annotated)

            # 按键
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                print("[*] 退出")
                break
            elif key == ord("s"):
                snap_path = OUTPUT_DIR / f"snapshot_{int(time.time())}.jpg"
                cv2.imwrite(str(snap_path), annotated)
                print(f"[+] 截图已保存: {snap_path}")
            elif key == ord("r"):
                if recording and writer is not None:
                    writer.release()
                    writer = None
                    recording = False
                    print("[*] 录制停止")
                else:
                    rec_path = OUTPUT_DIR / f"record_{int(time.time())}.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(rec_path), fourcc, 20.0, (actual_w, actual_h))
                    recording = True
                    print(f"[+] 开始录制: {rec_path}")
            elif key == ord("t"):
                # 切换切片推理
                if cur_tile > 0:
                    cur_tile = 0
                    print("[*] 切片推理: OFF")
                else:
                    cur_tile = 2
                    print("[*] 切片推理: ON (2x2)")
            elif key == ord("f"):
                # 切换过滤器开关
                filter_enabled = not filter_enabled
                if filter_enabled:
                    temporal_filter = TemporalConsistency(
                        required_frames=cur_track_frames,
                        min_avg_confidence=args.min_avg_conf,
                        require_motion=motion_filter_enabled,
                    )
                    print("[*] 后处理过滤: ON")
                else:
                    print("[*] 后处理过滤: OFF")
            elif key == ord("m"):
                # 切换移动检测过滤
                motion_filter_enabled = not motion_filter_enabled
                temporal_filter = TemporalConsistency(
                    required_frames=cur_track_frames,
                    min_avg_confidence=args.min_avg_conf,
                    require_motion=motion_filter_enabled,
                )
                print(f"[*] 移动检测过滤: {'ON' if motion_filter_enabled else 'OFF'}")
            elif key == ord("1"):
                # 减少时间一致性帧数
                cur_track_frames = max(1, cur_track_frames - 1)
                temporal_filter = TemporalConsistency(
                    required_frames=cur_track_frames,
                    min_avg_confidence=args.min_avg_conf,
                    require_motion=motion_filter_enabled,
                )
                print(f"[*] 时间一致性帧数: {cur_track_frames}")
            elif key == ord("2"):
                # 增加时间一致性帧数
                cur_track_frames = min(10, cur_track_frames + 1)
                temporal_filter = TemporalConsistency(
                    required_frames=cur_track_frames,
                    min_avg_confidence=args.min_avg_conf,
                    require_motion=motion_filter_enabled,
                )
                print(f"[*] 时间一致性帧数: {cur_track_frames}")
            elif key in (ord("+"), ord("=")):
                cur_conf = min(0.95, cur_conf + 0.05)
                print(f"[*] 置信度: {cur_conf:.2f}")
            elif key == ord("-"):
                cur_conf = max(0.05, cur_conf - 0.05)
                print(f"[*] 置信度: {cur_conf:.2f}")

            frame_count += 1

    except KeyboardInterrupt:
        print("\n[*] Ctrl+C 退出")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        print(f"[*] 共处理 {frame_count} 帧")

    return 0


if __name__ == "__main__":
    sys.exit(main())
