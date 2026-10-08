#!/usr/bin/env python3
"""
SkyGuard 无人机检测系统 - Web Demo 后端服务器
支持单模型和 WBF 集成模型的动态切换
"""
import os
import sys
import json
import time
import base64
import threading
import cv2
import numpy as np
from pathlib import Path
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit

ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = ROOT / "config/models.json"

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")


# ============================================================
# 模型配置加载
# ============================================================

def load_model_config():
    """加载模型配置"""
    with open(CONFIG_PATH, 'r') as f:
        return json.load(f)


MODEL_CONFIG = load_model_config()
AVAILABLE_MODELS = MODEL_CONFIG["models"]
AVAILABLE_ENSEMBLES = MODEL_CONFIG["ensembles"]


# ============================================================
# 模型管理器 - 支持单模型和集成模型
# ============================================================

class ModelManager:
    """管理多个模型,支持单模型和 WBF 集成"""

    def __init__(self):
        self.models = {}  # name -> YOLO model
        self.current_mode = "single"  # "single" or "ensemble"
        self.current_model_name = None
        self.current_ensemble_name = None
        self.current_ensemble_models = []

    def load_single(self, model_name):
        """加载单个模型"""
        from ultralytics import YOLO

        if model_name not in AVAILABLE_MODELS:
            return False, f"未知模型: {model_name}"

        info = AVAILABLE_MODELS[model_name]
        model_path = ROOT / info["path"]

        if not model_path.exists():
            return False, f"模型文件不存在: {info['path']}"

        if model_name not in self.models:
            self.models[model_name] = YOLO(str(model_path))

        self.current_mode = "single"
        self.current_model_name = model_name
        self.current_ensemble_name = None
        self.current_ensemble_models = [model_name]
        return True, f"已加载模型: {info['name']}"

    def load_ensemble(self, ensemble_name):
        """加载集成模型"""
        from ultralytics import YOLO

        if ensemble_name not in AVAILABLE_ENSEMBLES:
            return False, f"未知集成配置: {ensemble_name}"

        ensemble = AVAILABLE_ENSEMBLES[ensemble_name]
        model_names = ensemble["models"]

        # 验证所有模型存在
        for name in model_names:
            if name not in AVAILABLE_MODELS:
                return False, f"集成配置中未知模型: {name}"
            info = AVAILABLE_MODELS[name]
            model_path = ROOT / info["path"]
            if not model_path.exists():
                return False, f"模型文件不存在: {info['path']}"

        # 加载所有模型
        for name in model_names:
            if name not in self.models:
                info = AVAILABLE_MODELS[name]
                self.models[name] = YOLO(str(ROOT / info["path"]))

        self.current_mode = "ensemble"
        self.current_ensemble_name = ensemble_name
        self.current_model_name = None
        self.current_ensemble_models = model_names
        return True, f"已加载集成: {ensemble['name']}"

    def is_loaded(self):
        """检查是否已加载模型"""
        return len(self.current_ensemble_models) > 0

    def get_model_info(self):
        """获取当前模型信息"""
        if self.current_mode == "single" and self.current_model_name:
            info = AVAILABLE_MODELS[self.current_model_name]
            return {
                "mode": "single",
                "modelName": info["name"],
                "modelId": self.current_model_name,
                "architecture": info["architecture"],
                "classes": info["classes"],
                "mAP50": f"{info['mAP50']*100:.1f}%",
                "fps": info.get("fps", 0),
                "description": info.get("description", ""),
            }
        elif self.current_mode == "ensemble" and self.current_ensemble_name:
            ensemble = AVAILABLE_ENSEMBLES[self.current_ensemble_name]
            return {
                "mode": "ensemble",
                "modelName": ensemble["name"],
                "modelId": self.current_ensemble_name,
                "architecture": "WBF Ensemble",
                "classes": ["uav"],
                "mAP50": "100% (eval)",
                "fps": 0,
                "description": ensemble.get("description", ""),
                "subModels": [AVAILABLE_MODELS[m]["name"] for m in ensemble["models"]],
            }
        return None

    def detect(self, frame, conf, iou, imgsz, device, wbf_iou=0.55, filter_enabled=True):
        """执行检测,支持单模型和集成模式"""
        if not self.current_ensemble_models:
            return frame, []

        h, w = frame.shape[:2]

        if self.current_mode == "single" and self.current_model_name:
            return self._detect_single(frame, self.current_model_name, conf, iou, imgsz, device, filter_enabled)
        else:
            return self._detect_ensemble(frame, conf, iou, imgsz, device, wbf_iou, filter_enabled)

    def _detect_single(self, frame, model_name, conf, iou, imgsz, device, filter_enabled):
        """单模型检测"""
        model = self.models[model_name]
        info = AVAILABLE_MODELS[model_name]
        class_map = info.get("class_map", {"0": "uav"})
        filter_set = set(info.get("filter_classes", []))

        results = model(frame, conf=conf, iou=iou, imgsz=imgsz, device=device, verbose=False)
        result = results[0]

        detections = []
        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                orig_class = class_map.get(str(cls_id), class_map.get(cls_id, "uav"))
                if orig_class in filter_set:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf_val = float(box.conf[0])
                detections.append({
                    "id": len(detections) + 1,
                    "class": "uav",
                    "conf": round(conf_val, 4),
                    "x": x1,
                    "y": y1,
                    "w": x2 - x1,
                    "h": y2 - y1,
                })

        if filter_enabled:
            h, w = frame.shape[:2]
            detections = filter_detections(detections, w, h)

        return self._draw_detections(frame, detections)

    def _detect_ensemble(self, frame, conf, iou, imgsz, device, wbf_iou, filter_enabled):
        """WBF 集成检测"""
        # 捕获当前模型列表，避免在多线程环境下列表被修改导致长度不匹配
        model_names = list(self.current_ensemble_models)
        all_detections = []

        for name in model_names:
            model = self.models[name]
            info = AVAILABLE_MODELS[name]
            class_map = info.get("class_map", {"0": "uav"})
            filter_set = set(info.get("filter_classes", []))

            results = model(frame, conf=conf, iou=iou, imgsz=imgsz, device=device, verbose=False)
            result = results[0]

            dets = []
            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    orig_class = class_map.get(str(cls_id), class_map.get(cls_id, "uav"))
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
            all_detections.append(dets)

        # WBF 融合
        weights = []
        ensemble = AVAILABLE_ENSEMBLES.get(self.current_ensemble_name, {})
        ensemble_weights = ensemble.get("weights", {})
        for name in model_names:
            weights.append(ensemble_weights.get(name, 1.0))

        fused = weighted_box_fusion(all_detections, weights, iou_thr=wbf_iou)

        # 转为输出格式
        detections = []
        for i, d in enumerate(fused):
            detections.append({
                "id": i + 1,
                "class": "uav",
                "conf": round(d["conf"], 4),
                "x": int(d["x"]),
                "y": int(d["y"]),
                "w": int(d["w"]),
                "h": int(d["h"]),
                "modelCount": d.get("model_count", 1),
            })

        if filter_enabled:
            h, w = frame.shape[:2]
            detections = filter_detections(detections, w, h)

        return self._draw_detections(frame, detections, is_ensemble=True)

    def _draw_detections(self, frame, detections, is_ensemble=False):
        """绘制检测框"""
        for d in detections:
            x1, y1 = d["x"], d["y"]
            x2, y2 = x1 + d["w"], y1 + d["h"]

            if is_ensemble and d.get("modelCount", 1) >= 2:
                color = (255, 200, 0)
            else:
                color = (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"{d['class']} {d['conf']:.0%}"
            if is_ensemble and d.get("modelCount", 1) >= 2:
                label += f" [{d['modelCount']}M]"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return frame, detections


# ============================================================
# WBF 核心算法
# ============================================================

def _box_iou(a, b):
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


def weighted_box_fusion(detections_list, weights, iou_thr=0.55):
    """加权框融合"""
    # 防御性检查: 确保 detections_list 和 weights 长度一致
    if len(weights) != len(detections_list):
        weights = [1.0] * len(detections_list)
    total_w = sum(weights)
    if total_w == 0:
        total_w = 1.0
    weights = [w / total_w for w in weights]

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

    all_boxes.sort(key=lambda b: b["conf"], reverse=True)

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

    fused = []
    for cluster in clusters:
        if len(cluster) == 1:
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
        else:
            total_weight = sum(b["weight"] for b in cluster)
            fused_x1 = sum(b["x1"] * b["weight"] for b in cluster) / total_weight
            fused_y1 = sum(b["y1"] * b["weight"] for b in cluster) / total_weight
            fused_x2 = sum(b["x2"] * b["weight"] for b in cluster) / total_weight
            fused_y2 = sum(b["y2"] * b["weight"] for b in cluster) / total_weight
            fused_conf = sum(b["conf"] * b["weight"] for b in cluster) / total_weight
            fused.append({
                "x": fused_x1,
                "y": fused_y1,
                "w": fused_x2 - fused_x1,
                "h": fused_y2 - fused_y1,
                "conf": fused_conf,
                "class": cluster[0]["class"],
                "model_count": len(set(b["model_idx"] for b in cluster)),
            })

    fused.sort(key=lambda d: d["conf"], reverse=True)
    return fused


# ============================================================
# 后处理过滤
# ============================================================

def filter_detections(detections, img_w, img_h):
    """根据尺寸、宽高比、面积加权置信度过滤检测框"""
    img_area = img_w * img_h
    filtered = []
    for d in detections:
        bw = d["w"]
        bh = d["h"]
        box_area = bw * bh
        area_ratio = box_area / max(img_area, 1)

        if area_ratio < 0.001 or area_ratio > 0.5:
            continue

        aspect = bw / max(bh, 1)
        if aspect < 0.3 or aspect > 3.0:
            continue

        if area_ratio < 0.002:
            min_conf = 0.35
        elif area_ratio < 0.005:
            min_conf = 0.25
        elif area_ratio < 0.01:
            min_conf = 0.15
        else:
            min_conf = 0.0
        if d["conf"] < min_conf:
            continue

        filtered.append(d)
    return filtered


# ============================================================
# 时间一致性过滤
# ============================================================

class TemporalConsistency:
    def __init__(self, required_frames=2, iou_threshold=0.25, min_avg_confidence=0.3):
        self.required = required_frames
        self.iou_thr = iou_threshold
        self.min_avg_conf = min_avg_confidence
        self.history = {}
        self.next_id = 0

    def update(self, detections):
        if self.required <= 1:
            return detections

        current_boxes = []
        for d in detections:
            box = (d["x"], d["y"], d["x"] + d["w"], d["y"] + d["h"])
            current_boxes.append((box, d))

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
                track["box"] = current_boxes[best_idx][0]
                track["count"] = min(track["count"] + 1, self.required * 3)
                track["age"] = 0
                track["conf_sum"] = track.get("conf_sum", 0.0) + best_d["conf"]
                track["total_frames"] = track.get("total_frames", 0) + 1

                if track["count"] >= self.required:
                    avg_conf = track["conf_sum"] / max(track["total_frames"], 1)
                    if avg_conf >= self.min_avg_conf:
                        output.append(best_d)
            else:
                track["age"] += 1
                if track["age"] > self.required * 2:
                    del self.history[track_id]

        for i, (box, d) in enumerate(current_boxes):
            if i not in matched:
                self.history[self.next_id] = {
                    "box": box,
                    "count": 1,
                    "age": 0,
                    "conf_sum": d["conf"],
                    "total_frames": 1,
                }
                self.next_id += 1

        return output

    @staticmethod
    def _box_iou(box_a, box_b):
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter = (x2 - x1) * (y2 - y1)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - inter
        return inter / max(union, 1)


# ============================================================
# 全局状态
# ============================================================

model_mgr = ModelManager()
temporal_filter = TemporalConsistency(required_frames=2, iou_threshold=0.25, min_avg_confidence=0.3)

is_detecting = False
current_params = {
    "conf": 0.20,
    "iou": 0.50,
    "imgsz": 800,
    "device": "mps",
    "tile_mode": False,
    "filter_enabled": True,
    "wbf_iou": 0.55,
}
video_capture = None
detection_thread = None
stop_event = threading.Event()


# ============================================================
# API 路由
# ============================================================

@app.route("/api/models", methods=["GET"])
def get_models():
    """获取所有可用模型列表"""
    models_list = []
    for key, info in AVAILABLE_MODELS.items():
        model_path = ROOT / info["path"]
        models_list.append({
            "id": key,
            "name": info["name"],
            "architecture": info["architecture"],
            "dataset": info["dataset"],
            "mAP50": info["mAP50"],
            "fps": info.get("fps", 0),
            "size_mb": info.get("size_mb", 0),
            "classes": info["classes"],
            "description": info.get("description", ""),
            "tags": info.get("tags", []),
            "available": model_path.exists(),
            "type": "single",
        })

    ensembles_list = []
    for key, info in AVAILABLE_ENSEMBLES.items():
        ensembles_list.append({
            "id": key,
            "name": info["name"],
            "models": info["models"],
            "weights": info["weights"],
            "description": info.get("description", ""),
            "tags": info.get("tags", []),
            "type": "ensemble",
        })

    return jsonify({
        "models": models_list,
        "ensembles": ensembles_list,
        "current": model_mgr.get_model_info(),
    })


@app.route("/api/load-model", methods=["POST"])
def load_model():
    """加载单模型"""
    data = request.json
    model_id = data.get("model_id")

    if not model_id:
        return jsonify({"success": False, "message": "请指定模型ID"}), 400

    if model_id not in AVAILABLE_MODELS:
        return jsonify({"success": False, "message": f"未知模型: {model_id}"}), 400

    # 停止检测
    global is_detecting
    if is_detecting:
        stop_event.set()
        is_detecting = False

    success, message = model_mgr.load_single(model_id)
    return jsonify({
        "success": success,
        "message": message,
        "model_info": model_mgr.get_model_info() if success else None,
    })


@app.route("/api/load-ensemble", methods=["POST"])
def load_ensemble():
    """加载集成模型"""
    data = request.json
    ensemble_id = data.get("ensemble_id")

    if not ensemble_id:
        return jsonify({"success": False, "message": "请指定集成ID"}), 400

    if ensemble_id not in AVAILABLE_ENSEMBLES:
        return jsonify({"success": False, "message": f"未知集成: {ensemble_id}"}), 400

    # 停止检测
    global is_detecting
    if is_detecting:
        stop_event.set()
        is_detecting = False

    success, message = model_mgr.load_ensemble(ensemble_id)
    return jsonify({
        "success": success,
        "message": message,
        "model_info": model_mgr.get_model_info() if success else None,
    })


@app.route("/api/status", methods=["GET"])
def get_status():
    """获取系统状态"""
    return jsonify({
        "is_connected": model_mgr.is_loaded(),
        "is_detecting": is_detecting,
        "model_info": model_mgr.get_model_info(),
        "params": current_params,
    })


@app.route("/api/detect-image", methods=["POST"])
def detect_image():
    """图片推理 API - 上传图片进行检测"""
    import base64 as b64

    data = request.json
    if not data or "image" not in data:
        return jsonify({"success": False, "message": "请提供图片数据"}), 400

    if not model_mgr.is_loaded():
        return jsonify({"success": False, "message": "请先加载模型"}), 400

    try:
        img_data = data["image"]
        if "," in img_data:
            img_data = img_data.split(",")[1]
        img_bytes = b64.b64decode(img_data)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"success": False, "message": "图片解析失败"}), 400

        annotated, detections = model_mgr.detect(
            frame,
            conf=current_params["conf"],
            iou=current_params["iou"],
            imgsz=current_params["imgsz"],
            device=current_params["device"],
            wbf_iou=current_params.get("wbf_iou", 0.55),
            filter_enabled=current_params.get("filter_enabled", True),
        )

        _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_b64 = b64.b64encode(buffer).decode("utf-8")

        return jsonify({
            "success": True,
            "image": img_b64,
            "detections": detections,
            "count": len(detections),
            "model_info": model_mgr.get_model_info(),
        })
    except Exception as e:
        return jsonify({"success": False, "message": f"推理错误: {str(e)}"}), 500


@app.route("/api/params", methods=["POST"])
def update_params():
    """更新检测参数"""
    global current_params
    data = request.json
    for key in ["conf", "iou", "imgsz", "device", "wbf_iou"]:
        if key in data:
            current_params[key] = data[key]
    current_params["tile_mode"] = data.get("tile_mode", current_params.get("tile_mode", False))
    current_params["filter_enabled"] = data.get("filter_enabled", current_params.get("filter_enabled", True))
    return jsonify({"success": True, "params": current_params})


# ============================================================
# 检测工作线程
# ============================================================

def detection_worker():
    """检测工作线程"""
    global video_capture, is_detecting

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        socketio.emit("error", {"message": "无法打开摄像头"})
        is_detecting = False
        return

    video_capture = cap
    stop_event.clear()
    frame_count = 0
    last_fps_time = time.time()
    fps = 0

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        annotated_frame, detections = model_mgr.detect(
            frame.copy(),
            conf=current_params["conf"],
            iou=current_params["iou"],
            imgsz=current_params["imgsz"],
            device=current_params["device"],
            wbf_iou=current_params.get("wbf_iou", 0.55),
            filter_enabled=current_params.get("filter_enabled", True),
        )

        # 时间一致性过滤
        if current_params.get("filter_enabled", True):
            detections = temporal_filter.update(detections)

        frame_count += 1
        now = time.time()
        if now - last_fps_time >= 1:
            fps = frame_count
            frame_count = 0
            last_fps_time = now

        _, buffer = cv2.imencode(".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_data = base64.b64encode(buffer).decode("utf-8")

        socketio.emit("frame", {
            "image": frame_data,
            "detections": detections,
            "fps": fps,
        })

        time.sleep(0.01)

    cap.release()
    video_capture = None
    is_detecting = False


# ============================================================
# WebSocket 事件处理
# ============================================================

@socketio.on("update_params")
def handle_update_params(data):
    """通过 WebSocket 更新参数"""
    global current_params
    if "params" in data:
        for key in ["conf", "iou", "imgsz", "device", "wbf_iou"]:
            if key in data["params"]:
                current_params[key] = data["params"][key]
        current_params["tile_mode"] = data["params"].get("tile_mode", current_params.get("tile_mode", False))
        current_params["filter_enabled"] = data["params"].get("filter_enabled", current_params.get("filter_enabled", True))
    return {"success": True, "params": current_params}


@socketio.on("load_model")
def handle_load_model(data):
    """通过 WebSocket 加载单模型"""
    model_id = data.get("model_id")
    if not model_id:
        return {"success": False, "message": "请指定模型ID"}

    global is_detecting
    if is_detecting:
        stop_event.set()
        is_detecting = False

    success, message = model_mgr.load_single(model_id)
    info = model_mgr.get_model_info() if success else None
    emit("model_loaded", {"success": success, "message": message, "model_info": info})
    return {"success": success, "message": message}


@socketio.on("load_ensemble")
def handle_load_ensemble(data):
    """通过 WebSocket 加载集成模型"""
    ensemble_id = data.get("ensemble_id")
    if not ensemble_id:
        return {"success": False, "message": "请指定集成ID"}

    global is_detecting
    if is_detecting:
        stop_event.set()
        is_detecting = False

    success, message = model_mgr.load_ensemble(ensemble_id)
    info = model_mgr.get_model_info() if success else None
    emit("model_loaded", {"success": success, "message": message, "model_info": info})
    return {"success": success, "message": message}


@socketio.on("start_detection")
def start_detection(data):
    """开始检测"""
    global is_detecting, detection_thread

    if not model_mgr.is_loaded():
        return {"success": False, "message": "请先加载模型"}

    if is_detecting:
        return {"success": False, "message": "检测已在运行中"}

    if "params" in data:
        global current_params
        for key in ["conf", "iou", "imgsz", "device", "wbf_iou"]:
            if key in data["params"]:
                current_params[key] = data["params"][key]

    is_detecting = True
    stop_event.clear()
    temporal_filter.history.clear()

    detection_thread = threading.Thread(target=detection_worker, daemon=True)
    detection_thread.start()

    return {"success": True}


@socketio.on("stop_detection")
def stop_detection():
    """停止检测"""
    global is_detecting

    if not is_detecting:
        return {"success": False, "message": "检测未在运行"}

    stop_event.set()
    is_detecting = False
    temporal_filter.history.clear()
    return {"success": True}


@socketio.on("connect")
def on_connect():
    """客户端连接"""
    emit("status", {
        "is_connected": model_mgr.is_loaded(),
        "is_detecting": is_detecting,
        "params": current_params,
        "model_info": model_mgr.get_model_info(),
    })


@socketio.on("disconnect")
def on_disconnect():
    """客户端断开"""
    pass


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 60)
    print("  SkyGuard 无人机检测系统 - Web Demo 后端")
    print("=" * 60)

    # 加载默认模型
    default_model = MODEL_CONFIG.get("default_model", "v4")
    print(f"\n正在加载默认模型: {default_model}")
    success, message = model_mgr.load_single(default_model)
    if success:
        print(f"  ✅ {message}")
    else:
        print(f"  ⚠️  {message}")

    print("\n可用模型:")
    for key, info in AVAILABLE_MODELS.items():
        path = ROOT / info["path"]
        status = "✅" if path.exists() else "❌"
        print(f"  {status} {key}: {info['name']} ({info['architecture']})")

    print("\n可用集成方案:")
    for key, info in AVAILABLE_ENSEMBLES.items():
        print(f"  ✅ {key}: {info['name']}")

    print("\n启动服务器...")
    print("  前端地址: http://localhost:5173")
    print("  后端地址: http://localhost:5001")
    print("\n按 Ctrl+C 停止服务器\n")

    socketio.run(app, host="0.0.0.0", port=5001, debug=False, allow_unsafe_werkzeug=True)


if __name__ == "__main__":
    main()
