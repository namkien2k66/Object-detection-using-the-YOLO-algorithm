import csv
import math
import os
import shutil
import tempfile
import time
from collections import Counter, defaultdict

import cv2
import numpy as np
import supervision as sv
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

PT_NAME = "yolo11s.pt"
TRACKER_CFG = "custom_bytetrack.yaml"

SOURCE_PATH = os.path.join("Data", "Example Videos", "vehicle.mp4")
DEST_DIR = os.path.join("Data", "Example Results")
OUTPUT_PATH = os.path.join(DEST_DIR, "tracking_result.mp4")

VEHICLE_CLASSES = [2, 3, 5, 7]
CLASS_SHORT = {2: "car", 3: "moto", 5: "bus", 7: "truck"}
CLASS_COLORS = {2: "#2ECC71", 3: "#F1C40F", 5: "#3498DB", 7: "#E74C3C"}

IMGSZ = 1280
MODEL_DIR = f"{os.path.splitext(PT_NAME)[0]}_openvino_model_{IMGSZ}"
CONF = 0.1
IOU = 0.6
AGNOSTIC_NMS = True
SMOOTH_LEN = 2

MAX_LOST_FRAMES = 60
BASE_GATE = 30.0
GATE_GROWTH = 3.0
GATE_SIZE_RATIO = 0.75

VOTE_MIN_CONF = 0.25
VOTE_SWITCH_MARGIN = 1.3

SHOW_CLASS = True
LABEL_SCALE = 0.35
LABEL_THICKNESS = 1
LABEL_PADDING = 2
BOX_THICKNESS = 1

SAVE_VEHICLE_CROPS = True
CROP_DIR = os.path.join(DEST_DIR, "vehicle_crops")
CROP_MAX_SIDE = 256
CROP_PAD = 0.08
CROP_MIN_FRAMES = 5
CONTACT_COLS = 10
CONTACT_PER_SHEET = 100
CONTACT_CELL = 128

ANALYSIS_DIR = os.path.join(DEST_DIR, "analysis")
SNAPSHOT_DIR = os.path.join(ANALYSIS_DIR, "snapshots")
ENABLE_FAILURE_ANALYSIS = True
WARMUP_FRAMES = 5

SMALL_AREA = 32 * 32
SHORT_TRACK_FRAMES = 15
OCC_IOU = 0.3
BLUR_LAP_MAX = 30.0
BLUR_MIN_SPEED = 8.0
MIN_BLUR_SIDE = 32
CROP_NORM = 64
SNAP_MAX_PER_TYPE = 8
SNAP_MIN_GAP = 30


def ensure_model():
    if os.path.isdir(MODEL_DIR):
        return MODEL_DIR
    print(f"Chưa có {MODEL_DIR}, đang export {PT_NAME} sang OpenVINO imgsz={IMGSZ} (chỉ làm 1 lần)...")
    ckpt = getattr(YOLO(PT_NAME), "ckpt_path", PT_NAME)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        tmp_pt = os.path.join(tmp, os.path.basename(ckpt))
        shutil.copy(ckpt, tmp_pt)
        exported = YOLO(tmp_pt).export(format="openvino", imgsz=IMGSZ)
        shutil.move(str(exported), MODEL_DIR)
    return MODEL_DIR


def class_group(cid):
    return "moto" if cid == 3 else "vehicle"


def vote_weights(tracked, min_conf):
    n = len(tracked)
    conf = tracked.confidence if tracked.confidence is not None else np.ones(n)
    w = np.where(conf >= min_conf, conf, 0.0)
    if n >= 2:
        iou = sv.box_iou_batch(tracked.xyxy, tracked.xyxy)
        np.fill_diagonal(iou, 0)
        w = w * (1.0 - np.clip(iou.max(axis=1) / 0.5, 0, 1))
    return np.maximum(w, 0.02)


class StableIDManager:
    def __init__(self, max_lost, base_gate, growth, size_ratio, switch_margin):
        self.max_lost = max_lost
        self.base_gate = base_gate
        self.growth = growth
        self.size_ratio = size_ratio
        self.switch_margin = switch_margin
        self.id_map = {}
        self.history = {}
        self.votes = defaultdict(Counter)
        self.label = {}
        self.next_id = 1

    def _bind(self, raw_id, stable_id):
        for old_raw in [r for r, s in self.id_map.items() if s == stable_id]:
            del self.id_map[old_raw]
        self.id_map[raw_id] = stable_id

    def voted_class(self, sid):
        counter = self.votes.get(sid)
        if not counter:
            return None
        best, best_w = counter.most_common(1)[0]
        cur = self.label.get(sid)
        if cur is None or cur not in counter or best_w > self.switch_margin * counter[cur]:
            self.label[sid] = best
            return best
        return cur

    def update(self, raw_ids, centers, class_ids, sizes, weights, frame_idx):
        assigned = [None] * len(raw_ids)
        used = set()
        unmatched = []

        for i, raw_id in enumerate(raw_ids):
            sid = self.id_map.get(raw_id)
            if sid is not None and sid not in used:
                assigned[i] = sid
                used.add(sid)
            else:
                self.id_map.pop(raw_id, None)
                unmatched.append(i)

        lost = [
            sid for sid, info in self.history.items()
            if sid not in used and frame_idx - info["last_seen"] <= self.max_lost
        ]

        if unmatched and lost:
            cost = np.empty((len(unmatched), len(lost)))
            ok = np.zeros_like(cost, dtype=bool)
            for r, i in enumerate(unmatched):
                cx, cy = centers[i]
                grp = class_group(class_ids[i])
                for c, sid in enumerate(lost):
                    info = self.history[sid]
                    dt = frame_idx - info["last_seen"]
                    px = info["center"][0] + info["vel"][0] * dt
                    py = info["center"][1] + info["vel"][1] * dt
                    d = math.hypot(cx - px, cy - py)
                    gate = max(self.base_gate, self.size_ratio * info["size"]) + self.growth * dt
                    cost[r, c] = d
                    ok[r, c] = (
                        d <= gate
                        and class_group(self.voted_class(sid)) == grp
                    )

            rows, cols = linear_sum_assignment(np.where(ok, cost, 1e6))
            for r, c in zip(rows, cols):
                if not ok[r, c]:
                    continue
                i, sid = unmatched[r], lost[c]
                self._bind(raw_ids[i], sid)
                assigned[i] = sid
                used.add(sid)

        for i in unmatched:
            if assigned[i] is None:
                sid = self.next_id
                self.next_id += 1
                self.id_map[raw_ids[i]] = sid
                assigned[i] = sid
                used.add(sid)

        voted = []
        for sid, (cx, cy), cid, size, wgt in zip(assigned, centers, class_ids, sizes, weights):
            prev = self.history.get(sid)
            vel = (0.0, 0.0)
            if prev is not None:
                dt = frame_idx - prev["last_seen"]
                if dt > 0:
                    nv = ((cx - prev["center"][0]) / dt, (cy - prev["center"][1]) / dt)
                    pv = prev["vel"]
                    vel = (0.6 * nv[0] + 0.4 * pv[0], 0.6 * nv[1] + 0.4 * pv[1])
                else:
                    vel = prev["vel"]
            self.history[sid] = {
                "center": (float(cx), float(cy)),
                "last_seen": frame_idx,
                "vel": vel,
                "size": float(size),
            }
            self.votes[sid][int(cid)] += float(wgt)
            voted.append(self.voted_class(sid))

        expired = [s for s, info in self.history.items()
                   if frame_idx - info["last_seen"] > self.max_lost]
        for s in expired:
            del self.history[s]
            self.votes.pop(s, None)
            self.label.pop(s, None)

        return assigned, voted


def _pct(values, q):
    return float(np.percentile(values, q)) if len(values) else float("nan")


def _mean(values):
    return float(np.mean(values)) if len(values) else float("nan")


class FailureAnalyzer:
    def __init__(self, fps, total_frames):
        self.fps = fps or 30.0
        self.snap_gap = max(SNAP_MIN_GAP, (total_frames or 0) // SNAP_MAX_PER_TYPE)
        os.makedirs(SNAPSHOT_DIR, exist_ok=True)
        self.events = []
        self.snap_count = Counter()
        self.snap_last = {}
        self.frame_stats = {"n_small": 0, "n_blur": 0, "n_pairs": 0}
        self.n_instances = 0
        self.frames_seen = Counter()
        self.raw_ids_seen = set()
        self.n_class_flip = 0
        self.n_small_instances = 0
        self.small_ids = set()
        self.conf_small, self.conf_large = [], []
        self.active_pairs = set()
        self.all_pairs = set()
        self.last_raw = {}
        self.last_seen = {}
        self.recover_gaps = []
        self.prev_center = {}
        self.blur_active = set()
        self.blur_ids = set()
        self.n_blur_instances = 0
        self.sharp_all, self.speed_all = [], []

    def _log(self, frame_idx, ftype, sid, cls, detail):
        self.events.append({
            "frame": frame_idx,
            "time_s": round(frame_idx / self.fps, 2),
            "type": ftype,
            "stable_id": sid,
            "class": CLASS_SHORT.get(cls, cls),
            "detail": detail,
        })

    def idle(self):
        self.active_pairs = set()
        self.blur_active = set()
        self.frame_stats = {"n_small": 0, "n_blur": 0, "n_pairs": 0}

    def update(self, frame, tracked, raw_ids, raw_class_ids, voted_classes, frame_idx):
        pending = defaultdict(list)
        n = len(tracked)
        xyxy = tracked.xyxy
        sids = [int(s) for s in tracked.tracker_id]
        conf = tracked.confidence if tracked.confidence is not None else np.full(n, np.nan)
        centers = tracked.get_anchors_coordinates(sv.Position.CENTER)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_h, img_w = gray.shape
        n_small = n_blur = 0
        blur_now = set()

        for k in range(n):
            sid, cls, box = sids[k], voted_classes[k], xyxy[k]
            raw = int(raw_ids[k])
            w, h = float(box[2] - box[0]), float(box[3] - box[1])
            c = float(conf[k])

            self.n_instances += 1
            self.frames_seen[sid] += 1
            self.raw_ids_seen.add(raw)
            if raw_class_ids[k] != cls:
                self.n_class_flip += 1

            if w * h < SMALL_AREA:
                n_small += 1
                self.n_small_instances += 1
                if not math.isnan(c):
                    self.conf_small.append(c)
                pending["small_object"].append(box)
                if sid not in self.small_ids:
                    self.small_ids.add(sid)
                    self._log(frame_idx, "small_object", sid, cls,
                              f"box {w:.0f}x{h:.0f}px, conf {c:.2f}")
            elif not math.isnan(c):
                self.conf_large.append(c)

            prev_raw = self.last_raw.get(sid)
            if prev_raw is not None and prev_raw != raw:
                lost = frame_idx - self.last_seen[sid] - 1
                self.recover_gaps.append(lost)
                pending["id_recovered"].append(box)
                self._log(frame_idx, "id_recovered", sid, cls,
                          f"raw {prev_raw} -> {raw}, mất {lost} frame")
            self.last_raw[sid] = raw
            self.last_seen[sid] = frame_idx

            cx, cy = centers[k]
            prev = self.prev_center.get(sid)
            speed = None
            if prev is not None and frame_idx > prev[2]:
                speed = math.hypot(cx - prev[0], cy - prev[1]) / (frame_idx - prev[2])
            self.prev_center[sid] = (float(cx), float(cy), frame_idx)

            x1, y1 = max(int(box[0]), 0), max(int(box[1]), 0)
            x2, y2 = min(int(box[2]), img_w), min(int(box[3]), img_h)
            short_side = min(x2 - x1, y2 - y1)
            if short_side >= MIN_BLUR_SIDE:
                interp = cv2.INTER_AREA if short_side >= CROP_NORM else cv2.INTER_LINEAR
                crop = cv2.resize(gray[y1:y2, x1:x2], (CROP_NORM, CROP_NORM), interpolation=interp)
                sharp = float(cv2.Laplacian(crop, cv2.CV_64F).var())
                self.sharp_all.append(sharp)
                if speed is not None:
                    self.speed_all.append(speed)
                    if sharp < BLUR_LAP_MAX and speed >= BLUR_MIN_SPEED:
                        n_blur += 1
                        self.n_blur_instances += 1
                        self.blur_ids.add(sid)
                        blur_now.add(sid)
                        pending["motion_blur"].append(box)
                        if sid not in self.blur_active:
                            self._log(frame_idx, "motion_blur", sid, cls,
                                      f"độ nét {sharp:.0f}, tốc độ {speed:.1f}px/frame")
        self.blur_active = blur_now

        cur_pairs = set()
        if n >= 2:
            iou = sv.box_iou_batch(xyxy, xyxy)
            ii, jj = np.where(np.triu(iou, 1) > OCC_IOU)
            for i, j in zip(ii, jj):
                pair = (min(sids[i], sids[j]), max(sids[i], sids[j]))
                cur_pairs.add(pair)
                if pair not in self.active_pairs:
                    self.all_pairs.add(pair)
                    pending["occlusion"].extend([xyxy[i], xyxy[j]])
                    self._log(frame_idx, "occlusion", f"{pair[0]}&{pair[1]}", "",
                              f"IoU {iou[i, j]:.2f}")
        self.active_pairs = cur_pairs

        self.frame_stats = {"n_small": n_small, "n_blur": n_blur, "n_pairs": len(cur_pairs)}
        return pending

    def maybe_snapshot(self, frame, pending, frame_idx):
        for ftype, boxes in pending.items():
            if not boxes or self.snap_count[ftype] >= SNAP_MAX_PER_TYPE:
                continue
            if frame_idx - self.snap_last.get(ftype, -10 ** 9) < self.snap_gap:
                continue
            img = frame.copy()
            for x1, y1, x2, y2 in boxes:
                cv2.rectangle(img, (int(x1) - 3, int(y1) - 3), (int(x2) + 3, int(y2) + 3),
                              (255, 0, 255), 2)
            cv2.putText(img, f"{ftype} | frame {frame_idx} ({frame_idx / self.fps:.1f}s)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
            cv2.imwrite(os.path.join(SNAPSHOT_DIR, f"{ftype}_f{frame_idx:06d}.jpg"), img)
            self.snap_count[ftype] += 1
            self.snap_last[ftype] = frame_idx

    def summary_lines(self):
        ev = Counter(e["type"] for e in self.events)
        n_stable, n_raw = len(self.frames_seen), len(self.raw_ids_seen)
        short = sum(1 for c in self.frames_seen.values() if c <= SHORT_TRACK_FRAMES)
        pct_small = 100.0 * self.n_small_instances / max(self.n_instances, 1)
        L = ["===== FAILURE CASES (tự động gợi ý, cần đối chiếu bằng mắt) =====",
             f"Tổng lượt box được theo dõi: {self.n_instances} | stable ID: {n_stable} | "
             f"raw ID của ByteTrack: {n_raw} (tỉ lệ raw/stable = {n_raw / max(n_stable, 1):.2f})",
             "",
             f"[1] Vật thể nhỏ (diện tích < {SMALL_AREA}px²): {self.n_small_instances} lượt "
             f"({pct_small:.1f}%), {len(self.small_ids)} ID",
             f"    conf trung bình: nhỏ {_mean(self.conf_small):.2f} vs lớn {_mean(self.conf_large):.2f}",
             f"    ID sống ≤ {SHORT_TRACK_FRAMES} frame: {short}/{n_stable} "
             f"(detect chập chờn; xe mới vào/ra khung hình cũng bị tính)",
             "    Lưu ý: xe ở xa bị bỏ sót hoàn toàn thì code không thấy được -> xem snapshot/video để đánh giá",
             "",
             f"[2] Che khuất: {len(self.all_pairs)} cặp xe khác nhau từng chồng lấn (IoU > {OCC_IOU}), "
             f"{ev['occlusion']} lần bắt đầu chồng lấn",
             f"    ByteTrack đổi raw ID phải nối lại: {ev['id_recovered']} lần"
             + (f" (số frame mất trung bình {_mean(self.recover_gaps):.1f}, "
                f"tối đa {max(self.recover_gaps)})" if self.recover_gaps else ""),
             "",
             f"[3] Motion blur (Laplacian var < {BLUR_LAP_MAX} và tốc độ ≥ {BLUR_MIN_SPEED}px/frame): "
             f"{self.n_blur_instances} lượt, {len(self.blur_ids)} ID",
             f"    độ nét (Laplacian var) p10/p50/p90: {_pct(self.sharp_all, 10):.0f} / "
             f"{_pct(self.sharp_all, 50):.0f} / {_pct(self.sharp_all, 90):.0f}"
             f"  -> chỉnh BLUR_LAP_MAX cho hợp video nếu cần",
             f"    tốc độ (px/frame) p50/p90: {_pct(self.speed_all, 50):.1f} / {_pct(self.speed_all, 90):.1f}"
             f"  -> chỉnh BLUR_MIN_SPEED nếu cần",
             "",
             f"[+] Loại xe của detector thô khác nhãn sau bỏ phiếu: {self.n_class_flip} lượt "
             f"(detector nhầm car/truck/bus...)",
             f"Chi tiết từng sự kiện: failure_events.csv | ảnh minh họa: {SNAPSHOT_DIR}"]
        return L

    def save_events(self, path):
        write_csv(path, self.events, ["frame", "time_s", "type", "stable_id", "class", "detail"])


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class VehicleCropSaver:
    def __init__(self):
        self.best = {}

    def update(self, frame, tracked, weights, frame_idx):
        img_h, img_w = frame.shape[:2]
        for box, sid, cls, wgt in zip(tracked.xyxy, tracked.tracker_id, tracked.class_id, weights):
            sid = int(sid)
            entry = self.best.get(sid)
            if entry is None:
                entry = self.best[sid] = {"score": -1.0, "img": None, "cls": int(cls),
                                          "frame": frame_idx, "n": 0}
            entry["n"] += 1
            entry["cls"] = int(cls)

            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            score = w * h * float(wgt)
            if x1 <= 2 or y1 <= 2 or x2 >= img_w - 2 or y2 >= img_h - 2:
                score *= 0.3
            if score <= entry["score"]:
                continue

            px, py = w * CROP_PAD, h * CROP_PAD
            cx1, cy1 = max(int(x1 - px), 0), max(int(y1 - py), 0)
            cx2, cy2 = min(int(x2 + px), img_w), min(int(y2 + py), img_h)
            if cx2 - cx1 < 4 or cy2 - cy1 < 4:
                continue
            crop = frame[cy1:cy2, cx1:cx2]
            scale = CROP_MAX_SIDE / max(crop.shape[:2])
            if scale < 1.0:
                crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            entry["img"] = crop.copy()
            entry["score"] = score
            entry["frame"] = frame_idx

    def save_all(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        for name in os.listdir(out_dir):
            if name.lower().endswith(".jpg"):
                os.remove(os.path.join(out_dir, name))

        items = sorted(
            ((sid, e) for sid, e in self.best.items()
             if e["img"] is not None and e["n"] >= CROP_MIN_FRAMES),
            key=lambda x: x[0],
        )
        for sid, e in items:
            cname = CLASS_SHORT.get(e["cls"], str(e["cls"]))
            cv2.imwrite(os.path.join(out_dir, f"id{sid:04d}_{cname}_f{e['frame']:06d}.jpg"), e["img"])

        for page, start in enumerate(range(0, len(items), CONTACT_PER_SHEET), start=1):
            self._save_sheet(items[start:start + CONTACT_PER_SHEET],
                             os.path.join(out_dir, f"contact_sheet_{page:02d}.jpg"))
        return len(items)

    @staticmethod
    def _save_sheet(items, path):
        cell = CONTACT_CELL
        n_rows = math.ceil(len(items) / CONTACT_COLS)
        sheet = np.full((n_rows * cell, CONTACT_COLS * cell, 3), 30, dtype=np.uint8)
        for k, (sid, e) in enumerate(items):
            img = e["img"]
            s = (cell - 2) / max(img.shape[:2])
            interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR
            thumb = cv2.resize(img, None, fx=s, fy=s, interpolation=interp)
            th, tw = thumb.shape[:2]
            r, c = divmod(k, CONTACT_COLS)
            y0, x0 = r * cell, c * cell
            sheet[y0:y0 + th, x0:x0 + tw] = thumb
            label = f"{CLASS_SHORT.get(e['cls'], e['cls'])} {sid}"
            cv2.putText(sheet, label, (x0 + 3, y0 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3)
            cv2.putText(sheet, label, (x0 + 3, y0 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.imwrite(path, sheet)


def build_fps_report(rows, wall_s, video_info):
    n_total = len(rows)
    if n_total == 0:
        return ["===== FPS =====", "Không có frame nào được xử lý."]
    body = rows[WARMUP_FRAMES:] if n_total > WARMUP_FRAMES * 2 else rows

    def col(key):
        return np.array([r[key] for r in body], dtype=float)

    track = col("track_ms")
    yolo_pre, yolo_inf, yolo_post = col("yolo_pre_ms"), col("yolo_inf_ms"), col("yolo_post_ms")
    core = col("read_ms") + track + col("ids_ms") + col("draw_ms")
    tracker_ms = track - (yolo_pre + yolo_inf + yolo_post)
    pipe_fps = 1000.0 / core.mean()
    src_fps = video_info.fps or 30.0

    return [
        "===== FPS =====",
        f"Video: {video_info.width}x{video_info.height}, {src_fps:.1f} fps gốc, {n_total} frame "
        f"(bỏ {n_total - len(body)} frame warm-up khi tính trung bình)",
        f"Cấu hình: {PT_NAME} -> OpenVINO, imgsz={IMGSZ}, conf={CONF}, iou={IOU}, tracker={TRACKER_CFG}",
        f"Detect + track: {track.mean():.1f} ms/frame -> {1000.0 / track.mean():.2f} FPS "
        f"(p50 {_pct(track, 50):.1f} ms, p95 {_pct(track, 95):.1f} ms)",
        f"  - YOLO pre / inference / post: {yolo_pre.mean():.1f} / {yolo_inf.mean():.1f} / {yolo_post.mean():.1f} ms",
        f"  - Tracker + overhead khác: {tracker_ms.mean():.1f} ms",
        f"Gộp ID + smoother: {col('ids_ms').mean():.1f} ms | Vẽ + ghi video: {col('draw_ms').mean():.1f} ms "
        f"| Đọc frame: {col('read_ms').mean():.1f} ms",
        f"Toàn pipeline (không tính phân tích failure): {pipe_fps:.2f} FPS "
        f"= {pipe_fps / src_fps:.2f}x tốc độ video gốc",
        f"End-to-end thực tế (wall-clock, gồm cả phân tích): {n_total / wall_s:.2f} FPS",
        "Lưu ý: FPS phụ thuộc máy/CPU/IMGSZ; ghi rõ cấu hình khi báo cáo.",
    ]


def main():
    model = YOLO(ensure_model(), task="detect")
    class_names = [model.names.get(cid, str(cid)) for cid in VEHICLE_CLASSES]
    print(f"Theo dõi các class {VEHICLE_CLASSES} = {class_names}")

    os.makedirs(DEST_DIR, exist_ok=True)
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    video_info = sv.VideoInfo.from_video_path(SOURCE_PATH)
    frames = sv.get_video_frames_generator(SOURCE_PATH)

    palette = sv.ColorPalette(
        [sv.Color.from_hex(CLASS_COLORS.get(i, "#95A5A6")) for i in range(8)]
    )
    box_annotator = sv.BoxAnnotator(
        color=palette,
        thickness=BOX_THICKNESS,
        color_lookup=sv.ColorLookup.CLASS,
    )
    label_annotator = sv.LabelAnnotator(
        color=palette,
        text_position=sv.Position.TOP_CENTER,
        text_scale=LABEL_SCALE,
        text_thickness=LABEL_THICKNESS,
        text_padding=LABEL_PADDING,
        text_color=sv.Color.WHITE,
        color_lookup=sv.ColorLookup.CLASS,
    )
    smoother = sv.DetectionsSmoother(length=SMOOTH_LEN) if SMOOTH_LEN > 1 else None
    id_manager = StableIDManager(
        MAX_LOST_FRAMES, BASE_GATE, GATE_GROWTH, GATE_SIZE_RATIO, VOTE_SWITCH_MARGIN
    )
    analyzer = (
        FailureAnalyzer(video_info.fps, video_info.total_frames)
        if ENABLE_FAILURE_ANALYSIS else None
    )
    crop_saver = VehicleCropSaver() if SAVE_VEHICLE_CROPS else None
    rows = []

    print(f"Bắt đầu tracking: {SOURCE_PATH}")

    t_begin = t_prev_end = time.perf_counter()
    with sv.VideoSink(OUTPUT_PATH, video_info) as sink:
        for frame_idx, frame in enumerate(frames, start=1):
            t0 = time.perf_counter()
            read_s = t0 - t_prev_end

            results = model.track(
                frame,
                persist=True,
                tracker=TRACKER_CFG,
                classes=VEHICLE_CLASSES,
                conf=CONF,
                iou=IOU,
                imgsz=IMGSZ,
                agnostic_nms=AGNOSTIC_NMS,
                verbose=False,
            )[0]
            t1 = time.perf_counter()
            yolo_ms = results.speed
            detections = sv.Detections.from_ultralytics(results)

            tracked = sv.Detections.empty()
            if detections.tracker_id is not None and len(detections) > 0:
                detections = detections[np.isin(detections.class_id, VEHICLE_CLASSES)]
                if len(detections) > 0:
                    tracked = (
                        smoother.update_with_detections(detections)
                        if smoother is not None else detections
                    )

            has_tracks = tracked.tracker_id is not None and len(tracked) > 0
            raw_ids = tracked.tracker_id.tolist() if has_tracks else []
            class_ids = tracked.class_id.tolist() if has_tracks else []
            centers = (
                tracked.get_anchors_coordinates(sv.Position.CENTER) if has_tracks else []
            )
            sizes = (
                np.maximum(tracked.xyxy[:, 2] - tracked.xyxy[:, 0],
                           tracked.xyxy[:, 3] - tracked.xyxy[:, 1]).tolist()
                if has_tracks else []
            )
            weights = vote_weights(tracked, VOTE_MIN_CONF).tolist() if has_tracks else []

            stable_ids, voted_classes = id_manager.update(
                raw_ids, centers, class_ids, sizes, weights, frame_idx
            )
            t2 = time.perf_counter()

            diag_s = 0.0
            draw_s = 0.0
            pending = {}
            if has_tracks:
                tracked.tracker_id = np.array(stable_ids)
                tracked.class_id = np.array(voted_classes)
                if SHOW_CLASS:
                    labels = [
                        f"{CLASS_SHORT.get(c, c)} {tid}"
                        for c, tid in zip(voted_classes, stable_ids)
                    ]
                else:
                    labels = [str(tid) for tid in stable_ids]

                if analyzer is not None:
                    td = time.perf_counter()
                    pending = analyzer.update(
                        frame, tracked, raw_ids, class_ids, voted_classes, frame_idx
                    )
                    diag_s += time.perf_counter() - td

                if crop_saver is not None:
                    td = time.perf_counter()
                    crop_saver.update(frame, tracked, weights, frame_idx)
                    diag_s += time.perf_counter() - td

                td = time.perf_counter()
                frame = box_annotator.annotate(scene=frame, detections=tracked)
                frame = label_annotator.annotate(scene=frame, detections=tracked, labels=labels)
                draw_s += time.perf_counter() - td
            elif analyzer is not None:
                analyzer.idle()

            td = time.perf_counter()
            sink.write_frame(frame)
            draw_s += time.perf_counter() - td

            if analyzer is not None and pending:
                td = time.perf_counter()
                analyzer.maybe_snapshot(frame, pending, frame_idx)
                diag_s += time.perf_counter() - td

            fs = analyzer.frame_stats if analyzer is not None else {"n_small": 0, "n_blur": 0, "n_pairs": 0}
            rows.append({
                "frame": frame_idx,
                "read_ms": round(read_s * 1000, 3),
                "track_ms": round((t1 - t0) * 1000, 3),
                "yolo_pre_ms": round(yolo_ms.get("preprocess", 0.0), 3),
                "yolo_inf_ms": round(yolo_ms.get("inference", 0.0), 3),
                "yolo_post_ms": round(yolo_ms.get("postprocess", 0.0), 3),
                "ids_ms": round((t2 - t1) * 1000, 3),
                "draw_ms": round(draw_s * 1000, 3),
                "diag_ms": round(diag_s * 1000, 3),
                "n_tracks": len(tracked) if has_tracks else 0,
                "n_small": fs["n_small"],
                "n_blur": fs["n_blur"],
                "n_occluded_pairs": fs["n_pairs"],
            })
            t_prev_end = time.perf_counter()

            if frame_idx % 100 == 0:
                print(f"  {frame_idx}/{video_info.total_frames} frames")

    wall_s = time.perf_counter() - t_begin

    n_crops = crop_saver.save_all(CROP_DIR) if crop_saver is not None else 0

    report = build_fps_report(rows, wall_s, video_info)
    if analyzer is not None:
        report += [""] + analyzer.summary_lines()
        analyzer.save_events(os.path.join(ANALYSIS_DIR, "failure_events.csv"))
    if rows:
        write_csv(os.path.join(ANALYSIS_DIR, "frame_metrics.csv"), rows, list(rows[0].keys()))
    with open(os.path.join(ANALYSIS_DIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")

    print()
    print("\n".join(report))
    print(f"\nXử lý xong! Video tracking lưu tại: {OUTPUT_PATH}")
    print(f"Số liệu FPS + failure cases lưu tại: {ANALYSIS_DIR}")
    if crop_saver is not None:
        print(f"Ảnh {n_crops} phương tiện (mỗi ID 1 ảnh + contact_sheet) lưu tại: {CROP_DIR}")


if __name__ == "__main__":
    main()
