#!/usr/bin/env python3
"""
Posture Guard — real-time tech-neck / posture coach for your Mac webcam.

- MediaPipe Pose Landmarker (Tasks API) tracks head + shoulders
- Personal calibration → distance-invariant metrics (all normalized by shoulder width)
- Multi-second confirmation before an alert fires (no flicker alerts)
- Live dashboard at http://localhost:8765 (video stream + score + day timeline)
- CSV + daily-JSON logging for a full-day posture score

Run:  python3 app.py   (then open http://localhost:8765)
"""

import atexit
import csv
import json
import math
import os
import threading
import time
from collections import deque
from datetime import datetime, date

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision
from flask import Flask, Response, jsonify, request, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "pose_landmarker_lite.task")
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

STREAM_WIDTH = 960          # width of the JPEG stream sent to the browser
TARGET_FPS = 24

# ---------------------------------------------------------------- settings

DEFAULT_SETTINGS = {
    "slouch_sensitivity": 0.15,   # allowed % drop of head height vs calibrated baseline
    "tilt_thresh_deg": 12.0,      # head roll tolerance (degrees)
    "lean_thresh_deg": 15.0,      # shoulder line tilt tolerance (degrees)
    "alert_delay_s": 3.0,         # bad posture must persist this long before an alert
    "recover_delay_s": 2.0,       # good posture must persist this long to clear an alert
    "sound": True,
}
SETTINGS_PATH = os.path.join(LOG_DIR, "settings.json")

def load_settings():
    s = dict(DEFAULT_SETTINGS)
    try:
        with open(SETTINGS_PATH) as f:
            s.update(json.load(f))
    except Exception:
        pass
    return s

def save_settings(s):
    with open(SETTINGS_PATH, "w") as f:
        json.dump(s, f, indent=2)

SETTINGS = load_settings()

# ---------------------------------------------------------------- shared state

class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None                # latest annotated JPEG frame (np array)
        self.status = "BOOTING"          # BOOTING|NOCAM|NODETECT|CALIBRATING|GOOD|WARN|BAD
        self.reason = ""
        self.metrics = {}
        self.session_start = time.time()
        self.good_s = 0.0
        self.bad_s = 0.0
        self.alerts = 0
        self.current_streak_bad = 0.0    # seconds of continuous bad posture
        self.baseline = None             # dict from calibration
        self.calib_samples = None        # deque during calibration
        self.calib_progress = 0.0
        self.today = self._load_today()
        self.events = deque(maxlen=50)   # alert events (UI feed)

    @staticmethod
    def _today_path():
        return os.path.join(LOG_DIR, f"day-{date.today().isoformat()}.json")

    def _load_today(self):
        try:
            with open(self._today_path()) as f:
                return json.load(f)
        except Exception:
            return {"date": date.today().isoformat(), "good_s": 0.0, "bad_s": 0.0,
                    "alerts": 0, "buckets": {}}

    def flush_today(self):
        try:
            with open(self._today_path(), "w") as f:
                json.dump(self.today, f)
        except Exception:
            pass

STATE = State()

# ---------------------------------------------------------------- pose model

landmarker = None

def create_landmarker():
    global landmarker
    opts = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_pose_world_landmarks=True,
    )
    landmarker = vision.PoseLandmarker.create_from_options(opts)

# landmark indices (MediaPipe Pose, 33 points)
NOSE, LEYE_O, REYE_O, LEAR, REAR, LSH, RSH = 0, 3, 6, 7, 8, 11, 12
TORSO_PAIRS = [(LSH, RSH)]
HEAD_PAIRS = [(LEAR, REAR)]

def ema(prev, new, alpha=0.35):
    return new if prev is None else prev + alpha * (new - prev)

class MetricSmoother:
    """Exponential smoothing so the UI and logic don't jitter frame to frame."""
    def __init__(self):
        self.d = {}
    def push(self, name, val):
        self.d[name] = ema(self.d.get(name), val)
        return self.d[name]

SMOOTH = MetricSmoother()

def compute_metrics(lms, W, H):
    """Distance-invariant posture metrics, all normalized by shoulder width."""
    p = lms
    def pt(i):
        return p[i].x * W, p[i].y * H

    ls, rs = np.array(pt(LSH)), np.array(pt(RSH))
    le, re = np.array(pt(LEYE_O)), np.array(pt(REYE_O))
    lear, rear = np.array(pt(LEAR)), np.array(pt(REAR))
    nose = np.array(pt(NOSE))

    sh_mid = (ls + rs) / 2
    eye_mid = (le + re) / 2
    ear_mid = (lear + rear) / 2
    sh_w = float(np.linalg.norm(rs - ls))
    if sh_w < 1e-6:
        return None

    vis = min(p[LSH].visibility, p[RSH].visibility, p[LEAR].visibility, p[REAR].visibility)
    if vis < 0.5:
        return None

    # 1) head height: vertical gap ears→shoulders, normalized by shoulder width.
    #    Slouching / tech-neck sinks the head → ratio shrinks.
    drop = float((sh_mid[1] - ear_mid[1]) / sh_w)
    # 2) head roll: eye line angle vs horizontal
    tilt = abs(float(math.degrees(math.atan2(re[1] - le[1], max(1e-6, abs(re[0] - le[0]))))))
    # 3) shoulder roll (sideways lean / collapsed posture)
    lean = float(abs(math.degrees(math.atan2(rs[1] - ls[1], abs(rs[0] - ls[0])))))
    # 4) lateral head offset: nose x vs shoulder midpoint x, normalized
    yaw = float((nose[0] - (ear_mid[0])) / sh_w)

    return {
        "drop": SMOOTH.push("drop", drop),
        "tilt": SMOOTH.push("tilt", tilt),
        "lean": SMOOTH.push("lean", lean),
        "yaw": SMOOTH.push("yaw", yaw),
        "sh_w": sh_w,
    }

# ---------------------------------------------------------------- scoring / voting lock

def evaluate(m, baseline, s):
    """Return (status, reason). Same philosophy as the helmet system: a verdict is
    only 'locked' after it persists for several seconds."""
    if baseline is None:
        return "WARN", "Run calibration for a personalized baseline"
    drop_ref = baseline["drop"]
    limit = drop_ref * (1.0 - s["slouch_sensitivity"])
    problems = []
    if m["drop"] < limit:
        pct = max(0.0, (drop_ref - m["drop"]) / drop_ref * 100)
        problems.append(f"Head sunk {pct:.0f}% (tech-neck)")
    if m["tilt"] > s["tilt_thresh_deg"]:
        problems.append(f"Head tilt {m['tilt']:.0f}°")
    if m["lean"] > s["lean_thresh_deg"]:
        problems.append(f"Shoulders tilted {m['lean']:.0f}°")
    if problems:
        return "BAD", " · ".join(problems[:2])
    return "GOOD", "Upright — nice"

# ---------------------------------------------------------------- camera worker

def worker():
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        with STATE.lock:
            STATE.status = "NOCAM"
            STATE.reason = "Camera blocked. Allow camera access for your Terminal app."
        return
    create_landmarker()
    last_ts_ms = 0
    frame_interval = 1.0 / TARGET_FPS
    last_tick = time.time()
    log_accum = {"good": 0.0, "bad": 0.0}
    last_log_t = time.time()

    try:
        while True:
            loop_t = time.time()
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.2)
                continue

            rgb = cv2.cvtColor(cv2.resize(frame, (STREAM_WIDTH, int(STREAM_WIDTH * frame.shape[0] / frame.shape[1]))),
                               cv2.COLOR_BGR2RGB)
            H, W = rgb.shape[:2]
            ts_ms = int(time.monotonic() * 1000)
            if ts_ms <= last_ts_ms:
                ts_ms = last_ts_ms + 1
            last_ts_ms = ts_ms

            result = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)), ts_ms)

            now = time.time()
            dt = now - last_tick
            last_tick = now

            if result.pose_landmarks:
                m = compute_metrics(result.pose_landmarks[0], W, H)
            else:
                m = None

            with STATE.lock:
                s = dict(SETTINGS)
                if STATE.calib_samples is not None:
                    # collecting calibration
                    if m:
                        STATE.calib_samples.append(m)
                        STATE.calib_progress = min(1.0, len(STATE.calib_samples) / 75.0)
                        if len(STATE.calib_samples) >= 75:
                            base = {"drop": float(np.mean([x["drop"] for x in STATE.calib_samples])),
                                    "tilt": float(np.mean([x["tilt"] for x in STATE.calib_samples])),
                                    "lean": float(np.mean([x["lean"] for x in STATE.calib_samples]))}
                            STATE.baseline = base
                            STATE.calib_samples = None
                            STATE.status = "GOOD"; STATE.reason = "Calibrated"
                            STATE.events.appendleft({"t": time.time(), "type": "info",
                                                     "msg": "Calibration complete — baseline saved"})
                    else:
                        pass
                    STATE.calib_progress = STATE.calib_progress if STATE.calib_samples else 1.0
                    STATE.status = "CALIBRATING" if STATE.calib_samples else STATE.status
                    status, reason = STATE.status, STATE.reason
                elif m is None:
                    STATE.status = "NODETECT"
                    STATE.reason = "No person detected — sit in frame"
                    STATE.current_streak_bad = 0.0
                    status, reason = STATE.status, STATE.reason
                else:
                    STATE.metrics = {k: round(v, 4) for k, v in m.items() if k != "sh_w"}
                    status, reason = evaluate(m, STATE.baseline, s)
                    STATE.status, STATE.reason = status, reason

                    if status == "BAD":
                        STATE.bad_s += dt
                        STATE.current_streak_bad += dt
                        STATE.today["bad_s"] += dt
                        log_accum["bad"] += dt
                        if STATE.current_streak_bad >= s["alert_delay_s"]:
                            STATE.current_streak_bad = 0.0
                            STATE.alerts += 1
                            STATE.today["alerts"] += 1
                            STATE.events.appendleft({"t": now, "type": "alert", "msg": reason})
                            alert_event.set()
                    else:
                        STATE.good_s += dt
                        STATE.today["good_s"] += dt
                        log_accum["good"] += dt
                        STATE.current_streak_bad = 0.0

                # periodic logging: 5-second aggregates to CSV + day JSON bucket
                if now - last_log_t >= 5.0:
                    last_log_t = now
                    total = log_accum["good"] + log_accum["bad"]
                    if total > 0:
                        with open(csv_path(), "a", newline="") as f:
                            csv.writer(f).writerow(
                                [datetime.now().isoformat(timespec="seconds"),
                                 round(log_accum["good"], 1), round(log_accum["bad"], 1),
                                 STATE.status, STATE.reason])
                        bucket = str(datetime.now().minute // 5 * 5).zfill(2)
                        b = STATE.today["buckets"].setdefault(f"{datetime.now().hour:02d}:{bucket}", [0, 0])
                        b[0] += log_accum["good"]; b[1] += log_accum["bad"]
                        log_accum = {"good": 0.0, "bad": 0.0}
                        STATE.flush_today()

                # annotate frame (drawn under lock so it always matches state)
                annotated = draw_overlay(rgb.copy(), result.pose_landmarks[0] if result.pose_landmarks else None,
                                         STATE.status, STATE.reason, STATE.calib_progress)
                STATE.frame = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 82])[1].tobytes()
    finally:
        cap.release()

def csv_path():
    return os.path.join(LOG_DIR, f"posture-{date.today().isoformat()}.csv")

alert_event = threading.Event()

def draw_overlay(img, lms, status, reason, calib_progress):
    colors = {"GOOD": (80, 220, 100), "BAD": (90, 90, 255), "WARN": (0, 190, 255),
              "NODETECT": (150, 150, 150), "CALIBRATING": (255, 180, 60), "NOCAM": (90, 90, 255)}
    color = colors.get(status, (255, 255, 255))
    H, W = img.shape[:2]

    if lms is not None:
        def px(i):
            return int(lms[i].x * W), int(lms[i].y * H)
        for a, b in TORSO_PAIRS + HEAD_PAIRS:
            cv2.line(img, px(a), px(b), color, 3, cv2.LINE_AA)
        # ear → shoulder vertical guides (the "tech-neck" measure)
        for ear, sh in [(LEAR, LSH), (REAR, RSH)]:
            e, s_ = px(ear), px(sh)
            cv2.circle(img, e, 4, color, -1, cv2.LINE_AA)
            cv2.circle(img, s_, 5, color, -1, cv2.LINE_AA)
            cv2.line(img, e, (e[0], s_[1]), color, 1, cv2.LINE_AA)
        nose = px(NOSE)
        cv2.circle(img, nose, 4, color, -1, cv2.LINE_AA)

    # status pill
    label = {"GOOD": "GOOD POSTURE", "BAD": "SLOUCHING", "WARN": "CALIBRATE ME",
             "NODETECT": "NO PERSON", "CALIBRATING": "CALIBRATING…", "NOCAM": "NO CAMERA"}[status]
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(img, (10, 10), (26 + tw, 34 + th), (20, 20, 20), -1)
    cv2.rectangle(img, (10, 10), (26 + tw, 34 + th), color, 2)
    cv2.putText(img, label, (18, 28 + th // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    if reason:
        cv2.putText(img, reason[:64], (18, 52 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
    if status == "CALIBRATING":
        cv2.rectangle(img, (10, 70), (10 + int(calib_progress * (W - 20)), 78), (255, 180, 60), -1)
    return img

# ---------------------------------------------------------------- flask

app = Flask(__name__, static_folder="static")

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

def gen_mjpeg():
    boundary = b"--frame\r\n"
    while True:
        with STATE.lock:
            f = STATE.frame
        if f is None:
            time.sleep(0.1); continue
        yield boundary + b"Content-Type: image/jpeg\r\n\r\n" + f + b"\r\n"
        time.sleep(0.033)

@app.route("/video")
def video():
    return Response(gen_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/state")
def api_state():
    with STATE.lock:
        t = STATE.today
        return jsonify({
            "status": STATE.status, "reason": STATE.reason, "metrics": STATE.metrics,
            "session_s": round(time.time() - STATE.session_start),
            "good_s": round(STATE.good_s), "bad_s": round(STATE.bad_s),
            "alerts": STATE.alerts,
            "bad_streak": round(STATE.current_streak_bad, 1),
            "calibrating": STATE.calib_samples is not None,
            "calib_progress": round(STATE.calib_progress, 2),
            "calibrated": STATE.baseline is not None,
            "today": {"good_s": round(t["good_s"]), "bad_s": round(t["bad_s"]),
                      "alerts": t["alerts"], "buckets": t["buckets"]},
            "events": list(STATE.events)[:12],
            "settings": SETTINGS,
        })

@app.route("/api/calibrate", methods=["POST"])
def api_calibrate():
    with STATE.lock:
        if STATE.status in ("NOCAM", "BOOTING"):
            return jsonify(ok=False, error="camera not available"), 409
        STATE.calib_samples = deque(maxlen=75)
        STATE.calib_progress = 0.0
        STATE.status = "CALIBRATING"; STATE.reason = "Sit upright and stay still"
    return jsonify(ok=True)

@app.route("/api/reset", methods=["POST"])
def api_reset():
    with STATE.lock:
        STATE.session_start = time.time()
        STATE.good_s = STATE.bad_s = STATE.alerts = 0.0
        STATE.current_streak_bad = 0.0
        STATE.events.clear()
        STATE.events.append({"t": time.time(), "type": "info", "msg": "Session stats reset"})
    return jsonify(ok=True)

@app.route("/api/settings", methods=["POST"])
def api_settings():
    global SETTINGS
    try:
        body = request.get_json(force=True) or {}
    except Exception:
        body = {}
    with STATE.lock:
        for k in DEFAULT_SETTINGS:
            if k in body:
                SETTINGS[k] = body[k] if not isinstance(DEFAULT_SETTINGS[k], bool) else bool(body[k])
        save_settings(SETTINGS)
    return jsonify(ok=True, settings=SETTINGS)

# ---------------------------------------------------------------- main

def main():
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    atexit.register(STATE.flush_today)
    print("\n  🪑 Posture Guard running → http://localhost:8765\n")
    app.run(host="127.0.0.1", port=8765, threaded=True, use_reloader=False)

if __name__ == "__main__":
    main()
