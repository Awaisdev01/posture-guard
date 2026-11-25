#!/usr/bin/env python3
"""
Posture Guard for macOS — native desktop app (PySide6).

- Live webcam posture coaching with MediaPipe Pose Landmarker
- Personal calibration → distance-invariant metrics (÷ shoulder width)
- Vote-locked verdicts: bad posture must persist `alert_delay_s` before alerting
- On alert: in-app banner + macOS notification + system sound
- Menu-bar (tray) icon with live status color
- CSV + daily JSON logs, persisted across restarts

Run:  python3 app_mac.py
"""

import json
import math
import os
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import date, datetime

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

from PySide6.QtCore import Qt, QTimer, QThread, Signal, QSize, QPointF, QRectF
from PySide6.QtGui import (QIcon, QPixmap, QPainter, QColor, QFont, QImage, QPen,
                           QBrush, QAction, QGuiApplication, QFontMetrics)
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout,
                               QHBoxLayout, QGridLayout, QPushButton, QSlider, QCheckBox,
                               QSystemTrayIcon, QMenu, QListWidget, QListWidgetItem,
                               QFrame, QSizePolicy, QGraphicsDropShadowEffect)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "pose_landmarker_full.task")
FACE_MODEL_PATH = os.path.join(BASE_DIR, "models", "face_landmarker.task")
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

TARGET_FPS = 24
OVERLAY_W = 960

GOOD = "#3ddc84"; BAD = "#ff5d5d"; WARN = "#ffb454"; MUT = "#8a93a6"
BG = "#0b0e14"; PANEL = "#131722"; PANEL2 = "#181e2c"; BORDER = "#232a3a"
TEXT = "#e8ecf4"; MUTED = "#8a93a6"; ACCENT = "#6c8cff"

DEFAULT_SETTINGS = {
    "slouch_sensitivity": 0.18,
    "tilt_thresh_deg": 14.0,
    "lean_thresh_deg": 18.0,
    "alert_delay_s": 5.0,
    "recover_delay_s": 2.0,
    "break_interval_min": 45,
    "sound": True,
}

def load_settings():
    s = dict(DEFAULT_SETTINGS)
    try:
        with open(os.path.join(LOG_DIR, "settings.json")) as f:
            s.update(json.load(f))
    except Exception:
        pass
    return s

def save_settings(s):
    with open(os.path.join(LOG_DIR, "settings.json"), "w") as f:
        json.dump(s, f, indent=2)

SETTINGS = load_settings()

# ------------------------------------------------------------------ crafted sounds

SOUND_DIR = os.path.join(LOG_DIR, "sounds")

def ensure_sounds():
    """Generate small, pleasant synthesized tones (much nicer than raw system beeps).
    nudge  = soft rising two-note  — 'ease up' early warning
    alert  = clear doorbell-style double chime — slouch alert
    recover= short major arpeggio — positive reinforcement
    """
    try:
        import wave
        os.makedirs(SOUND_DIR, exist_ok=True)
        sr = 22050
        def tone(path, notes, gap=0.03, vol=0.5):
            frames = []
            for f, dur in notes:
                n = int(sr * dur)
                t = np.arange(n) / sr
                env = np.minimum(t / 0.02, 1.0) * np.exp(-3.2 * t / dur)
                w = np.sin(2 * np.pi * f * t) * env
                w += 0.22 * np.sin(2 * np.pi * f * 2 * t) * env   # soft overtone
                frames.append(w)
                frames.append(np.zeros(int(sr * gap)))
            data = np.concatenate(frames)
            data = (data / max(1e-9, np.abs(data).max()) * vol * 32767).astype(np.int16)
            with wave.open(path, "w") as wf:
                wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
                wf.writeframes(data.tobytes())
        tone(os.path.join(SOUND_DIR, "nudge.wav"),   [(520, 0.16), (660, 0.20)], vol=0.34)
        tone(os.path.join(SOUND_DIR, "alert.wav"),   [(660, 0.13), (880, 0.20)], vol=0.55)
        tone(os.path.join(SOUND_DIR, "recover.wav"), [(523, 0.09), (659, 0.09), (784, 0.16)], vol=0.42)
    except Exception:
        pass

# ------------------------------------------------------------------ pose metrics

NOSE, LEYE_O, REYE_O, LEAR, REAR, LSH, RSH = 0, 3, 6, 7, 8, 11, 12
TORSO_PAIRS = [(LSH, RSH)]
HEAD_PAIRS = [(LEAR, REAR)]

def ema(prev, new, alpha=0.35):
    return new if prev is None else prev + alpha * (new - prev)

_S = {}
_MED = {}

def sm(name, v):
    """Exponential smoothing on top of the median filter."""
    _S[name] = ema(_S.get(name), v)
    return _S[name]

def med(name, v, k=5):
    """Median-of-5 filter: kills single-frame landmark jumps before smoothing."""
    d = _MED.setdefault(name, deque(maxlen=k))
    d.append(v)
    return float(np.median(d))

def reset_smoother():
    _S.clear()
    _MED.clear()

# neutral face orientation captured at calibration; pose_delta is the rotation
# magnitude between current and neutral (invariant to Euler conventions)
FACE_BASE = {"R": None}

def compute_metrics(lms, face_rot=None):
    """Metrics from 3D WORLD landmarks (meters, origin at hip mid, y down, z toward camera).
    Distance-independent and stable across camera angles — real physical measures."""
    """Metrics from 3D WORLD landmarks (meters, origin at hip mid, y down, z toward camera).
    Distance-independent and stable across camera angles — real physical measures."""
    p = lms
    def w(i):
        return np.array([p[i].x, p[i].y, p[i].z])
    ls, rs = w(LSH), w(RSH)
    le, re = w(LEYE_O), w(REYE_O)
    lear, rear = w(LEAR), w(REAR)
    sh_mid = (ls + rs) / 2
    ear_mid = (lear + rear) / 2
    ear_vis = min(p[LEAR].visibility, p[REAR].visibility)
    sh_vis = min(p[LSH].visibility, p[RSH].visibility)
    if ear_vis < 0.5 or sh_vis < 0.3:
        return None
    # forward head posture: ears closer to the camera than the shoulder line.
    # World-landmark z: smaller (more negative) = closer to camera.
    forward_mm = float((sh_mid[2] - ear_mid[2]) * 1000)
    # head height above shoulder line (shrinks when you slump)
    vert_mm = float((sh_mid[1] - ear_mid[1]) * 1000)
    tilt = abs(float(math.degrees(math.atan2(re[1] - le[1], max(1e-6, abs(re[0] - le[0]))))))
    lean = float(abs(math.degrees(math.atan2(rs[1] - ls[1], max(1e-6, abs(rs[0] - ls[0]))))))
    pose_delta = 0.0
    if face_rot is not None and FACE_BASE["R"] is not None:
        Rd = face_rot @ FACE_BASE["R"].T
        tr = float(np.clip((np.trace(Rd) - 1.0) / 2.0, -1.0, 1.0))
        pose_delta = float(math.degrees(math.acos(tr)))
    return {
        "forward": sm("forward", med("forward", forward_mm)),
        "vert": sm("vert", med("vert", vert_mm)),
        "tilt": sm("tilt", med("tilt", tilt)),
        "lean": sm("lean", med("lean", lean)),
        "pose": sm("pose", med("pose", pose_delta)),
        "sh_low": bool(sh_vis < 0.55),
    }

def evaluate(m, base, s):
    """3-tier verdict: GOOD / WARN (early nudge, still fine) / BAD (violation).
    WARN fires at ~55% of the tolerance so you correct before an alert is due."""
    if m is None:
        return "NODETECT", "No person detected — sit in frame"
    if base is None:
        return "WARN", "Learning your baseline…"
    if m.get("sh_low"):
        return "NODETECT", "Move back a little — show head and shoulders"
    problems, warns = [], []
    dfwd = m["forward"] - base["forward"]
    fwd_tol = 10.0 + 100.0 * s["slouch_sensitivity"]
    vref = base["vert"]
    sunk_pct = max(0.0, (vref - m["vert"]) / vref * 100) if vref > 0 else 0.0
    drift = max(dfwd / fwd_tol if fwd_tol > 0 else 0.0,
                sunk_pct / (100.0 * s["slouch_sensitivity"]) if s["slouch_sensitivity"] > 0 else 0.0)
    if dfwd > fwd_tol:
        problems.append(f"Head {dfwd:.0f} mm forward (tech-neck)")
    elif dfwd > fwd_tol * 0.55:
        warns.append(f"head drifting forward ({dfwd:.0f} mm)")
    if vref > 0:
        if m["vert"] < vref * (1.0 - s["slouch_sensitivity"]):
            problems.append(f"Head sunk {sunk_pct:.0f}%")
        elif sunk_pct > 100.0 * s["slouch_sensitivity"] * 0.55:
            warns.append(f"sinking ({sunk_pct:.0f}%)")
    # FaceLandmarker confirmation: real 3D head rotation from neutral.
    # Escalates a moderate sink+rotation to BAD (classic chin-tuck) — this catches
    # nodding that the ear-height measure alone under-calls, and never fires for
    # just turning to look sideways unless you are also sinking.
    pd = m.get("pose", 0.0)
    if pd > 12.0 and drift > 0.55:
        problems.append(f"Chin tuck {pd:.0f}°")
    if m["tilt"] > s["tilt_thresh_deg"]:
        problems.append(f"Head tilt {m['tilt']:.0f}°")
    elif m["tilt"] > s["tilt_thresh_deg"] * 0.65:
        warns.append(f"head tilting ({m['tilt']:.0f}°)")
    if m["lean"] > s["lean_thresh_deg"]:
        problems.append(f"Shoulders tilted {m['lean']:.0f}°")
    elif m["lean"] > s["lean_thresh_deg"] * 0.65:
        warns.append(f"shoulders rolling ({m['lean']:.0f}°)")
    if problems:
        return "BAD", " · ".join(problems[:2])
    if warns:
        return "WARN", "Ease up · " + warns[0]
    return "GOOD", "Upright — nice"

def draw_overlay(img, lms, status, reason, calib_progress):
    colors = {"GOOD": GOOD, "BAD": BAD, "WARN": WARN, "NODETECT": "#8a93a6",
              "CALIBRATING": WARN, "NOCAM": BAD}
    c = colors.get(status, "#ffffff")
    c = QColor(c); r, g, b = c.red(), c.green(), c.blue()
    H, W = img.shape[:2]
    if lms is not None:
        def px(i):
            return int(lms[i].x * W), int(lms[i].y * H)
        for a, bb in TORSO_PAIRS + HEAD_PAIRS:
            cv2.line(img, px(a), px(bb), (b, g, r), 3, cv2.LINE_AA)
        for ear, sh in [(LEAR, LSH), (REAR, RSH)]:
            e, s2 = px(ear), px(sh)
            cv2.circle(img, e, 4, (b, g, r), -1, cv2.LINE_AA)
            cv2.circle(img, s2, 5, (b, g, r), -1, cv2.LINE_AA)
            cv2.line(img, e, (e[0], s2[1]), (b, g, r), 1, cv2.LINE_AA)
        cv2.circle(img, px(NOSE), 4, (b, g, r), -1, cv2.LINE_AA)
    label = {"GOOD": "GOOD POSTURE", "BAD": "SLOUCHING", "WARN": "CALIBRATE ME",
             "NODETECT": "NO PERSON", "CALIBRATING": "CALIBRATING…", "NOCAM": "NO CAMERA"}[status]
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(img, (10, 10), (26 + tw, 34 + th), (20, 20, 20), -1)
    cv2.rectangle(img, (10, 10), (26 + tw, 34 + th), (b, g, r), 2)
    cv2.putText(img, label, (18, 28 + th // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (b, g, r), 2, cv2.LINE_AA)
    if reason:
        cv2.putText(img, reason[:64], (18, 52 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1, cv2.LINE_AA)
    if status == "CALIBRATING":
        cv2.rectangle(img, (10, 70), (10 + int(calib_progress * (W - 20)), 78), (60, 180, 255), -1)
    return img

# ------------------------------------------------------------------ worker thread

class Worker(QThread):
    frame_ready = Signal(QImage)
    state_ready = Signal(dict)
    alert_fired = Signal(str)
    event_added = Signal(str, str)   # type, msg

    def __init__(self):
        super().__init__()
        self.alive = True
        self.monitoring = True
        self.lock = threading.Lock()
        self.calib_base = None     # anchor learned at calibration time
        self.base = None           # adaptive baseline (drifts slowly while GOOD, clamped)
        self.calib_samples = None
        self.calib_progress = 0.0
        self.last_detect = 0.0
        self.good_s = 0.0
        self.bad_s = 0.0
        self.alerts = 0
        self.streak = 0.0
        self.warn_streak = 0.0
        self.last_nudge_t = 0.0
        self.last_alert_t = 0.0
        self.recover_pending = False
        self.good_streak_s = 0.0
        self.next_milestone = 0
        self.sit_s = 0.0
        self.status = "BOOTING"
        self.reason = ""
        self.metrics = {}
        self.today = self._load_today()
        self.events = deque(maxlen=50)

    def _today_path(self):
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

    def csv_path(self):
        return os.path.join(LOG_DIR, f"posture-{date.today().isoformat()}.csv")

    def start_calibration(self):
        with self.lock:
            if self.status in ("NOCAM", "BOOTING"):
                return False
            self.calib_samples = deque(maxlen=75)
            self.calib_progress = 0.0
            self.status = "CALIBRATING"
            self.reason = "Sit upright and stay still"
            reset_smoother()
        return True

    def reset_session(self):
        with self.lock:
            self.good_s = self.bad_s = self.alerts = 0.0
            self.streak = 0.0
            self.events.clear()
            self.events.appendleft({"t": time.time(), "type": "info", "msg": "Session stats reset"})
            self.event_added.emit("info", "Session stats reset")

    def _adapt_baseline(self, m, dt, now):
        """Adaptive baseline: drifts slowly toward your current posture while GOOD,
        clamped around the calibration anchor so gradual slouch can't fool it.
        A large sustained drift = camera setup changed (laptop ⇄ LCD) → re-anchor."""
        tau = 420.0                      # ~7 min time constant
        a = min(0.5, dt / tau)
        for k in ("forward", "vert"):
            self.base[k] += a * (m[k] - self.base[k])
        # clamp around anchor
        self.base["forward"] = max(self.calib_base["forward"] - 30.0,
                                   min(self.calib_base["forward"] + 30.0, self.base["forward"]))
        vc = max(1.0, self.calib_base["vert"])
        self.base["vert"] = max(vc * 0.78, min(vc * 1.22, self.base["vert"]))
        # sustained drift beyond clamp → the setup itself changed → re-anchor
        if abs(self.base["forward"] - self.calib_base["forward"]) >= 29.0 or \
           abs(self.base["vert"] / vc - 1.0) >= 0.20:
            self.calib_base = dict(self.base)
            self.base = dict(self.calib_base)
            self.events.appendleft({"t": now, "type": "info",
                                    "msg": "Camera setup changed — baseline updated automatically"})
            self.event_added.emit("info", "Camera setup changed — baseline updated automatically")

    def run(self):
        cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not cap.isOpened():
            with self.lock:
                self.status, self.reason = "NOCAM", "Camera blocked"
            self._emit_state()
            self.notify("Posture Guard needs camera access — System Settings → Privacy & Security → Camera, then restart.")
            return
        opts = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO, num_poses=1,
            min_pose_detection_confidence=0.5, min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5)
        landmarker = vision.PoseLandmarker.create_from_options(opts)
        face_opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=FACE_MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO, num_faces=1,
            min_face_detection_confidence=0.5, min_tracking_confidence=0.5,
            output_facial_transformation_matrixes=True)
        face_lm = vision.FaceLandmarker.create_from_options(face_opts)
        ensure_sounds()

        last_ts = 0
        last_tick = time.time()
        log_accum = {"good": 0.0, "bad": 0.0}
        last_log_t = time.time()

        while self.alive:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.2)
                continue
            scale = OVERLAY_W / frame.shape[1]
            frame = cv2.resize(frame, (OVERLAY_W, int(frame.shape[0] * scale)))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            H, W = rgb.shape[:2]
            ts = int(time.monotonic() * 1000)
            if ts <= last_ts:
                ts = last_ts + 1
            last_ts = ts
            result = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)), ts)
            # face head-pose every 3rd frame (rotation changes slowly)
            self.frame_n = getattr(self, "frame_n", 0) + 1
            if self.frame_n % 3 == 0:
                fres = face_lm.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)), ts)
                if fres.facial_transformation_matrixes:
                    M = np.array(fres.facial_transformation_matrixes[0].data,
                                 dtype=float).reshape(4, 4)
                    self.last_face_rot = M[:3, :3].copy()

            now = time.time()
            dt = now - last_tick
            last_tick = now
            m = None
            if result.pose_landmarks and self.monitoring and result.pose_world_landmarks:
                m = compute_metrics(result.pose_world_landmarks[0],
                                    getattr(self, "last_face_rot", None))

            with self.lock:
                s = dict(SETTINGS)
                if m is not None:
                    self.last_detect = now
                    self.sit_s += dt
                elif self.calib_samples is not None and self.base is None and now - self.last_detect > 2.0:
                    # person left mid auto-calibration — start over when they return
                    self.calib_samples = None
                    self.calib_progress = 0.0

                if self.calib_samples is not None:
                    # collecting calibration (manual button or automatic first run)
                    if m:
                        # stability gate: reject frames where you're still moving
                        prev = self.calib_samples[-1] if self.calib_samples else None
                        moving = prev is not None and (
                            abs(m["forward"] - prev["forward"]) > 8.0 or
                            abs(m["vert"] - prev["vert"]) > 10.0)
                        if not moving:
                            self.calib_samples.append(m)
                        self.calib_progress = min(1.0, len(self.calib_samples) / 75)
                        if len(self.calib_samples) >= 75:
                            self.calib_base = {
                                "forward": float(np.mean([x["forward"] for x in self.calib_samples])),
                                "vert": float(np.mean([x["vert"] for x in self.calib_samples])),
                                "tilt": float(np.mean([x["tilt"] for x in self.calib_samples])),
                                "lean": float(np.mean([x["lean"] for x in self.calib_samples]))}
                            self.base = dict(self.calib_base)
                            FACE_BASE["R"] = getattr(self, "last_face_rot", None)
                            self.calib_samples = None
                            self.status, self.reason = "GOOD", "Upright — nice"
                            self.events.appendleft({"t": now, "type": "info", "msg": "Calibration complete"})
                            self.event_added.emit("info", "Baseline learned — tracking your posture now")
                    status, reason = self.status, self.reason
                elif m is None:
                    self.status, self.reason = "NODETECT", "No person detected — sit in frame"
                    self.streak = 0.0
                    self.warn_streak = 0.0
                    self.good_streak_s = 0.0
                    self.next_milestone = 0
                    status, reason = self.status, self.reason
                else:
                    if self.base is None:
                        # AUTO-CALIBRATION: learn the upright baseline on first stable detection
                        self.calib_samples = deque(maxlen=75)
                        self.calib_progress = 0.0
                        self.status, self.reason = "CALIBRATING", "Sit upright — learning your baseline…"
                        reset_smoother()
                        status, reason = self.status, self.reason
                    else:
                        self.metrics = {k: round(v, 1) for k, v in m.items() if k != "sh_low"}
                        status, reason = evaluate(m, self.base, s)
                        self.status, self.reason = status, reason
                        if status == "BAD":
                            self.bad_s += dt; self.today["bad_s"] += dt
                            self.streak += dt
                            self.warn_streak = 0.0
                            log_accum["bad"] += dt
                            if self.streak >= s["alert_delay_s"]:
                                self.streak = 0.0
                                if now - self.last_alert_t >= 45.0:
                                    # cooldown: transient dips and repeat nagging suppressed
                                    self.last_alert_t = now
                                    self.alerts += 1; self.today["alerts"] += 1
                                    self.events.appendleft({"t": now, "type": "alert", "msg": reason})
                                    self.event_added.emit("alert", reason)
                                    self.recover_pending = True
                                    self.alert_fired.emit(reason)
                        elif status == "WARN":
                            # gentle early nudge — correct before it counts as slouch
                            self.warn_streak += dt
                            if self.warn_streak >= 2.0 and now - self.last_nudge_t >= 20.0:
                                self.last_nudge_t = now
                                self.events.appendleft({"t": now, "type": "info", "msg": reason})
                                self.event_added.emit("info", "Ease up — " + reason.split("· ")[-1])
                                self.play_sound("nudge")
                        else:
                            self.good_s += dt; self.today["good_s"] += dt
                            self.streak = 0.0
                            self.warn_streak = 0.0
                            log_accum["good"] += dt
                            if self.recover_pending:
                                self.recover_pending = False
                                self.events.appendleft({"t": now, "type": "info", "msg": "Posture recovered ✓"})
                                self.event_added.emit("info", "Posture recovered ✓")
                                self.play_sound("recover")
                            self.good_streak_s += dt
                            milestones = [(600, "10 minutes"), (1800, "30 minutes"), (3600, "1 hour")]
                            if self.next_milestone < len(milestones) and \
                               self.good_streak_s >= milestones[self.next_milestone][0]:
                                label = milestones[self.next_milestone][1]
                                self.next_milestone += 1
                                msg = f"🎉 {label} of continuous good posture"
                                self.events.appendleft({"t": now, "type": "info", "msg": msg})
                                self.event_added.emit("info", msg)
                                self.play_sound("recover")
                            self._adapt_baseline(m, dt, now)

                        # micro-break reminder (sitting of any kind counts)
                        interval = s.get("break_interval_min", 45)
                        if interval and interval > 0 and self.sit_s >= interval * 60:
                            self.sit_s = 0.0
                            self.good_streak_s = 0.0
                            self.next_milestone = 0
                            self.events.appendleft({"t": now, "type": "info",
                                                    "msg": "⏳ Time for a 30-second stand-up break"})
                            self.event_added.emit("info", "⏳ Time for a 30-second stand-up break")
                            self.notify("Stand up and stretch for 30 seconds — your spine will thank you.",
                                        subtitle="Micro-break")
                            self.play_sound("nudge")

                # periodic CSV + day-bucket logging
                if now - last_log_t >= 5.0:
                    last_log_t = now
                    total = log_accum["good"] + log_accum["bad"]
                    if total > 0:
                        try:
                            with open(self.csv_path(), "a", newline="") as f:
                                import csv as _csv
                                _csv.writer(f).writerow(
                                    [datetime.now().isoformat(timespec="seconds"),
                                     round(log_accum["good"], 1), round(log_accum["bad"], 1),
                                     status, reason])
                        except Exception:
                            pass
                        key = f"{datetime.now().hour:02d}:{str(datetime.now().minute // 5 * 5).zfill(2)}"
                        b = self.today["buckets"].setdefault(key, [0, 0])
                        b[0] += log_accum["good"]; b[1] += log_accum["bad"]
                        log_accum = {"good": 0.0, "bad": 0.0}
                        self.flush_today()

                annotated = draw_overlay(frame.copy(),
                                         result.pose_landmarks[0] if result.pose_landmarks else None,
                                         status, self.reason, self.calib_progress)
                img = QImage(annotated.data, W, H, 3 * W, QImage.Format_RGB888).copy()
                self._emit_state()
            self.frame_ready.emit(img)
        cap.release()
        self.flush_today()

    def _emit_state(self):
        self.state_ready.emit({
            "status": self.status, "reason": self.reason, "metrics": dict(self.metrics),
            "good_s": round(self.good_s), "bad_s": round(self.bad_s),
            "alerts": self.alerts, "bad_streak": round(self.streak, 1),
            "warn_streak": round(self.warn_streak, 1), "sit_s": round(self.sit_s),
            "calibrating": self.calib_samples is not None,
            "calib_progress": round(self.calib_progress, 2),
            "calibrated": self.calib_base is not None,
            "today": {"good_s": round(self.today["good_s"]), "bad_s": round(self.today["bad_s"]),
                      "alerts": self.today["alerts"], "buckets": dict(self.today["buckets"])},
            "events": list(self.events)[:12],
            "settings": dict(SETTINGS),
            "monitoring": self.monitoring,
        })

    _SYSTEM_FALLBACK = {"nudge": "Pop", "alert": "Glass", "recover": "Tink"}

    def play_sound(self, kind):
        """Play a crafted tone; fall back to a system sound if unavailable."""
        if not SETTINGS.get("sound", True):
            return
        path = os.path.join(SOUND_DIR, f"{kind}.wav")
        if os.path.exists(path):
            subprocess.Popen(["afplay", path],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        name = self._SYSTEM_FALLBACK.get(kind, "Pop")
        p = f"/System/Library/Sounds/{name}.aiff"
        if os.path.exists(p):
            subprocess.Popen(["afplay", p],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def play_alert_sound(self):
        """Alert chime; escalates to a stronger system sound if alerts repeat quickly."""
        if not SETTINGS.get("sound", True):
            return
        self.play_sound("alert")
        if time.time() - self.last_alert_t < 180:
            strong = "/System/Library/Sounds/Funk.aiff"
            if os.path.exists(strong):
                subprocess.Popen(["afplay", strong],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def notify(self, msg, subtitle=""):
        try:
            esc = msg.replace('"', '\\"')
            sub = f' subtitle "{subtitle}"' if subtitle else ""
            subprocess.Popen(
                ["osascript", "-e",
                 f'display notification "{esc}" with title "Posture Guard"{sub}'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

    def play_alert_sound(self):
        if not SETTINGS.get("sound", True):
            return
        for name in ("Ping", "Glass", "Funk"):
            path = f"/System/Library/Sounds/{name}.aiff"
            if os.path.exists(path):
                subprocess.Popen(["afplay", path],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return

# ------------------------------------------------------------------ icons (drawn, crisp at any DPI)

def make_icon(kind, color=TEXT, size=64):
    """Crisp vector-style icons painted with QPainter."""
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    pen = QPen(QColor(color), size * 0.055, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
    p.setPen(pen)
    c = QPointF(size / 2, size / 2)
    r = size * 0.38
    if kind == "target":
        p.drawEllipse(QPointF(c.x(), c.y()), r, r)
        p.drawEllipse(QPointF(c.x(), c.y()), r * 0.55, r * 0.55)
        pen2 = QPen(QColor(color), size * 0.055, Qt.SolidLine, Qt.RoundCap)
        p.setPen(pen2)
        for ang in (0, 90, 180, 270):
            a = math.radians(ang)
            x1 = c.x() + math.cos(a) * r * 0.78
            y1 = c.y() + math.sin(a) * r * 0.78
            x2 = c.x() + math.cos(a) * r * 1.12
            y2 = c.y() + math.sin(a) * r * 1.12
            p.drawLine(QPointF(x1, y1), QPointF(x2, y2))
    elif kind == "bell":
        p.drawArc(QRectF(c.x() - r * 0.7, c.y() - r * 0.75, r * 1.4, r * 1.5), 0, 180 * 16)
        p.drawLine(QPointF(c.x() - r * 0.7, c.y()), QPointF(c.x() + r * 0.7, c.y()))
        p.drawLine(QPointF(c.x() - r * 0.7, c.y()), QPointF(c.x() - r * 0.55, c.y() + r * 0.35))
        p.drawLine(QPointF(c.x() + r * 0.7, c.y()), QPointF(c.x() + r * 0.55, c.y() + r * 0.35))
        p.drawArc(QRectF(c.x() - r * 0.35, c.y() + r * 0.25, r * 0.7, r * 0.7), 180 * 16, 180 * 16)
        p.drawLine(QPointF(c.x() - r * 0.35, c.y() + r * 0.6), QPointF(c.x() + r * 0.35, c.y() + r * 0.6))
    elif kind == "reset":
        p.drawArc(QRectF(c.x() - r, c.y() - r, 2 * r, 2 * r), 45 * 16, 280 * 16)
        p.setBrush(QBrush(QColor(color)))
        p.setPen(Qt.NoPen)
        p.drawPolygon([QPointF(c.x() + r * 0.95, c.y() - r * 0.75),
                       QPointF(c.x() + r * 0.95, c.y() - r * 0.15),
                       QPointF(c.x() + r * 0.45, c.y() - r * 0.45)])
    elif kind == "pause":
        p.setBrush(QBrush(QColor(color)))
        p.setPen(Qt.NoPen)
        w = r * 0.35
        p.drawRoundedRect(QRectF(c.x() - w * 1.4, c.y() - r * 0.9, w, r * 1.8), w * 0.3, w * 0.3)
        p.drawRoundedRect(QRectF(c.x() + w * 0.4, c.y() - r * 0.9, w, r * 1.8), w * 0.3, w * 0.3)
    elif kind == "play":
        p.setBrush(QBrush(QColor(color)))
        p.setPen(Qt.NoPen)
        p.drawPolygon([QPointF(c.x() - r * 0.5, c.y() - r * 0.8),
                       QPointF(c.x() - r * 0.5, c.y() + r * 0.8),
                       QPointF(c.x() + r * 0.8, c.y())])
    elif kind == "chair":
        p.drawLine(QPointF(c.x() - r * 0.35, c.y() - r), QPointF(c.x() - r * 0.35, c.y() + r * 0.2))
        p.drawLine(QPointF(c.x() - r * 0.35, c.y() + r * 0.2), QPointF(c.x() + r * 0.75, c.y() + r * 0.2))
        p.drawLine(QPointF(c.x() + r * 0.75, c.y() + r * 0.2), QPointF(c.x() + r * 0.75, c.y() - r * 0.35))
        p.drawLine(QPointF(c.x() - r * 0.2, c.y() + r * 0.35), QPointF(c.x() - r * 0.2, c.y() + r))
        p.drawLine(QPointF(c.x() + r * 0.6, c.y() + r * 0.35), QPointF(c.x() + r * 0.6, c.y() + r))
    p.end()
    return pm

# ------------------------------------------------------------------ custom widgets

class ScoreRing(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(170, 170)
        self.pct = None
        self.color = QColor(GOOD)
        self.sub = "no data yet"

    def set_score(self, pct, color, sub):
        self.pct = pct; self.color = QColor(color); self.sub = sub
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        side = min(self.width(), self.height())
        cx, cy = self.width() / 2, self.height() / 2
        R = side * 0.40
        pen = QPen(QColor(BORDER), 13, Qt.SolidLine, Qt.RoundCap)
        p.setPen(pen)
        p.drawArc(QRectF(cx - R, cy - R, 2 * R, 2 * R), 0, 360 * 16)
        if self.pct is not None:
            pen.setColor(self.color)
            p.setPen(pen)
            span = int(-360 * 16 * self.pct / 100)
            p.drawArc(QRectF(cx - R, cy - R, 2 * R, 2 * R), 90 * 16, span)
        f = QFont(self.font()); f.setPixelSize(34); f.setBold(True); f.setLetterSpacing(QFont.AbsoluteSpacing, -1)
        p.setFont(f)
        p.setPen(QColor(TEXT if self.pct is None else self.color))
        txt = "—" if self.pct is None else f"{round(self.pct)}%"
        p.drawText(QRectF(0, cy - 26, self.width(), 40), Qt.AlignCenter, txt)
        f2 = QFont(self.font()); f2.setPixelSize(11)
        p.setFont(f2); p.setPen(QColor(MUTED))
        p.drawText(QRectF(0, cy + 12, self.width(), 18), Qt.AlignCenter, self.sub)
        p.end()

class TimelineChart(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(96)
        self.buckets = {}

    def set_buckets(self, b):
        self.buckets = dict(b)
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        keys = sorted(self.buckets.keys())
        p.setPen(QColor(MUTED))
        f = QFont(self.font()); f.setPixelSize(11); p.setFont(f)
        if not keys:
            p.drawText(QRectF(0, 0, w, h), Qt.AlignCenter,
                       "Today's timeline appears after a few minutes of monitoring")
            p.end(); return
        n = len(keys); gap = 2
        bw = max(4.0, (w - 8) / n - gap)
        base = h - 14; Hmax = h - 22
        cg = QColor(GOOD); cb = QColor(BAD)
        for i, k in enumerate(keys):
            g, b = self.buckets[k]; tot = g + b
            if not tot:
                continue
            hb = Hmax * min(1.0, tot / 300)
            hg = hb * (g / tot)
            x = 4 + i * (bw + gap)
            p.setPen(Qt.NoPen)
            p.setBrush(cb)
            p.drawRect(QRectF(x, base - hb, bw, hb - hg))
            p.setBrush(cg)
            p.drawRect(QRectF(x, base - hb, bw, hg))
        f2 = QFont(self.font()); f2.setPixelSize(10)
        p.setFont(f2); p.setPen(QColor(MUTED))
        p.drawText(QRectF(0, h - 13, w, 12), Qt.AlignCenter, "today · 5-minute buckets")
        p.end()

class StatBox(QFrame):
    def __init__(self, label):
        super().__init__()
        self.setStyleSheet(f"background:{PANEL2};border:1px solid {BORDER};border-radius:10px")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 9, 12, 9)
        lay.setSpacing(2)
        self.k = QLabel(label)
        self.k.setStyleSheet(f"color:{MUTED};font-size:11px;font-weight:600;border:0;background:transparent")
        self.v = QLabel("0s")
        self.v.setStyleSheet(f"font-size:19px;font-weight:700;border:0;background:transparent")
        lay.addWidget(self.k); lay.addWidget(self.v)

    def set(self, text, color=None):
        self.v.setText(text)
        if color:
            self.v.setStyleSheet(f"font-size:19px;font-weight:700;color:{color};border:0;background:transparent")

# ------------------------------------------------------------------ alert banner

class AlertBanner(QFrame):
    def __init__(self, parent):
        super().__init__(parent)
        self.setStyleSheet(f"background:#2a1214;border:1px solid {BAD};border-radius:12px")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(10)
        self.ic = QLabel("⚠️")
        self.msg = QLabel("")
        self.msg.setWordWrap(True)
        self.msg.setStyleSheet(f"color:#ffb1b1;font-size:13px;font-weight:600;background:transparent")
        self.dismiss = QPushButton("✕")
        self.dismiss.setFixedSize(26, 26)
        self.dismiss.setStyleSheet(f"background:transparent;color:{MUTED};font-size:13px;border:0")
        self.dismiss.clicked.connect(self.hide)
        lay.addWidget(self.ic); lay.addWidget(self.msg, 1); lay.addWidget(self.dismiss)
        self.hide()

    def show_alert(self, reason):
        self.msg.setText(f"<b>Fix your posture</b> — {reason}")
        parent = self.parentWidget()
        if parent:
            self.adjustSize()
            self.move((parent.width() - self.width()) // 2, 16)
        self.show()
        self.raise_()
        QTimer.singleShot(7000, self.hide)

# ------------------------------------------------------------------ main window

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Posture Guard")
        self.resize(1120, 700)
        self.worker = None
        self.latest = None
        self.scope = "session"
        self._build_ui()
        self._build_tray()
        self.start_worker()

    def start_worker(self):
        if self.worker is not None:
            self.worker.alive = False
            self.worker.wait(1500)
        self.worker = Worker()
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.state_ready.connect(self.on_state)
        self.worker.alert_fired.connect(self.on_alert)
        self.worker.event_added.connect(self.on_event)
        self.worker.start()

    # ---------- UI construction

    def _build_ui(self):
        central = QWidget()
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{ background:{BG}; color:{TEXT}; }}
            QLabel {{ color:{TEXT}; background:transparent; }}
            QFrame#card {{ background:{PANEL}; border:1px solid {BORDER}; border-radius:14px; }}
            QFrame#cardhead {{ border:0; background:transparent; }}
            QPushButton {{ border:0; border-radius:10px; padding:11px 16px; font-size:14px; font-weight:700; }}
            QPushButton#primary {{ background:{ACCENT}; color:#fff; }}
            QPushButton#primary:hover {{ background:#7d9bff; }}
            QPushButton#ghost {{ background:{PANEL2}; border:1px solid {BORDER}; color:{TEXT}; }}
            QPushButton#ghost:hover {{ background:#1f2735; }}
            QSlider::groove:horizontal {{ height:5px; background:{BORDER}; border-radius:2px; }}
            QSlider::handle:horizontal {{ width:16px; margin:-6px 0; border-radius:8px; background:{ACCENT}; }}
            QSlider::sub-page:horizontal {{ background:{ACCENT}; border-radius:2px; }}
            QCheckBox {{ font-size:13.5px; }}
            QCheckBox::indicator {{ width:18px; height:18px; border-radius:5px;
                                    border:1px solid {BORDER}; background:{PANEL2}; }}
            QCheckBox::indicator:checked {{ background:{ACCENT}; border-color:{ACCENT}; }}
            QListWidget {{ background:transparent; border:0; }}
            QListWidget::item {{ padding:7px 8px; border-radius:8px; }}
            QListWidget::item:selected {{ background:{PANEL2}; }}
        """)
        self.central_lay = QGridLayout(central)
        self.central_lay.setContentsMargins(20, 18, 20, 16)
        self.central_lay.setSpacing(16)

        # header
        head = QHBoxLayout()
        logo = QLabel(); logo.setPixmap(make_icon("chair", ACCENT, 46))
        title = QLabel("Posture Guard")
        title.setStyleSheet("font-size:20px;font-weight:800;")
        sub = QLabel("Tech-neck coach — live from your Mac webcam")
        sub.setStyleSheet(f"color:{MUTED};font-size:13px")
        tcol = QVBoxLayout(); tcol.addWidget(title); tcol.addWidget(sub)
        self.pauseBtn = QPushButton()
        self.pauseBtn.setIcon(QIcon(make_icon("pause", TEXT, 64)))
        self.pauseBtn.setText(" Pause monitoring")
        self.pauseBtn.setObjectName("ghost")
        self.pauseBtn.clicked.connect(self.toggle_monitoring)
        head.addWidget(logo); head.addSpacing(6); head.addLayout(tcol)
        head.addStretch(); head.addWidget(self.pauseBtn)
        self.central_lay.addLayout(head, 0, 0, 1, 2)

        # ---- left: video card
        vcard = QFrame(); vcard.setObjectName("card")
        vlay = QVBoxLayout(vcard); vlay.setContentsMargins(0, 0, 0, 12); vlay.setSpacing(0)
        vh = QFrame(); vh.setObjectName("cardhead")
        vhl = QHBoxLayout(vh); vhl.setContentsMargins(16, 12, 16, 10)
        vhl.addWidget(QLabel("Live camera"))
        self.statusPill = QLabel("Booting…")
        self.statusPill.setStyleSheet(f"""
            background:#0d1017;border:1px solid {BORDER};border-radius:12px;
            padding:5px 12px;font-size:12.5px;font-weight:700;color:{GOOD}""")
        vhl.addStretch(); vhl.addWidget(self.statusPill)
        self.video = QLabel()
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setMinimumSize(560, 315)
        self.video.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video.setStyleSheet(f"background:#000;border-radius:0 0 0 0; border:0")
        self.video.setContentsMargins(0, 0, 0, 0)
        # --- camera-blocked overlay (shown when NOCAM)
        self.nocam = QFrame(self.video)
        self.nocam.setStyleSheet(f"background:#0d1017;border:1px solid {BORDER};border-radius:14px")
        nl = QVBoxLayout(self.nocam); nl.setContentsMargins(24, 20, 24, 20); nl.setSpacing(8)
        nl.addStretch()
        ntitle = QLabel("📷  Camera blocked")
        ntitle.setStyleSheet("font-size:16px;font-weight:700")
        ntitle.setAlignment(Qt.AlignCenter)
        nmsg = QLabel("1. macOS should show a permission dialog — click Allow.\n"
                      "   (System Settings → Privacy & Security → Camera → enable Posture Guard)\n"
                      "2. Then click the button below.")
        nmsg.setStyleSheet(f"color:{MUTED};font-size:12.5px")
        nmsg.setAlignment(Qt.AlignCenter)
        self.retryBtn = QPushButton("  Check camera again")
        self.retryBtn.setIcon(QIcon(make_icon("reset", "#ffffff", 64)))
        self.retryBtn.setObjectName("primary")
        self.retryBtn.setFixedWidth(240)
        self.retryBtn.clicked.connect(self.restart_camera)
        nr = QHBoxLayout(); nr.addStretch(); nr.addWidget(self.retryBtn); nr.addStretch()
        nl.addWidget(ntitle); nl.addWidget(nmsg); nl.addSpacing(4); nl.addLayout(nr); nl.addStretch()
        self.nocam.hide()
        self.reasonLbl = QLabel("")
        self.reasonLbl.setStyleSheet(f"""
            background:rgba(13,16,23,0.8);border:1px solid {BORDER};border-radius:10px;
            padding:5px 12px;font-size:12.5px;color:#dfe5ef""")
        self.reasonLbl.hide()
        self.streakBar = QFrame()
        self.streakBar.setFixedHeight(6)
        self.streakBar.setStyleSheet(f"background:{BORDER};border-radius:3px;margin:0 14px")
        self.streakFill = QLabel(self.streakBar)
        self.streakFill.setStyleSheet(f"background:{BAD};border-radius:3px")
        self.streakFill.setGeometry(0, 0, 0, 6)
        vlay.addWidget(vh); vlay.addWidget(self.video, 1)
        vwrap = QWidget(); wl = QHBoxLayout(vwrap)
        wl.setContentsMargins(14, 8, 14, 0)
        wl.addWidget(self.reasonLbl); wl.addStretch()
        vlay.addWidget(vwrap)
        self.timeline = TimelineChart()
        tlab = QLabel("Today")
        tlab.setStyleSheet(f"color:{MUTED};font-size:13px;font-weight:600;margin:8px 16px 0 16px")
        vlay.addWidget(tlab)
        vlay.addWidget(self.timeline)
        self._shadow(vcard)
        self.central_lay.addWidget(vcard, 1, 0)

        # banner overlay on video
        self.banner = AlertBanner(self)

        # ---- right column
        right = QVBoxLayout(); right.setSpacing(16)

        score_card = QFrame(); score_card.setObjectName("card")
        sl = QVBoxLayout(score_card); sl.setContentsMargins(16, 14, 16, 14)
        sh = QHBoxLayout()
        sh.addWidget(QLabel("Posture score"))
        self.scopeBtn = QPushButton("session ⇄")
        self.scopeBtn.setStyleSheet(f"background:transparent;color:{ACCENT};font-size:12.5px;padding:2px")
        self.scopeBtn.setCursor(Qt.PointingHandCursor)
        self.scopeBtn.clicked.connect(self.toggle_scope)
        sh.addStretch(); sh.addWidget(self.scopeBtn)
        sl.addLayout(sh)
        self.ring = ScoreRing()
        self.ringsub = QLabel("Session score = good ÷ monitored time")
        self.ringsub.setStyleSheet(f"color:{MUTED};font-size:12px")
        self.ringsub.setAlignment(Qt.AlignCenter)
        rc = QVBoxLayout(); rc.addWidget(self.ring, 0, Qt.AlignCenter); rc.addWidget(self.ringsub)
        sl.addLayout(rc)
        grid = QGridLayout(); grid.setSpacing(10)
        self.stGood = StatBox("Good time"); self.stBad = StatBox("Slouch time")
        self.stAlerts = StatBox("Alerts today"); self.stMon = StatBox("Monitored")
        for i, sb in enumerate((self.stGood, self.stBad, self.stAlerts, self.stMon)):
            grid.addWidget(sb, i // 2, i % 2)
        sl.addLayout(grid)
        right.addWidget(score_card)
        self._shadow(score_card)

        ctrl_card = QFrame(); ctrl_card.setObjectName("card")
        cl = QVBoxLayout(ctrl_card); cl.setContentsMargins(16, 14, 16, 14); cl.setSpacing(12)
        row = QHBoxLayout(); row.setSpacing(10)
        self.calBtn = QPushButton("  Calibrate (sit upright)")
        self.calBtn.setIcon(QIcon(make_icon("target", "#ffffff", 64)))
        self.calBtn.setObjectName("primary")
        self.calBtn.clicked.connect(self.calibrate)
        resetBtn = QPushButton()
        resetBtn.setIcon(QIcon(make_icon("reset", TEXT, 64)))
        resetBtn.setToolTip("Reset session stats")
        resetBtn.setFixedWidth(46)
        resetBtn.setObjectName("ghost")
        resetBtn.clicked.connect(self.reset_session)
        row.addWidget(self.calBtn, 1); row.addWidget(resetBtn)
        cl.addLayout(row)
        self.sliders = {}
        def fmt_val(key, v):
            return {"slouch_sensitivity": f"{v}%", "tilt_thresh_deg": f"{v}°",
                    "alert_delay_s": f"{v}s",
                    "break_interval_min": ("off" if v == 0 else f"{v}m")}[key]
        for key, label, lo, hi in (
                ("slouch_sensitivity", "Slouch tolerance", 5, 35),
                ("tilt_thresh_deg", "Head tilt limit", 5, 25),
                ("alert_delay_s", "Alert delay", 1, 15),
                ("break_interval_min", "Break reminder", 0, 90)):
            h = QHBoxLayout()
            lab = QLabel(label); lab.setStyleSheet(f"color:{MUTED};font-size:12.5px")
            lab.setFixedWidth(120)
            sl_ = QSlider(Qt.Horizontal)
            sl_.setRange(lo, hi)
            sl_.setSingleStep(5 if key == "break_interval_min" else 1)
            val = QLabel(""); val.setFixedWidth(40); val.setAlignment(Qt.AlignRight)
            sl_.valueChanged.connect(lambda v, k=key, l=val: l.setText(fmt_val(k, v)))
            sl_.sliderReleased.connect(lambda s=sl_, k=key: self.push_setting(k, s.value()))
            self.sliders[key] = (sl_, val)
            h.addWidget(lab); h.addWidget(sl_, 1); h.addWidget(val)
            cl.addLayout(h)
        soundRow = QHBoxLayout()
        bell = QLabel(); bell.setPixmap(make_icon("bell", MUTED, 30))
        soundRow.addWidget(bell)
        soundRow.addWidget(QLabel("Sound alerts (system sound + notification)"))
        soundRow.addStretch()
        self.soundChk = QCheckBox()
        self.soundChk.setChecked(bool(SETTINGS.get("sound", True)))
        self.soundChk.toggled.connect(lambda v: self.push_setting("sound", bool(v)))
        soundRow.addWidget(self.soundChk)
        cl.addLayout(soundRow)
        right.addWidget(ctrl_card)
        self._shadow(ctrl_card)

        ev_card = QFrame(); ev_card.setObjectName("card")
        el = QVBoxLayout(ev_card); el.setContentsMargins(10, 10, 10, 10)
        ehead = QLabel("Activity")
        ehead.setStyleSheet(f"color:{MUTED};font-size:13px;font-weight:600;margin-left:6px")
        el.addWidget(ehead)
        self.events = QListWidget()
        self.events.setFixedHeight(180)
        el.addWidget(self.events)
        right.addWidget(ev_card, 1)
        self._shadow(ev_card)

        rwrap = QWidget(); rwrap.setLayout(right)
        self.central_lay.addWidget(rwrap, 1, 1)
        self.setCentralWidget(central)

        # sync sliders initially
        def fmt_val2(key, v):
            return {"slouch_sensitivity": f"{v}%", "tilt_thresh_deg": f"{v}°",
                    "alert_delay_s": f"{v}s",
                    "break_interval_min": ("off" if v == 0 else f"{v}m")}[key]
        for key, (s_, v_) in self.sliders.items():
            val0 = SETTINGS.get(key)
            s_.setValue(int(val0)); v_.setText(fmt_val2(key, int(val0)))

    def _build_tray(self):
        self.tray = QSystemTrayIcon(self)
        self.tray.setIcon(QIcon(make_icon("chair", GOOD, 128)))
        menu = QMenu()
        self.act_show = QAction("Show dashboard", self)
        self.act_show.triggered.connect(self.show_and_raise)
        self.act_toggle = QAction("Pause monitoring", self)
        self.act_toggle.triggered.connect(self.toggle_monitoring)
        self.act_cal = QAction("Calibrate…", self)
        self.act_cal.triggered.connect(self.calibrate)
        act_quit = QAction("Quit Posture Guard", self)
        act_quit.triggered.connect(self.closeEvent and self.quit_app)
        menu.addAction(self.act_show); menu.addAction(self.act_toggle)
        menu.addAction(self.act_cal); menu.addSeparator(); menu.addAction(act_quit)
        self.tray.setContextMenu(menu)
        self.tray.setToolTip("Posture Guard")
        self.tray.activated.connect(lambda r: self.show_and_raise() if r == QSystemTrayIcon.Trigger else None)
        self.tray.show()

    # ---------- slots

    def on_frame(self, img):
        self.latest = img
        self.video.setPixmap(QPixmap.fromImage(img).scaled(
            self.video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        if self.banner.isVisible() and self.banner.parentWidget():
            self.banner.move((self.width() - self.banner.width()) // 2, self.centralWidget().y() + 16)

    def on_state(self, st):
        col = {"GOOD": GOOD, "BAD": BAD, "WARN": WARN, "CALIBRATING": WARN,
               "NODETECT": MUT, "NOCAM": BAD, "BOOTING": MUT}.get(st["status"], MUT)
        self.statusPill.setText({"GOOD": "Good posture", "BAD": "Slouching",
                                 "WARN": "Ease up — drifting", "NODETECT": "No person detected",
                                 "CALIBRATING": "Calibrating…", "NOCAM": "No camera access",
                                 "BOOTING": "Booting…"}[st["status"]])
        self.statusPill.setStyleSheet(f"""
            background:#0d1017;border:1px solid {BORDER};border-radius:12px;
            padding:5px 12px;font-size:12.5px;font-weight:700;color:{col}""")
        self.video.setStyleSheet(f"background:#000;border:3px solid {col};border-radius:0")
        self.nocam.setVisible(st["status"] == "NOCAM")
        if st["reason"]:
            self.reasonLbl.setText(st["reason"]); self.reasonLbl.show()
        else:
            self.reasonLbl.hide()
        if st["status"] == "WARN":
            frac = min(1.0, st.get("warn_streak", 0.0) / 2.0)
            fill = WARN
        else:
            frac = min(1.0, st["bad_streak"] / max(1.0, SETTINGS["alert_delay_s"]))
            fill = BAD
        self.streakFill.setStyleSheet(f"background:{fill};border-radius:3px")
        self.streakFill.setFixedWidth(int(self.streakBar.width() * frac))
        # score
        use_session = self.scope == "session"
        g = st["good_s"] if use_session else st["today"]["good_s"]
        b = st["bad_s"] if use_session else st["today"]["bad_s"]
        tot = g + b
        pct = (g / tot * 100) if tot > 0 else None
        colr = GOOD if pct is None or pct >= 85 else (WARN if pct >= 60 else BAD)
        self.ring.set_score(pct, colr, "waiting…" if pct is None and st["calibrated"] else
                            ("no data yet" if pct is None else ("session" if use_session else "today")))
        self.stGood.set(self.fmt_s(g), GOOD)
        self.stBad.set(self.fmt_s(b), BAD)
        self.stAlerts.set(str(st["alerts"] if use_session else st["today"]["alerts"]))
        self.stMon.set(self.fmt_s(tot))
        self.timeline.set_buckets(st["today"]["buckets"])
        if st["calibrating"]:
            self.calBtn.setText(f"  Calibrating… {int(st['calib_progress'] * 100)}%")
            self.calBtn.setEnabled(False)
        elif not self.calBtn.isEnabled():
            self.calBtn.setText("  Calibrate (sit upright)")
            self.calBtn.setEnabled(True)

    @staticmethod
    def fmt_s(s):
        s = int(s)
        if s < 60:
            return f"{s}s"
        m, h = s // 60, (s // 60) // 60
        return f"{h}h {m % 60}m" if h else f"{m}m {s % 60}s"

    def on_alert(self, reason):
        self.banner.show_alert(reason)
        self.worker.play_alert_sound()
        self.worker.notify(reason, subtitle="Slouching detected")
        self.tray.setIcon(QIcon(make_icon("chair", BAD, 128)))
        QTimer.singleShot(8000, lambda: self.tray.setIcon(QIcon(make_icon("chair", GOOD, 128))))

    def on_event(self, etype, msg):
        t = datetime.now().strftime("%H:%M:%S")
        icon = "⚠️" if etype == "alert" else "✅"
        item = QListWidgetItem(f"{icon}  {t}   {msg}")
        item.setForeground(QColor("#ffb1b1" if etype == "alert" else "#bcd0ff"))
        self.events.insertItem(0, item)

    # ---------- actions

    def calibrate(self):
        if self.worker:
            self.worker.start_calibration()

    def restart_camera(self):
        self.start_worker()

    def reset_session(self):
        if self.worker:
            self.worker.reset_session()

    def _shadow(self, w, blur=30, y=10):
        e = QGraphicsDropShadowEffect(self)
        e.setBlurRadius(blur); e.setXOffset(0); e.setYOffset(y)
        e.setColor(QColor(0, 0, 0, 120))
        w.setGraphicsEffect(e)

    def toggle_monitoring(self):
        if not self.worker:
            return
        self.worker.monitoring = not self.worker.monitoring
        on = self.worker.monitoring
        self.pauseBtn.setText(" Pause monitoring" if on else " Resume monitoring")
        self.pauseBtn.setIcon(QIcon(make_icon("pause" if on else "play", TEXT, 64)))
        self.act_toggle.setText("Pause monitoring" if on else "Resume monitoring")

    def toggle_scope(self):
        self.scope = "today" if self.scope == "session" else "session"
        self.scopeBtn.setText("today ⇄" if self.scope == "today" else "session ⇄")

    def push_setting(self, key, value):
        SETTINGS[key] = value
        save_settings(SETTINGS)

    def show_and_raise(self):
        self.show(); self.raise_(); self.activateWindow()

    def quit_app(self):
        if self.worker:
            self.worker.alive = False
            self.worker.wait(2000)
            self.worker.flush_today()
        QApplication.quit()

    def closeEvent(self, ev):
        # keep monitoring in the menu bar when window is closed
        self.hide()
        self.worker.notify("Still watching your posture from the menu bar.", subtitle="Minimized")
        ev.ignore()

    def resizeEvent(self, ev):
        self.nocam.setGeometry(self.video.rect())
        if self.banner.isVisible():
            self.banner.move((self.centralWidget().width() - self.banner.width()) // 2,
                             self.centralWidget().y() + 16)
        super().resizeEvent(ev)

# ------------------------------------------------------------------ main

def preflight_camera(timeout=90):
    """Trigger the macOS camera permission prompt from the MAIN thread.
    OpenCV shows the TCC dialog while opening the camera, and it can only spin
    the run loop on the main thread. Blocking-wait here until the user answers
    (or the timeout). Sets SKIP_AUTH afterwards so worker threads can open the
    device without run-loop errors."""
    deadline = time.time() + timeout
    opened = False
    while time.time() < deadline:
        cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            cap.release()
            opened = True
            break
        cap.release()
        time.sleep(0.4)
    os.environ["OPENCV_AVFOUNDATION_SKIP_AUTH"] = "1"
    return opened

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Posture Guard")
    app.setWindowIcon(QIcon(make_icon("chair", ACCENT, 256)))
    # show the camera permission prompt BEFORE any background thread touches the camera.
    # OpenCV can only spin the run loop for the TCC dialog on the main thread.
    preflight_camera(timeout=90)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
