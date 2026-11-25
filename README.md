# 🪑 Posture Guard

> Real-time tech-neck & posture coach for macOS — MediaPipe 3D body + face tracking,
> adaptive personal baselines, vote-locked alerts, and a native desktop + menu-bar experience.

![python](https://img.shields.io/badge/python-3.9%2B-blue) ![macos](https://img.shields.io/badge/platform-macOS%2012%2B-black) ![mediapipe](https://img.shields.io/badge/MediaPipe-Tasks%20API-orange) ![license](https://img.shields.io/badge/license-MIT-green)

---

## What it does

Posture Guard watches you through your Mac's webcam and coaches you to keep a healthy
posture — using the same engineering pattern as a production traffic-violation CV system
(multi-frame voting, locked verdicts, evidence logging), applied to *you*:

| Capability | Implementation |
|---|---|
| **3D posture metrics (mm, not pixels)** | MediaPipe **Pose Landmarker (full)** world landmarks — forward-head distance (mm) from ear-to-shoulder depth, head-height above shoulders (mm), head tilt & shoulder roll (°). Distance-invariant: works near or far from the camera. |
| **True head-rotation ("chin tuck") confirmation** | MediaPipe **FaceLandmarker** facial transformation matrix — rotation magnitude vs. neutral, invariant to Euler conventions and camera angle. Escalates moderate sink + rotation to a violation; never flags simply looking sideways. |
| **Personal auto-calibration** | ~3 s stability-gated sampling on first detection (rejects frames where you're moving). No manual baseline needed; manual recalibration available. |
| **Adaptive baseline** | Baseline drifts slowly (~7 min time constant) while posture is GOOD, **clamped** around the calibration anchor (±30 mm / ±22%) so posture creep can't fool it. Sustained drift beyond the clamp = camera setup changed (laptop ⇄ external LCD) → automatic re-anchor. |
| **3-tier verdict ladder** | 🟢 GOOD → 🟠 WARN ("Ease up", soft nudge at ~55% of tolerance) → 🔴 BAD (violation, sustained 5 s). Transient movements (reaching, fidgeting) never fire. |
| **De-bounced alerts** | Sustained-slouch requirement + 45 s cooldown between alerts. No nagging. |
| **Positive reinforcement** | Recovery chime when you correct, streak milestones at 10/30/60 min. |
| **Micro-break reminders** | Configurable (default 45 min) stand-up break notification. |
| **Crafted audio** | Synthesized tones (soft nudge / doorbell chime / recovery arpeggio) generated with numpy + wave; system-sound fallback; escalation on ignored alerts. |
| **Native macOS experience** | PySide6 desktop app: live video with skeleton overlay, score ring, 5-min timeline chart, activity feed, sliders — plus **menu-bar (tray) presence** that keeps monitoring with the window closed. |
| **Evidence logging** | `logs/posture-YYYY-MM-DD.csv` (5 s aggregates) + `logs/day-YYYY-MM-DD.json` (daily score, alerts, timeline buckets), persisted across restarts. |

## Quick start (end users)

```bash
git clone https://github.com/<you>/posture-guard.git
cd posture-guard
./download_models.sh   # one-time fetch of MediaPipe models (~13 MB)
./run.sh
```

- First run: allow **camera** access when macOS prompts (the app is promptable via the
  bundled `PostureGuard.app` — see *Packaging* below).
- Sit upright → it auto-calibrates in ~3 s → live coaching begins.
- Target a **session score ≥ 85%**.

## For developers

### Requirements

- macOS 12+ (Apple Silicon or Intel)
- Python 3.9+
- Camera permission for the launching app (Terminal, IDE, or the packaged bundle)

```bash
python3 -m pip install --upgrade pip
python3 -m pip install opencv-python mediapipe flask numpy PySide6
```

### Project layout

```
app_mac.py          native macOS app (entry point)
                    ├─ Worker(QThread)      camera → pose+face landmarkers → metrics →
                    │                       vote-locked state machine → signals
                    ├─ compute_metrics()    3D world-landmark metrics + median/EMA filters
                    ├─ evaluate()           3-tier verdict (GOOD / WARN / BAD)
                    ├─ _adapt_baseline()    clamped adaptive baseline + setup-change re-anchor
                    ├─ ensure_sounds()      numpy-synthesized alert tones
                    └─ MainWindow           PySide6 UI + tray + alert banner
app.py              alternate headless web dashboard (Flask + MJPEG at :8765)
static/index.html   web dashboard UI (for app.py only)
models/             MediaPipe .task models (see download_models.sh)
logs/               runtime data: settings.json, CSV + daily JSON logs, sounds/
run.sh              launcher (desktop app)
packaging/          .app bundle build script (see below)
```

### Core algorithm

```
frame (AVFoundation, main-thread TCC preflight → worker thread)
  → PoseLandmarker(VIDEO) ──► world landmarks ──► forward_mm, vert_mm, tilt, lean
  → FaceLandmarker(VIDEO, every 3rd frame) ──► rotation matrix
  │      pose_delta = deg(acos(trace(R · R_neutralᵀ / normalized)))   # convention-free
  → median-of-5 → EMA smoothing
  → evaluate(vs adaptive baseline):
       BAD   if forward > tol  or  vert drop > tol  or  (chin-tuck: pose>12° and drift>55%)
             or tilt > limit  or  lean > limit
       WARN  if any metric in the 55–100% band of tolerance
       GOOD  otherwise
  → vote-lock: BAD must persist alert_delay_s (5 s default) → alert (45 s cooldown)
  → adaptive baseline updates while GOOD (clamped); re-anchor on sustained drift
```

Key constants (all in `app_mac.py`): `TARGET_FPS`, `OVERLAY_W`, `tau` (adaptation time
constant), clamp bounds, `fwd_tol = 10 mm + 100·sensitivity`, chin-tuck threshold 12°.

### Metrics reference (all normalized/personalized)

| Metric | Meaning | Violation when |
|---|---|---|
| `forward` | ear midpoint depth in front of shoulder-line (mm) | > baseline + tolerance |
| `vert` | ear height above shoulder-line (mm) | < baseline · (1 − sensitivity) |
| `pose` | 3D head-rotation magnitude from neutral (°) | > 12° **and** combined drift > 55% |
| `tilt` | eye-line roll (°) | > threshold (default 14°) |
| `lean` | shoulder-line roll (°) | > threshold (default 18°) |

### Tuning & settings

Runtime-tunable in the UI (persisted to `logs/settings.json`):

- `slouch_sensitivity` (5–35%) — width of the good-posture band
- `tilt_thresh_deg`, `lean_thresh_deg` — angular limits
- `alert_delay_s` (1–15 s) — sustained-slouch requirement
- `break_interval_min` (0 = off) — micro-break reminder cadence
- `sound` — audio on/off

### Packaging as a real macOS app (camera-promptable)

The repo includes a script to build `~/Applications/PostureGuard.app`:

```bash
./packaging/build_app.sh
open ~/Applications/PostureGuard.app
```

The bundle has its own `CFBundleIdentifier` (`com.postureguard.app`), compiled arm64
launcher, ad-hoc signature and `NSCameraUsageDescription`, so macOS shows a proper
*"PostureGuard wants to access the camera"* prompt and the permission is grantable in
System Settings → Privacy & Security → Camera.

> Why it matters: processes launched from a plain shell inherit the shell's TCC
> responsibility and macOS may refuse to show the prompt ("Policy disallows prompt").
> Also: OpenCV can only spin the run-loop for the TCC dialog on the **main thread** —
> `preflight_camera()` handles this before the worker starts.

### Extending

- **New violation type** (e.g., seatbelt, drowsiness): implement a metrics function and a
  rule in `evaluate()` — the vote-lock/alert/logging pipeline is rule-agnostic.
- **Multi-person**: set `num_poses > 1` and key baselines per person (needs tracking IDs).
- **Cloud dashboard**: `app.py` already exposes `/api/state` as JSON — pipe it anywhere.

### Troubleshooting

| Symptom | Fix |
|---|---|
| "No camera access" | System Settings → Privacy & Security → Camera → enable your terminal app or PostureGuard, then use the in-app "Check camera again" button |
| Camera prompt never appears | Launch via the packaged `PostureGuard.app` (shell-launched python may be blocked from prompting) |
| `OpenCV bindings requires numpy` / arch errors | Rosetta-vs-arm64 mismatch — run `/usr/bin/arch -arm64 /usr/bin/python3` |
| Alerts too chatty | Raise **Alert delay** slider; cooldown is fixed at 45 s |
| Status stuck on "No person detected" | Sit so head **and shoulders** are in frame |

## Privacy

Everything runs **locally**. No video, frames, or metrics ever leave your machine.
Logs contain only timestamps and numeric posture aggregates — never images.

## License

MIT — see [LICENSE](LICENSE).
