#!/bin/bash
# Fetch MediaPipe .task models (one-time, ~13 MB)
set -e
cd "$(dirname "$0")/models"
curl -L -o pose_landmarker_full.task "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
curl -L -o face_landmarker.task "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
echo "models ready ✓"
