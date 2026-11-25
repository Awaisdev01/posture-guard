#!/bin/bash
# Posture Guard launcher — native macOS app (PySide6).
# First run will trigger a macOS camera permission prompt: allow it for Terminal/python.
cd "$(dirname "$0")"
echo "🪑 Starting Posture Guard (native macOS app)…"
python3 app_mac.py
