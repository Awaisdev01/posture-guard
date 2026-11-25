#!/bin/bash
# Build ~/Applications/PostureGuard.app — a real macOS bundle with its own camera
# permission identity (com.postureguard.app). See README "Packaging".
set -e
APP="$HOME/Applications/PostureGuard.app"
SRC="$(cd "$(dirname "$0")/.." && pwd)"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$SRC/models/../packaging/PostureGuard.icns" "$APP/Contents/Resources/PostureGuard.icns" 2>/dev/null || \
  echo "note: no .icns found (optional)"
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>Posture Guard</string>
    <key>CFBundleDisplayName</key><string>Posture Guard</string>
    <key>CFBundleIdentifier</key><string>com.postureguard.app</string>
    <key>CFBundleVersion</key><string>1.3</string>
    <key>CFBundleShortVersionString</key><string>1.3</string>
    <key>CFBundleExecutable</key><string>PostureGuard</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleIconFile</key><string>PostureGuard</string>
    <key>LSMinimumSystemVersion</key><string>12.0</string>
    <key>NSCameraUsageDescription</key><string>Posture Guard uses the camera to detect your head and shoulder posture and warn you when you slouch.</string>
    <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
PLIST
cat > /tmp/pg_launcher.c <<C
#include <unistd.h>
int main(void) {
    chdir("$SRC");
    execl("/usr/bin/python3", "python3", "$SRC/app_mac.py", (char *)0);
    return 1;
}
C
clang -arch arm64 -O2 -o "$APP/Contents/MacOS/PostureGuard" /tmp/pg_launcher.c
codesign --force --sign - "$APP"
tccutil reset Camera com.postureguard.app >/dev/null 2>&1 || true
echo "built $APP — open it and allow camera access"
