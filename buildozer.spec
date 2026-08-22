[app]
title = Farm_land Detect and Classification System
package.name = fdcs
package.domain = org.danielkrause.gctu

source.dir = .
source.include_exts = py,png,jpg,jpeg,kv,json,txt,tflite
# Keep the training-image dataset, local scan history, and grown dataset
# OUT of the APK — they're huge, personal, and not needed for the app to
# run (only for retraining, which happens on a dev machine, not on-device).
source.exclude_dirs = dataset,storage,collected_data,train,.github,.git,__pycache__

version = 1.0

# tflite-runtime is loaded via a recipe/wheel on Android; numpy & pillow are
# needed for preprocessing; camera + android permissions handled by pyjnius;
# plyer backs the Home screen's "Upload" button (native gallery/file picker).
# python3/hostpython3 are pinned: this p4a release defaults its Android
# target Python to 3.14.2, which was breaking an earlier build (a corrupted
# hostpython3 "desktop" pip3). hostpython3's version must exactly match
# python3's (p4a enforces this with a hard check), so both are pinned
# together to 3.11.9 — a long-proven, widely-used target for Kivy/Android
# builds.
# numpy IS pinned (to v2.5.2, note p4a's numpy recipe fetches by git tag so
# this needs the "v" prefix): the recipe's own default, numpy 2.3.0, fails
# to compile for Android — its unique.cpp is missing #include <unordered_map>,
# a confirmed numpy bug (fixed upstream in numpy/numpy#29662, merged Sept
# 2025). Any numpy release from after that fix works; 2.5.2 is current.
requirements = python3==3.11.9,hostpython3==3.11.9,kivy==2.3.0,pillow,numpy==v2.5.2,tflite-runtime,plyer

orientation = portrait
fullscreen = 0

icon.filename = %(source.dir)s/assets/icon.png

android.permissions = CAMERA,WRITE_EXTERNAL_STORAGE,READ_EXTERNAL_STORAGE,INTERNET

android.api = 33
android.minapi = 24
android.ndk = 25b
# arm64-v8a only: the tflite-runtime p4a recipe is known to fail building
# for armeabi-v7a (32-bit) — see https://github.com/Android-for-Python/c4k_tflite_example.
# Every phone from the last ~7 years is arm64, so this isn't a real limitation.
android.archs = arm64-v8a
android.allow_backup = True

# Pin python-for-android to its last tagged stable release instead of the
# default unpinned "master" branch, for reproducible builds — p4a has had
# NO tagged release between 2024.01.21 and 2026.05.09, so an unpinned build
# tracks whatever bleeding-edge master happens to be on the day it runs.
# (This pin alone did NOT fix the Android build failure — that turned out
# to be this release's default target Python being 3.14.2; see the
# python3==3.11.9 pin above.)
p4a.branch = master
p4a.commit = v2026.05.09

[buildozer]
log_level = 2
warn_on_root = 1
