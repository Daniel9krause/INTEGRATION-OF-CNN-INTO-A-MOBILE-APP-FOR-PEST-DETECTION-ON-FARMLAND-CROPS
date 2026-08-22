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
# python3/hostpython3 are pinned to 3.13.1 (must match each other exactly -
# p4a hard-checks this). Went through 3.11.9 (wrong theory: thought Python
# 3.14, this p4a release's default, caused a corrupted hostpython3 "desktop"
# pip3 - the real cause was a checkout-path length issue, unrelated to
# Python version, fixed in .github/workflows/build.yml) and 3.12.8 (needed
# for numpy>=2.4's Python>=3.12 requirement, but CPython's stock Modules/
# grpmodule.c unconditionally calls setgrent/getgrent/endgrent, which
# Android's libc doesn't implement - "not available on Android" per
# Python's own grp module docs. That gap was specifically fixed as part of
# CPython's own official Android platform support work (PEP 738), which
# targeted 3.13, so 3.13+ should have the real configure-level fix rather
# than just unconditionally trying to build grp.
# numpy IS pinned (to v2.5.2, note p4a's numpy recipe fetches by git tag so
# this needs the "v" prefix): the recipe's own default, numpy 2.3.0, fails
# to compile for Android — its unique.cpp is missing #include <unordered_map>,
# a confirmed numpy bug (fixed upstream in numpy/numpy#29662, merged Sept
# 2025). Any numpy release from after that fix works; 2.5.2 is current.
requirements = python3==3.13.1,hostpython3==3.13.1,kivy==2.3.0,pillow,numpy==v2.5.2,tflite-runtime,plyer

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
p4a.branch = master
p4a.commit = v2026.05.09

[buildozer]
log_level = 2
warn_on_root = 1
