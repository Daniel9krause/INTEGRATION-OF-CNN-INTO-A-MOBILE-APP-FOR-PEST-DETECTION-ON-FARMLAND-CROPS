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
# numpy is left unpinned deliberately: p4a's numpy recipe (kivy/python-for-
# android) is a MesonRecipe with version="v2.3.0" hardcoded — every numpy
# version builds through the same meson path regardless, so pinning an
# older numpy (tried, reverted) doesn't help and can actively break things
# (old numpy has no meson.build at all). The real fix is pinning p4a itself
# — see p4a.branch/p4a.commit below.
requirements = python3,kivy==2.3.0,pillow,numpy,tflite-runtime,plyer

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
# default unpinned "master" branch. p4a has had NO tagged release between
# 2024.01.21 and 2026.05.09 — meaning every unpinned build tracks whatever
# bleeding-edge master happens to be that day. Our first two builds hit a
# master-branch regression in the numpy/MesonRecipe hostpython3 bootstrap
# ("Exec format error" building a native pip3) that this pin avoids.
p4a.branch = master
p4a.commit = v2026.05.09

[buildozer]
log_level = 2
warn_on_root = 1
