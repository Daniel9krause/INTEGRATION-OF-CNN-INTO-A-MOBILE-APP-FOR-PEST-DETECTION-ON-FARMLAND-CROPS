# Farm_land Detect and Classification System (FDCS)

A mobile app that helps farmers detect pests and diseases on their crops in
real time. Point your phone camera at a leaf or pest, capture, and the app
classifies it into one of 13 organism groups (insect pests, fungal disease,
viral disease, bacterial disease) and gives practical advisory steps.

## Features (mapped to your requirements)

| Requirement | Where it lives |
|---|---|
| Capture real-world images (camera) or upload an existing photo | `screens/home_screen.py` — live `Camera` widget + native "Upload" file picker (`plyer`) |
| Classify pest + organism group | `model/classifier.py` (MobileNetV2 TFLite) + `data/advisory_data.json` |
| View Classification Result | `screens/result_screen.py` |
| View Advisory Information | Built into Result screen + standalone `screens/advisory_screen.py` |
| View Scan History | `screens/history_screen.py` + `utils/database.py` (SQLite) |
| Store new pest info gathered | `utils/data_collector.py` — files every captured image into a class-labeled folder, growing your own dataset over time, plus a "flag as new/unknown organism" button on the Result screen |

## Project structure

```
farmland_detect/
├── main.py                    # App entry point, screen manager
├── buildozer.spec             # Android build configuration
├── requirements-desktop.txt   # For testing on your PC
├── screens/
│   ├── home_screen.py         # Camera capture
│   ├── result_screen.py       # Classification + advisory
│   ├── history_screen.py      # Past scans (SQLite)
│   └── advisory_screen.py     # Full 13-class reference library
├── model/
│   ├── classifier.py          # TFLite inference wrapper
│   ├── labels.txt             # 13 class names
│   └── pest_model.tflite      # <-- python train/train_model.py generates this (see below)
├── train/
│   └── train_model.py         # MobileNetV2 transfer-learning + TFLite export
├── data/
│   ├── advisory_data.json     # Advisory knowledge base per class
│   └── dataset/                # <-- YOU ADD training images here, one folder per class
│       └── README.md          # Exact folder layout + dataset sources per class
├── utils/
│   ├── database.py            # SQLite scan history
│   └── data_collector.py      # Grows a labeled dataset from real captures
├── assets/                    # .kv layout files + icon
└── .github/workflows/build.yml # Cloud APK build (works around Windows/Buildozer)
```

## 1. Run and test on your PC first (fastest feedback loop)

Kivy runs fine on Windows/Mac/Linux — you don't need a phone to develop.

```bash
cd farmland_detect
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux
pip install -r requirements-desktop.txt
python main.py
```

This opens a phone-sized window (400x720) on your desktop. Your webcam
substitutes for the phone camera. Everything — capture, classification
(mock mode if no model yet), history, advisory — works exactly the same.

## 2. Add your trained model

Until a real model is in place, `model/classifier.py` runs in **mock
inference mode**: every capture/upload gets a random label from
`model/labels.txt`, just so the rest of the app (UI, database, advisory
flow) is testable end-to-end before training is done. This is why results
look unrelated to the actual photo — it's not a bug, it just isn't looking
at the image yet.

To make classification real:

1. Install training deps (same file used for desktop testing —
   `tensorflow-cpu` is already in there, nothing extra to install):
   ```bash
   pip install -r requirements-desktop.txt
   ```
2. Gather images and sort them into `data/dataset/<Class Name>/` — one
   folder per line in `model/labels.txt`, exact spelling. See
   [`data/dataset/README.md`](data/dataset/README.md) for dataset sources
   for each of the 13 classes and how to lay the folders out.
3. Train and export in one step:
   ```bash
   python train/train_model.py
   ```
   This runs MobileNetV2 transfer learning (frozen-base head training,
   then fine-tuning) and writes the result straight to
   `model/pest_model.tflite`. On a CPU-only laptop this can take anywhere
   from ~15 minutes to a couple hours depending on dataset size — normal
   for CPU training. No GPU? Google Colab gives free GPU time: upload
   `data/dataset/`, run the same script there, then copy the resulting
   `pest_model.tflite` back into this project's `model/` folder.
4. Restart the app (`python main.py`) — `PestClassifier` auto-detects the
   `.tflite` file and mock mode turns off automatically. No other code
   changes needed.

`model/labels.txt` must keep listing classes in the same order the model
was trained on (index 0 = first line, etc) — `train/train_model.py`
enforces this automatically, so as long as you don't hand-edit
`labels.txt` after training, it stays correct.

## 3. Build the Android APK (no Mac/Linux machine needed)

Buildozer (the tool that packages Kivy apps into APKs) **only runs on
Linux**. Since you're on Windows, use the included GitHub Actions
workflow to build in the cloud for free:

1. Push this whole `farmland_detect/` folder to a GitHub repository.
2. Go to your repo → **Actions** tab → select **"Build FDCS Android APK"**
   → click **"Run workflow"**.
3. Wait ~15-25 minutes (first build is slowest; it downloads the Android
   SDK/NDK). Subsequent builds are much faster.
4. When it finishes, open the workflow run → scroll to **Artifacts** →
   download `fdcs-debug-apk` (a zip containing your `.apk`).

## 4. Install on your Android phone

1. Transfer the `.apk` file to your phone (USB cable, Google Drive, email
   to yourself, etc).
2. On your phone, tap the `.apk` file. You'll get a prompt to
   **"Allow installation from this source"** — enable it (this only
   applies since it's not from the Play Store).
3. Tap **Install**. Once installed, open **"FDCS"** — grant Camera and
   Storage permissions when prompted.
4. Point the camera at a leaf/pest, tap **CAPTURE**, and you'll see the
   classification, organism group, and advisory instantly.

## 5. Where your data lives on the phone

- **Scan history**: SQLite database in the app's private storage
  (`utils/database.py` → `app_storage_path()`), survives app restarts.
- **Collected dataset**: `collected_data/<ClassName>/` — every image you
  capture is automatically filed by predicted class. If you tap
  **"flag as new/unknown organism"** on a result, that image goes into
  `collected_data/_unclassified_new/` instead, ready for you to review
  and potentially fold into your next training round as a 14th class.

## Notes on the Android build (issues already hit and fixed)

- `android.archs = arm64-v8a` only (not `armeabi-v7a`) — the
  python-for-android recipe for `tflite-runtime` is known to fail on
  32-bit `armeabi-v7a` builds ([reference](https://github.com/Android-for-Python/c4k_tflite_example)),
  and every phone from roughly the last 7 years is arm64 anyway.
- `p4a.branch = master` + `p4a.commit = v2026.05.09` pin
  python-for-android to its last tagged stable release, for reproducible
  builds (p4a has had no tagged release between 2024.01.21 and
  2026.05.09, so an unpinned build tracks whatever bleeding-edge master
  happens to exist that day).
- `python3==3.13.1,hostpython3==3.13.1` — went through 3.11.9 (wrong
  theory that Python 3.14, this p4a release's default, caused the `pip3`
  failure below) and 3.12.8 (needed for `numpy>=2.4`'s `Python>=3.12`
  requirement, but hit a *different* build failure: CPython's stock
  `Modules/grpmodule.c` unconditionally calls `setgrent`/`getgrent`/
  `endgrent`, which Android's libc doesn't implement — `grp` is
  officially "not available on Android" per Python's own docs. That gap
  was fixed as part of CPython's own official Android platform support
  work ([PEP 738](https://peps.python.org/pep-0738/)), which targeted
  3.13 — so 3.13+ has the real `configure`-level fix.
- `android.ndk = 27d` — bumped from `25b` after `numpy`'s FP16 SIMD code
  hit a genuine LLVM/clang AArch64 backend crash compiling for arm64
  (`fatal error: error in backend: Cannot select: ...
  AArch64ISD::STRICT_FCMP`, a known, iteratively-fixed limitation in
  older clang's strict-FP/half-precision codegen — confirmed via several
  matching `android/ndk` and `llvm-project` issue reports). NDK 25b
  bundles clang-14; 27d (the current NDK LTS release) bundles a much
  newer clang.
- **The actual root cause of `OSError: [Errno 8] Exec format error`
  running a freshly-built `pip3`** (hit on every attempt, regardless of
  numpy version, Python version, or p4a version — none of those were
  actually it): GitHub Actions checks this repo out under a path
  containing its own name **twice**
  (`/home/runner/work/<repo>/<repo>/...`), and this repo's name is 73
  characters long. Buildozer's `hostpython3` recipe generates a `pip3`
  script whose shebang line embeds the *full absolute path* to that
  build's own Python interpreter, many directories deeper still — the
  resulting shebang line is ~300 characters, well past the Linux
  kernel's ~128–256 byte shebang-line limit, so it gets silently
  truncated/corrupted and the kernel can't execute it. Fixed in
  `.github/workflows/build.yml` by copying the checkout to a short
  `/tmp/fdcs` path before running buildozer.
- `numpy==v2.5.2` (note the "v" prefix — p4a's numpy recipe fetches numpy
  via a git tag checkout, so it needs to match numpy's actual tag format
  exactly). Once the shebang-length issue above was fixed, the build got
  much further and hit a real numpy bug: the recipe's own default,
  numpy 2.3.0, is missing `#include <unordered_map>` in
  `numpy/_core/src/multiarray/unique.cpp`, a confirmed upstream bug fixed
  in [numpy/numpy#29662](https://github.com/numpy/numpy/pull/29662)
  (merged Sept 2025). Any numpy release after that fix works.

If the GitHub Actions build fails on something else entirely, paste the
failed step's log (Actions tab → the failed run → the red "Build APK with
Buildozer" step) and it can be diagnosed the same way these were: search
backward from the "Command failed:" line for the actual Python traceback
or compiler error, which is usually much further up than the final error
dump.
