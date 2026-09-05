# Farm_land Detect and Classification System (FDCS)

A mobile app that helps farmers detect pests and diseases on their crops in
real time. Point your phone camera at a leaf or pest, capture, and the app
classifies it into one of 13 organism groups (insect pests, fungal disease,
viral disease, bacterial disease) and gives practical advisory steps.

If the photo is not a plant, or is too dark or blurred to read, the app says
so and declines to diagnose rather than guessing — see
[Knowing when not to answer](#knowing-when-not-to-answer).

## Features (mapped to your requirements)

| Requirement | Where it lives |
|---|---|
| Capture real-world images (camera) or upload an existing photo | `screens/home_screen.py` — live `Camera` widget + native "Upload" file picker (`plyer` on desktop, a direct Android Storage-Access-Framework picker on-device — see "Real-device issues found and fixed" below for why) |
| Classify pest + organism group | `model/classifier.py` (MobileNetV2 TFLite) + `data/advisory_data.json` |
| Refuse to guess on non-plants / unusable photos | `utils/image_quality.py` + the `Not Plant` class + `model/ood_stats.json` — see [Knowing when not to answer](#knowing-when-not-to-answer) |
| View Classification Result | `screens/result_screen.py` |
| View Advisory Information | Built into Result screen + standalone `screens/advisory_screen.py` |
| View Scan History | `screens/history_screen.py` + `utils/database.py` (SQLite) — each row has **View** and **Remove**; Remove deletes that one scan and its photo after a confirmation, so a list full of blurred retries can be tidied without clearing the lot |
| Store new pest info gathered | `utils/data_collector.py` — files every captured image into a class-labeled folder, growing your own dataset over time, plus a "flag as new/unknown organism" button on the Result screen |

## Project structure

```
farmland_detect/
├── main.py                    # App entry point, screen manager
├── buildozer.spec             # Android build configuration
├── requirements-desktop.txt   # For testing on your PC
├── screens/
│   ├── home_screen.py         # Camera capture + orientation control
│   ├── result_screen.py       # Classification, advisory, or an explicit refusal
│   ├── history_screen.py      # Past scans (SQLite)
│   └── advisory_screen.py     # Full pest reference library
├── model/
│   ├── classifier.py          # TFLite inference + the three refusal layers
│   ├── labels.txt             # 13 pest/disease classes + "Not Plant"
│   ├── pest_model.tflite      # <-- train/train_model.py generates this
│   └── ood_stats.json         # <-- and this: open-set reference statistics
├── train/                     # Desktop/Colab only; excluded from the APK
│   ├── train_model.py         # Transfer learning + TFLite export + OOD stats
│   ├── fetch_negatives.py     # Builds the "Not Plant" class from Commons
│   ├── evaluate_rejection.py  # Does it refuse what it should? (pass/fail)
│   └── calibrate_thresholds.py# Grounds the quality gate in real data
├── tests/
│   ├── test_camera_transform.py  # Orientation maths, no device needed
│   └── test_decision_logic.py    # Refusal rules, network stubbed out
├── data/
│   ├── advisory_data.json     # Advisory knowledge base per class
│   └── dataset/                # <-- YOU ADD training images here, one folder per class
│       └── README.md          # Exact folder layout + dataset sources per class
├── utils/
│   ├── image_quality.py       # Too dark / blurred / blank gate
│   ├── camera_transform.py    # Upright preview AND matching capture
│   ├── result_presentation.py # One place that decides what the UI shows
│   ├── app_settings.py        # Persisted preferences (camera rotation)
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

## 2. The trained model

`model/pest_model.tflite` is trained and committed — the app classifies
real photos out of the box, no setup needed for this step. Current
validation accuracy: **80.9%** across all 13 classes, trained on 1,175
real, deduplicated photos (see "Dataset provenance" below for exactly
where they came from and known weak spots).

If no `.tflite` file were present, `model/classifier.py` would fall back
to a **mock inference mode** (a random label from `model/labels.txt`) so
the rest of the app stays testable even before a model exists — that's
not the current state, but it's worth knowing if you ever delete or
retrain over the model file.

To retrain (e.g. after adding more images, especially for the classes
flagged as thin below):

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

### Dataset provenance and known weak spots

Per-class image counts actually used for the current model:

| Class | Images | Source |
|---|---|---|
| Beetle, Grasshopper, Healthy Leaf, Leaf Blight, Leaf Spot, Mosaic Virus, Moth, Weevil, Whitefly | 85–108 each | Kaggle datasets linked in [`data/dataset/README.md`](data/dataset/README.md) |
| Aphids | 57 | 21 from the same Kaggle sources + 36 real photos pulled from Wikimedia Commons |
| Armyworm | 67 | 30 + 37 from Commons |
| Bollworm | 74 | 31 + 43 from Commons |
| Stem Borer | 40 | 33 + 7 from Commons — still the thinnest class; Commons' `Chilo partellus`/`Busseola fusca`/`Sesamia` categories turned out sparse |

Two data-quality issues were found and fixed before training on this
data:
- **Duplicate-file padding**: those same four classes originally claimed
  100–120 images each, but 75–90% of each was the *same* underlying
  photo, re-saved under Windows' "- Copy (2)", "- Copy (3)" auto-rename.
  Left in place, `image_dataset_from_directory`'s random train/validation
  split would have scattered duplicates of one photo across both splits,
  letting the model "validate" against images it had already memorized
  byte-for-byte — an inflated, meaningless accuracy number. Deduplicated
  by content hash before training; real, replacement images were then
  pulled from Wikimedia Commons (via its category API, no login needed)
  to partially make up the shortfall.
- **Non-photo contamination**: Commons categories mix real photographs
  with historical illustrations, scientific diagrams, and even unrelated
  images that inherit a species category incorrectly (two protein
  crystal-structure renders showed up under `Helicoverpa armigera` and
  `Spodoptera frugiperda` simply because those proteins were first
  isolated from those insects). Filtering by filename keywords alone
  missed a 1928 watercolor of an adult moth; the reliable signal turned
  out to be each file's *own* Commons categories (`Category:Paintings in
  Te Papa`, `Category:Illustrations by ...`, `Category:Insect life cycle
  diagrams`, etc.) — 12 of the 135 downloaded candidates were caught and
  removed this way.

A 14th class, **Not Plant**, is assembled automatically rather than by
hand — `python train/fetch_negatives.py` pulls people, hands, furniture,
walls, floors, animals, food, roads, screens and tools from curated
Wikimedia Commons categories, rejects anything strongly green-dominant (a
bicycle photographed against a hedge would otherwise teach the model that
foliage means "not a plant"), and synthesises dim/soft-focus variants,
because a legible-but-dark indoor photo is precisely the case that used to
come back as "Healthy Leaf".

If you want to improve accuracy further, **Stem Borer** is the class
most worth growing first (40 images, well under the 150+ recommendation)
— see [`data/dataset/README.md`](data/dataset/README.md) for sources,
then rerun `python train/train_model.py`.

### A bug this training run surfaced and fixed

`model/classifier.py` was applying softmax to the model's output
unconditionally — but `train/train_model.py`'s exported model already
ends in a `Dense(..., activation="softmax")` layer, so its raw output
*is already* a probability distribution. Softmaxing an already-softmaxed
output over-smooths it toward uniform: a genuine `[0.97, 0.01, ...]`
prediction was coming out as roughly `[0.19, 0.07, ...]`. The classifier
still picked the right label (softmax doesn't change the argmax), but
every displayed confidence number was badly deflated — silently pushing
every single real prediction below `screens/result_screen.py`'s 60%
"confident" threshold, so the app would have shown "Uncertain" on every
correct classification. Fixed in `model/classifier.py` to only apply
softmax when the raw output doesn't already look like a probability
distribution.

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
  **Remove** on a history row deletes that row and the photo behind it;
  **Clear History** does the same for every row. Both ask first, and both
  only ever delete files inside the app's own `captures/` folder — a photo
  you picked through **Upload** is copied in before it is recorded, so your
  original in the gallery is never touched. The training copy under
  `collected_data/` is a separate file and is also left alone.
- **Collected dataset**: `collected_data/<ClassName>/` — every image you
  capture is automatically filed by predicted class. If you tap
  **"flag as new/unknown organism"** on a result, that image goes into
  `collected_data/_unclassified_new/` instead, ready for you to review
  and potentially fold into your next training round as a 14th class.

## Real-device issues found and fixed

The build succeeding and running the app on an actual phone are two
different milestones. These only showed up during real on-device testing.

### Upload always said "No image selected"

Android's gallery/Photos picker hands back a `content://` URI, not a
filesystem path, so `choose_from_gallery()`'s `os.path.exists(filepath)`
check could never be true. `plyer`'s Android filechooser tries to resolve
such URIs itself, but only for a few known content-provider authorities and
via a `_data` column that Android 10+'s Scoped Storage leaves null for most
providers — including the system Photo Picker most phones now show by
default. Fixed by bypassing `plyer` on Android: `home_screen.py` launches
the picker `Intent` directly and reads the result through
`ContentResolver.openFileDescriptor(...).detachFd()` into a plain OS file
descriptor, which `os.fdopen()` reads natively.

### The camera preview needed a rotation — and nothing else

Kivy's Android camera provider does **no** orientation handling at all —
in Kivy's own `camera_android.py` the call that would fix it is commented
out (`# self._android_camera.setDisplayOrientation()`). Phone sensors are
mounted landscape and report `CameraInfo.orientation = 90` on almost every
back camera, so Kivy passes the raw sensor frame straight through and a
portrait app shows a sideways preview. That is why the phone had to be
physically turned to frame a leaf.

That rotation is the *whole* fault, and getting there took three attempts,
because the first two both added a **reflection** on top of it:

1. Rotate a hardcoded 180°. A 180° turn is a vertical flip **plus** a
   horizontal one, so it left the sideways sensor uncorrected *and*
   mirrored the frame.
2. Derive the rotation properly from the sensor, but also un-flip the frame
   top-to-bottom, on the theory that Kivy's Fbo hands it back inverted. It
   does not: the fragment shader in `camera_android.py` computes a flipped
   `coord` and then never uses it — it samples plain `tex_coord0` — and the
   Fbo texture and the widget drawing it share GL's bottom-left row
   convention, so the two cancel. A lone vertical flip added to a rotation
   is, once again, a reflection.

**Why this took two goes.** A mirrored preview is invisible in a still
frame. It looks like a perfectly ordinary photo; nothing is wrong until you
*move*, and then panning the phone left slides the scene right and you
cannot aim it at anything. Reflections also compose treacherously — a
vertical flip followed by a quarter turn is pixel-for-pixel identical to a
horizontal flip — so a wrong flip on *any* axis anywhere in the chain
reaches the user as "the camera is mirrored", which tells you nothing about
where it came from.

`utils/camera_transform.py` now rotates, and only rotates:

```
upright = rotate_cw( θ, preview_frame )
```

where θ comes from `CameraInfo.orientation` combined with the current
display rotation, per Android's documented formula. The live preview
applies this as a canvas transform (free, on the GPU) and `capture_image()`
applies the identical transform to the texture bytes with Pillow — so the
photo handed to the classifier is always oriented exactly like the preview
it was framed in.

`tests/test_camera_transform.py` forward-models the whole pipeline and
asserts the round-trip is exact for all four rotations. It also asserts the
*parity* directly, which is the check that would have caught both earlier
attempts: whatever rotation is in force, the output must be one of the four
rotations of the input and none of its four reflections.

Because some vendor ROMs misreport sensor orientation and no one can test
every handset, the computed angle is only a **default**. Two buttons on the
camera preview persist a per-device correction (`utils/app_settings.py`),
so anything left over is a one-tap fix that survives restarts:

- **Rotate** — adds a 90° offset, for a preview that is sideways or upside
  down.
- **Flip** — mirrors the preview left-to-right, for a preview that runs
  backwards when you pan the phone.

### "Uncertain, closest guess: Beetle" for photos that were not plants

This was **not** a symptom of the orientation bug, as previously assumed. It
was the model doing exactly what a closed-set softmax must do. Measured on
the previous build:

| Input | Prediction |
|---|---|
| pure black frame | Healthy Leaf, 97.4% |
| flat gray wall | Healthy Leaf, 98.7% |
| random noise | Healthy Leaf, 99.1% |
| skin tone | Healthy Leaf, 99.3% |
| photo of a person in a dark room | Healthy Leaf, 62.8% |

Every one of those cleared the old 0.60 confidence threshold, so raising
the threshold was never going to help — the model was not unsure, it was
confidently wrong, and `Healthy Leaf` was acting as a sink for everything
unfamiliar. See **"Knowing when not to answer"** below for the fix.

## Knowing when not to answer

Telling a farmer their crop is healthy when the camera was pointing at a
wall is the worst failure this app can have, so refusing to answer is
treated as a first-class outcome. Three independent layers have to agree
before a diagnosis is shown:

1. **Quality gate** — `utils/image_quality.py`, runs before the network.
   Rejects frames that are too dark, blown out, blurred or featureless,
   using only numpy + Pillow (OpenCV is not in the Android build). Every
   threshold is calibrated against this project's own data — run
   `python train/calibrate_thresholds.py` to see the metric distributions
   and the false-reject rate.

2. **A "Not Plant" class** — the model is trained on a 14th class of
   people, hands, furniture, walls, floors, animals, food, roads, screens
   and dim indoor shots, so it can say "this is not a plant" directly
   instead of being forced to name a disease. Populate it with
   `python train/fetch_negatives.py`.

3. **Embedding distance (open-set check)** — the negative class can only
   cover things someone thought to include. So the model also exports the
   1280-d feature vector behind its prediction, and `model/ood_stats.json`
   records the average vector of each plant class. A photo whose features
   sit far from every class centroid is refused even when the softmax was
   confident and the negative class missed it. Both outputs come from one
   forward pass, so this costs no extra inference time on the phone.

Only if all three pass is a result shown as a diagnosis; a small margin
between the top two classes downgrades it to *uncertain*. `predict()`
returns a `status` (`ok` / `uncertain` / `not_plant` / `unusable`) and the
UI branches on that, never on the confidence number. A refusal shows no
pest name and no treatment advice — the guess was the harm, not the wording
around it.

### Verifying it

Ordinary validation accuracy cannot catch this class of bug: the previous
model scored 90% while also calling a black frame a healthy leaf, because
no non-plant image was ever in the validation set.

```bash
python train/evaluate_rejection.py --fetch
```

This downloads a set of negatives from Wikimedia Commons categories
**deliberately disjoint** from the ones used for training (into
`train/eval_negatives/`, outside `data/dataset/`, so training can never see
them) and then measures two rates with explicit pass/fail budgets:

- **sensitivity** — real plant photos must still be diagnosed
- **specificity** — non-plant photos must be refused

Measured on the current model:

| Check | Result | Budget |
|---|---|---|
| Real plant photos wrongly refused | **5.2%** | ≤ 15% |
| Unseen non-plant photos refused (153 held-out images) | **98.0%** | ≥ 85% |
| Synthetic junk frames given a diagnosis | **0.0%** | ≤ 2% |
| `Not Plant` recall on the validation split | **98.4%** | — |

The same non-leaf inputs, before and after:

| Input | Before | After |
|---|---|---|
| pure black frame | Healthy Leaf 97% | Too dark to identify |
| gray wall | Healthy Leaf 99% | Nothing clear in the frame |
| skin tone | Healthy Leaf 99% | Nothing clear in the frame |
| random noise | Healthy Leaf 99% | No plant or leaf detected |
| wood table | Healthy Leaf 99% | No plant or leaf detected |
| photo of a person in a dark room | Healthy Leaf 63% | No plant or leaf detected |

### What it costs

Refusing has a price, and it is worth stating honestly. Over 390 real
plant photos:

| | Old build | New build |
|---|---|---|
| Names the correct pest (refusals counted as misses) | 90.3% | 86.7% |
| Willing to give a diagnosis at all | 100% | 95.1% |
| **Correct when it does answer** | **90.3%** | **91.2%** |

So the drop in raw top-1 is entirely the app declining to answer on ~5% of
photos — mostly genuinely dark or blurred ones. When it does answer it is
slightly *more* accurate than before, and it no longer answers at all when
there is no plant in the picture.

Overall validation accuracy also moved from 90.1% to 84.6%, but the two
numbers are not comparable: the task changed from 13 classes to 14, and the
new one ("Not Plant", 98.4% recall) is a far larger and more varied class
than any pest. Per-class accuracy on the pest classes is essentially
unchanged — the weak ones (Aphids, Armyworm, Stem Borer) were weak before
and are weak for the same reason, too few training images.

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
- `kivy==2.3.1` (not `2.3.0`) — `2.3.0`'s Cython-generated C code calls
  CPython internal APIs (`_PyUnicode_FastCopyCharacters`,
  `_PyDict_SetItem_KnownHash`, etc.) whose signatures changed in Python
  3.13, producing compile errors across most of Kivy's core (`_event.c`,
  `_window_sdl2.c`, `vertex_instructions.c`, ...). Kivy 2.3.1 explicitly
  added Python 3.13 support; `requirements-desktop.txt` bumped to match.
- `local_recipes/requests/` — a hand-written p4a recipe pinned to
  `2.25.1`. We don't use `requests` directly; it's a transitive
  dependency of Kivy's own requirements (Kivy places no version
  constraint on it), and has no built-in p4a recipe. An unpinned
  `requests` resolves to a recent release depending on
  `charset-normalizer>=~3.4`, which ships an optional mypyc-compiled
  wheel per platform (including Android). p4a's generic fallback for such
  "extra pure Python" packages (`pythonforandroid/build.py`'s
  `run_pymodules_install`) has two confirmed bugs that combine badly with
  that: it silently drops user version pins for anything without a
  dedicated recipe (confirmed two ways — pinning `charset-normalizer`
  directly did nothing, and even giving *it* a local recipe didn't help,
  since this generic resolver independently re-satisfies whatever floor
  `requests` declares regardless of what's already installed), and its
  final `pip install --target ...` call omits the `--platform`/
  `--python-version` overrides it used moments earlier to *resolve*
  packages, so it correctly rejects the Android-tagged wheel it just
  picked for itself (`ERROR: ...whl is not a supported wheel on this
  platform`). `requests==2.25.1` (Dec 2020, confirmed via PyPI metadata)
  predates requests' switch to `charset-normalizer` entirely — it depends
  only on `chardet`/`idna`/`urllib3`/`certifi`, all universal "none-any"
  wheels with no platform-specific-wheel problem. Giving `requests`
  itself a real recipe (version pins ARE respected for recipe-based
  packages, as seen throughout this project) removes `charset-normalizer`
  from the dependency graph entirely, rather than continuing to chase
  which version of it might satisfy every constraint at once.
- `local_recipes/kivy/` — giving `requests` its own recipe (above) fixed
  the `charset-normalizer` wheel error, but immediately hit a new one:
  `AssertionError` at p4a's `toolchain.py`
  (`assert set(build_order).intersection(set(python_modules)) == set()`).
  Root cause, traced into p4a's actual source
  (`pythonforandroid/graph.py`'s `get_recipe_order_and_bootstrap` and
  `pythonforandroid/recipes/kivy/__init__.py`): p4a's built-in Kivy
  recipe hardcodes
  `python_depends = ['certifi', 'chardet', 'idna', 'requests', 'urllib3', 'filetype']`
  — a flat list of pip-installed names, unconditionally merged into the
  build's `python_modules` set whenever the Kivy recipe resolves, with no
  check for whether any of those names *also* has its own dedicated
  recipe elsewhere in the same build. Once `requests` had a dedicated
  recipe (previous entry above), it existed simultaneously in
  `build_order` (via its own recipe) and `python_modules` (via Kivy's
  hardcoded list) — tripping the assertion. There's no per-package way to
  tell p4a "this python_depends entry is actually satisfied by a
  recipe"; the only lever is to change what Kivy's recipe declares, and
  since p4a's local-recipes directory is checked *before* its own
  built-in `pythonforandroid/recipes/` (confirmed in `recipe.py`'s
  `Recipe.get_recipe`/`recipe_dirs`), a same-named recipe in
  `local_recipes/` fully shadows the built-in one. `local_recipes/kivy/`
  is an unmodified copy of upstream's `kivy==2.3.1` recipe (`__init__.py`
  + its 3 `.patch` files, pulled directly from the `p4a.commit` tag
  pinned above) with exactly one change: `requests` removed from
  `python_depends`, since our own `local_recipes/requests/` recipe now
  provides it instead.
- `Set up JDK 17` step in `.github/workflows/build.yml` (using
  `actions/setup-java@v4`) — with the two entries above fixed, the build
  got all the way to its final stage (`gradlew clean assembleDebug`) and
  failed instantly there with `Android Gradle plugin requires Java 17 to
  run. You are currently using Java 11.` `ubuntu-22.04` runners ship
  several JDKs side by side (11, 17, 21, ...) with **JDK 11 as the
  ambient default** — `sudo apt-get install openjdk-17-jdk` installs a
  second JDK alongside it but doesn't change which one `javac`/
  `JAVA_HOME` resolve to by default, confirmed directly in the log
  (p4a's own "Search for Java compiler" step found
  `/usr/lib/jvm/temurin-11-jdk-amd64/bin/javac`, and that JDK 11 stayed
  the active one through every later step, including the final Gradle
  invocation). Replaced the apt install with an explicit
  `actions/setup-java@v4` step (`distribution: temurin, java-version:
  '17'`), which exports `JAVA_HOME` and prepends JDK 17's `bin/` to
  `PATH` for every subsequent step in the job — including buildozer's
  own subprocesses — which installing a second JDK via apt alone
  doesn't do.
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

- **`HTTP Error 502: Bad Gateway` downloading freetype**, on a tree with no
  build-related changes at all. python-for-android's freetype recipe
  hardcodes `download.savannah.gnu.org`, and Savannah goes down for tens of
  minutes at a time; p4a allows a failed download only four retries across
  ~15 seconds and then aborts the entire build. Two consecutive builds died
  this way, both at ~4 minutes — long before any app code was compiled, which
  is the tell: a real code error cannot fail during dependency *download*.
  Fixed with two steps in `.github/workflows/build.yml`:
  - **`Prefetch freetype from a working mirror`** downloads the identical
    tarball from SourceForge (which the recipe's own docstring points at) or
    the nongnu mirror network, and drops it into p4a's packages directory
    **together with its `.mark-` marker file**. That marker is not optional,
    and getting it wrong cost a third red build: `Recipe.download()` accepts
    a pre-existing tarball only when both are present, and treats one without
    a marker as an interrupted download —

    ```python
    if exists(filename) and isfile(filename):
        if not exists(marker_filename):
            shprint(sh.rm, filename)      # deletes it, then re-downloads
        else:
            ...verify digests...
            do_download = False
    ```

    — so the first version of this step watched p4a delete the file it had
    just seeded and go straight back to the dead mirror. Seeding both needs
    no recipe override and no patched URL.

    The freetype recipe declares **no checksum**, so p4a would not notice a
    substituted file — the SHA-256 is therefore pinned in the workflow and
    verified before the file is trusted. If all mirrors are down the step
    warns and does nothing, rather than becoming a new way for the build to
    fail.
  - **A retry loop around buildozer**, which retries *only* when the log
    names a transport fault. Retrying everything would rebuild a genuine
    compile error three times over, turning a 4-minute red build into a
    40-minute one with the real message buried under two repeats.

  `tests/test_workflow.sh` extracts both `run:` blocks straight out of
  `build.yml` and exercises them against stubbed `curl`/`buildozer` — mirror
  fallback, a rejected digest, a total outage, a genuine build error that
  must *not* be retried. Worth having because the APK build is a 20-minute
  round trip, so a typo in a shell step otherwise costs 20 minutes to find.

  Two of its assertions are worth calling out, because both guard mistakes
  that were actually made here:

  - The prefetch checks go through `p4a_verdict()`, a transcription of
    `Recipe.download()`'s acceptance rule, rather than merely checking that
    a file landed in the right directory. The first version of the test did
    the latter and passed while the real build still re-downloaded — proving
    the script did what it was told, which says nothing about whether the
    tool downstream accepts the result. Deleting the `touch` from the
    workflow now fails six assertions.
  - The retry loop must **not** use `set -o pipefail`, because
    `yes | buildozer` leaves `yes` dying of `SIGPIPE`, and pipefail would
    read that as a failed build on every *successful* run.

If the GitHub Actions build fails on something else entirely, paste the
failed step's log (Actions tab → the failed run → the red "Build APK with
Buildozer" step) and it can be diagnosed the same way these were: search
backward from the "Command failed:" line for the actual Python traceback
or compiler error, which is usually much further up than the final error
dump.
