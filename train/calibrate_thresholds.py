"""
calibrate_thresholds.py
------------------------
Prints the distribution of utils/image_quality's metrics over three
populations so the thresholds in that module are grounded in this
project's real data instead of guessed:

  GOOD  - data/dataset/**  (real plant/pest photos; the "Not Plant" class is
          excluded - it is what the app is meant to reject)
  FIELD - collected_data/** (real photos taken on the phone in the field)
  BAD   - synthetic unusable frames (black, dark noise, flat wall, blown
          out, heavily blurred) that MUST be rejected

Run:  python train/calibrate_thresholds.py

For each metric it reports the good-image percentiles next to the bad-image
values, then re-runs assess() over everything and reports the false-reject
rate on good images and the catch rate on bad ones. If you ever retune a
threshold, run this again - a false-reject rate above a couple of percent
means the gate is too aggressive and farmers will be told to retake photos
that were fine.
"""

import os
import random
import sys

import numpy as np
from PIL import Image, ImageFilter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from model.classifier import NEGATIVE_LABEL  # noqa: E402
from utils import image_quality as iq  # noqa: E402

DATASET_DIR = os.path.join(BASE_DIR, "data", "dataset")
COLLECTED_DIR = os.path.join(BASE_DIR, "collected_data")
METRICS = ["brightness", "p75", "dynamic_range", "contrast", "sharpness", "edge_density"]
SAMPLE_PER_CLASS = 40


def _iter_images(root, per_folder=None, exclude=()):
    if not os.path.isdir(root):
        return
    for folder in sorted(os.listdir(root)):
        d = os.path.join(root, folder)
        if not os.path.isdir(d) or folder in exclude:
            continue
        files = [f for f in os.listdir(d)
                 if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
        if per_folder and len(files) > per_folder:
            files = random.sample(files, per_folder)
        for f in files:
            yield os.path.join(d, f)


def _synthetic_bad():
    """Frames that are unusable by construction - the gate must reject
    every one of these."""
    rng = np.random.default_rng(0)
    out = {}
    out["pure black"] = np.zeros((480, 640, 3), np.uint8)
    out["night noise"] = (rng.random((480, 640, 3)) * 18).astype(np.uint8)
    out["dim room"] = (rng.random((480, 640, 3)) * 55).astype(np.uint8)
    out["flat gray wall"] = np.full((480, 640, 3), 128, np.uint8)
    out["flat white"] = np.full((480, 640, 3), 252, np.uint8)
    gradient = np.tile(np.linspace(90, 200, 640, dtype=np.float32), (480, 1))
    out["clear sky gradient"] = np.stack([gradient * 0.6, gradient * 0.8, gradient],
                                         axis=-1).astype(np.uint8)
    return {k: Image.fromarray(v) for k, v in out.items()}


def _blurred_real():
    """Real dataset photos put heavily out of focus - the realistic
    'moved the phone while tapping capture' failure."""
    paths = list(_iter_images(DATASET_DIR, per_folder=2, exclude=(NEGATIVE_LABEL,)))
    random.shuffle(paths)
    out = {}
    for p in paths[:8]:
        img = Image.open(p).convert("RGB")
        img.thumbnail((640, 640))
        out[f"blurred {os.path.basename(os.path.dirname(p))}"] = img.filter(
            ImageFilter.GaussianBlur(radius=7)
        )
    return out


def _summarise(name, rows):
    if not rows:
        print(f"  {name}: (none)")
        return
    print(f"\n  {name}  (n={len(rows)})")
    print(f"    {'metric':<15}{'min':>9}{'p01':>9}{'p05':>9}{'median':>9}{'p95':>9}{'max':>9}")
    for m in METRICS:
        v = np.array([r[m] for r in rows], dtype=float)
        print(f"    {m:<15}{v.min():>9.3f}{np.percentile(v, 1):>9.3f}"
              f"{np.percentile(v, 5):>9.3f}{np.median(v):>9.3f}"
              f"{np.percentile(v, 95):>9.3f}{v.max():>9.3f}")


def main():
    random.seed(42)

    print("Scanning training dataset ...")
    # "Not Plant" is excluded: it is the population the app is supposed to
    # reject, and fetch_negatives.py deliberately darkens some of it. Counting
    # those as false rejects would make the gate look broken and invite
    # loosening a threshold that is doing exactly its job.
    good_paths = list(_iter_images(DATASET_DIR, per_folder=SAMPLE_PER_CLASS,
                                   exclude=(NEGATIVE_LABEL,)))
    good = [iq.measure(p) for p in good_paths]

    print("Scanning field-collected photos ...")
    field_paths = list(_iter_images(COLLECTED_DIR))
    field = [iq.measure(p) for p in field_paths]

    bad_images = _synthetic_bad()
    bad_images.update(_blurred_real())
    bad = {k: iq.measure(v) for k, v in bad_images.items()}

    print("\n=== METRIC DISTRIBUTIONS ===")
    _summarise("GOOD (training dataset)", good)
    _summarise("FIELD (collected_data)", field)

    print("\n  BAD (must be rejected)")
    print(f"    {'image':<26}" + "".join(f"{m[:9]:>11}" for m in METRICS))
    for k, m in bad.items():
        print(f"    {k:<26}" + "".join(f"{m[x]:>11.3f}" for x in METRICS))

    print("\n=== CURRENT THRESHOLDS ===")
    for n, v in [("MIN_BRIGHTNESS", iq.MIN_BRIGHTNESS),
                 ("MIN_P75_BRIGHTNESS", iq.MIN_P75_BRIGHTNESS),
                 ("MAX_BRIGHTNESS", iq.MAX_BRIGHTNESS),
                 ("MIN_DYNAMIC_RANGE", iq.MIN_DYNAMIC_RANGE),
                 ("MIN_CONTRAST", iq.MIN_CONTRAST),
                 ("MIN_SHARPNESS", iq.MIN_SHARPNESS),
                 ("MIN_EDGE_DENSITY", iq.MIN_EDGE_DENSITY)]:
        print(f"    {n:<22} {v}")

    print("\n=== GATE BEHAVIOUR ===")
    rejected = [(p, iq.assess(p)) for p in good_paths]
    bad_rejects = [(p, r) for p, r in rejected if not r["usable"]]
    print(f"    GOOD images falsely rejected: {len(bad_rejects)}/{len(good_paths)}"
          f" = {len(bad_rejects) / max(len(good_paths), 1) * 100:.1f}%")
    for p, r in bad_rejects[:12]:
        print(f"      - {os.path.basename(os.path.dirname(p))}/"
              f"{os.path.basename(p)[:34]:<34} {r['reason']}")

    if field_paths:
        f_rej = [(p, iq.assess(p)) for p in field_paths]
        print(f"\n    FIELD photos rejected: "
              f"{sum(1 for _, r in f_rej if not r['usable'])}/{len(field_paths)}")
        for p, r in f_rej:
            mark = r["reason"] if not r["usable"] else "ok"
            print(f"      - {os.path.basename(p):<26} {mark}")

    caught = sum(1 for v in bad_images.values() if not iq.assess(v)["usable"])
    print(f"\n    BAD images correctly rejected: {caught}/{len(bad_images)}")
    for k, v in bad_images.items():
        r = iq.assess(v)
        print(f"      - {k:<26} {'REJECTED: ' + r['reason'] if not r['usable'] else '*** PASSED (bad!) ***'}")


if __name__ == "__main__":
    main()
