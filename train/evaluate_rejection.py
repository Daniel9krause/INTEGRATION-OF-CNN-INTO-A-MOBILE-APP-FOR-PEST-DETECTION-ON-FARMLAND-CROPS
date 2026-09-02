"""
evaluate_rejection.py
----------------------
Proves - or disproves - that the app refuses to diagnose what it should not.

Ordinary validation accuracy cannot catch the bug this whole layer exists
to fix. The previous model scored 90% validation accuracy while also
returning "Healthy Leaf" at 97% confidence for a pitch-black frame, because
no non-plant image was ever in the validation set. So this harness measures
the two things that actually matter in the field:

    SENSITIVITY  real plant photos must still get through and be diagnosed
    SPECIFICITY  everything else must be refused

Run:
    python train/evaluate_rejection.py            # evaluate
    python train/evaluate_rejection.py --fetch    # download held-out
                                                  # negatives first, then run

THE HELD-OUT NEGATIVE SET
-------------------------
Scoring the model on data/dataset/Not Plant/ would be marking its own
homework - those images trained it. --fetch pulls a separate set from
Wikimedia Commons categories that are deliberately DISJOINT from the ones
train/fetch_negatives.py uses, into train/eval_negatives/ (which is outside
data/dataset/, so training can never see it). Umbrellas, bicycles ridden by
nobody, saucepans, sheep - things the model was never taught, standing in
for the endless list of objects a farmer's phone might point at.

Every number printed is a rate over real images, not a spot check. The
summary at the end applies explicit pass/fail budgets so a regression shows
up as FAIL rather than as a slightly worse number nobody notices.
"""

import argparse
import os
import random
import sys
from collections import Counter

import numpy as np
from PIL import Image, ImageFilter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from model.classifier import NEGATIVE_LABEL, PestClassifier  # noqa: E402

DATASET_DIR = os.path.join(BASE_DIR, "data", "dataset")
COLLECTED_DIR = os.path.join(BASE_DIR, "collected_data")
EVAL_NEG_DIR = os.path.join(BASE_DIR, "train", "eval_negatives")
SCRATCH_DIR = os.path.join(BASE_DIR, "train", "_eval_scratch")

# Categories intentionally NOT used by train/fetch_negatives.py.
HELDOUT_CATEGORIES = [
    "Umbrellas", "Handbags", "Glass bottles", "Keys", "Coins", "Wristwatches",
    "Fences", "Metal gates", "Table lamps", "Mirrors", "Towels", "Pillows",
    "Curtains", "Carpets", "Acoustic guitars", "Drum kits", "Wheelbarrows",
    "Ladders", "Ropes", "Screws", "Sponges", "Bars of soap", "Toothbrushes",
    "Cooking pots", "Frying pans", "Refrigerators", "Washing machines",
    "Electric fans", "Wall clocks", "Pocket calculators", "Headphones",
    "Photographic cameras", "Printers", "Human eyes", "Human feet",
    "Horses", "Sheep", "Pigs", "Ducks", "Aquarium fish", "Staplers",
    "Light switches", "Power sockets", "Banknotes", "Playing cards",
]

# --- Pass/fail budgets ---------------------------------------------------
# A farmer told "no plant detected" for a real leaf just retakes the photo.
# A farmer told "Healthy Leaf" for a photo of a wall may leave an actual
# infestation untreated. The budgets are asymmetric on purpose.
MAX_FALSE_REFUSAL = 0.15     # real plant photos wrongly refused
MIN_NEGATIVE_CATCH = 0.85    # non-plant photos correctly refused
MAX_CONFIDENT_ON_JUNK = 0.02  # synthetic junk given a real diagnosis


def _images_in(root, per_folder=None, exclude=()):
    out = []
    if not os.path.isdir(root):
        return out
    entries = sorted(os.listdir(root))
    folders = [e for e in entries if os.path.isdir(os.path.join(root, e))]
    if not folders:
        return [os.path.join(root, e) for e in entries
                if e.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
    for folder in folders:
        if folder in exclude:
            continue
        d = os.path.join(root, folder)
        files = [f for f in sorted(os.listdir(d))
                 if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
        if per_folder and len(files) > per_folder:
            files = random.sample(files, per_folder)
        out.extend(os.path.join(d, f) for f in files)
    return out


def _make_junk():
    """Synthetic frames that must never receive a diagnosis."""
    os.makedirs(SCRATCH_DIR, exist_ok=True)
    rng = np.random.default_rng(1)
    frames = {
        "pure black": np.zeros((480, 640, 3), np.uint8),
        "night noise": (rng.random((480, 640, 3)) * 18).astype(np.uint8),
        "dim room": (rng.random((480, 640, 3)) * 55).astype(np.uint8),
        "flat gray wall": np.full((480, 640, 3), 128, np.uint8),
        "flat white": np.full((480, 640, 3), 252, np.uint8),
        "random noise": (rng.random((480, 640, 3)) * 255).astype(np.uint8),
        "skin tone": np.full((480, 640, 3), 0, np.uint8) + np.array([222, 184, 150], np.uint8),
        "wood table": np.full((480, 640, 3), 0, np.uint8) + np.array([150, 111, 71], np.uint8),
        "blue sky": np.full((480, 640, 3), 0, np.uint8) + np.array([120, 170, 235], np.uint8),
    }
    # Structured (non-flat) junk, so the quality gate cannot take all the
    # credit - these have plenty of edges and contrast, and must be refused
    # by the model rather than by a brightness check.
    def _gray_to_rgb(mask, on, off):
        """np.repeat, not mask[..., None] - a trailing axis of length 1 is a
        single-channel image, which Pillow refuses to save as JPEG."""
        return np.repeat(np.where(mask, on, off).astype(np.uint8)[:, :, None], 3, axis=2)

    checker = np.indices((480, 640)).sum(axis=0) % 64 < 32
    frames["checkerboard"] = _gray_to_rgb(checker, 230, 25)
    frames["printed text-ish"] = _gray_to_rgb(rng.random((480, 640)) < 0.08, 20, 235)

    paths = {}
    for name, arr in frames.items():
        p = os.path.join(SCRATCH_DIR, name.replace(" ", "_") + ".jpg")
        Image.fromarray(arr).save(p, quality=92)
        paths[name] = p
    return paths


def _make_dark_plants(n=25):
    """Real leaf photos dimmed to dusk levels. These SHOULD still be
    diagnosed or at worst called uncertain - they must not be dismissed as
    'not a plant', or the app becomes useless in the early morning and late
    afternoon when field work actually happens."""
    os.makedirs(SCRATCH_DIR, exist_ok=True)
    srcs = _images_in(DATASET_DIR, per_folder=3, exclude=(NEGATIVE_LABEL,))
    random.shuffle(srcs)
    out = []
    for i, src in enumerate(srcs[:n]):
        try:
            img = Image.open(src).convert("RGB")
        except Exception:
            continue
        arr = np.asarray(img, dtype=np.float32) * 0.45
        p = os.path.join(SCRATCH_DIR, f"dusk_{i:03d}.jpg")
        Image.fromarray(arr.astype(np.uint8)).save(p, quality=92)
        out.append(p)
    return out


def fetch_heldout_negatives(per_source=4):
    """Download the disjoint evaluation negatives. Reuses the downloader in
    fetch_negatives.py so there is one implementation of the API etiquette,
    hashing and vegetation filtering."""
    from train import fetch_negatives as fn

    os.makedirs(EVAL_NEG_DIR, exist_ok=True)
    seen = fn._existing_hashes(EVAL_NEG_DIR)
    print(f"Fetching held-out negatives into {EVAL_NEG_DIR} "
          f"({len(seen)} already present)")
    sources = [("cat", c) for c in HELDOUT_CATEGORIES]
    saved = fn.harvest(sources, per_source, EVAL_NEG_DIR, seen)
    total = len([f for f in os.listdir(EVAL_NEG_DIR) if f.lower().endswith(".jpg")])
    print(f"\nDownloaded {saved} new; {total} held-out negatives available.\n")


def run_group(clf, name, paths, expect):
    """expect: 'diagnose' (status ok/uncertain) or 'refuse' (not_plant/unusable)."""
    if not paths:
        print(f"\n{name}: (no images)")
        return None

    statuses = Counter()
    wrong = []
    for p in paths:
        try:
            r = clf.predict(p)
        except Exception as e:
            statuses["error"] += 1
            wrong.append((p, f"error: {e}"))
            continue
        statuses[r["status"]] += 1
        refused = r["status"] in ("not_plant", "unusable")
        if expect == "refuse" and not refused:
            wrong.append((p, f"{r['status']}: {r['label']} {r['confidence'] * 100:.0f}%"))
        elif expect == "diagnose" and refused:
            wrong.append((p, f"{r['status']}: {r['headline']}"))

    n = len(paths)
    good = n - len(wrong)
    rate = good / n
    print(f"\n{name}  (n={n})")
    print(f"    statuses: {dict(statuses)}")
    print(f"    as expected ({expect}): {good}/{n} = {rate * 100:.1f}%")
    for p, why in wrong[:8]:
        print(f"      MISS  {os.path.basename(p)[:40]:<42} {why}")
    if len(wrong) > 8:
        print(f"      ... and {len(wrong) - 8} more")
    return rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true",
                    help="download held-out evaluation negatives first")
    ap.add_argument("--per-class", type=int, default=25,
                    help="plant images sampled per class (default 25)")
    args = ap.parse_args()

    if args.fetch:
        fetch_heldout_negatives()

    random.seed(11)
    clf = PestClassifier()
    print(f"\nModel loaded: {clf.interpreter is not None}"
          f" | negative class: {clf.has_negative_class}"
          f" | embedding OOD: {clf.centroids is not None}"
          f" (threshold {clf.ood_threshold})")

    plants = _images_in(DATASET_DIR, per_folder=args.per_class, exclude=(NEGATIVE_LABEL,))
    dark_plants = _make_dark_plants()
    train_negs = _images_in(os.path.join(DATASET_DIR, NEGATIVE_LABEL))
    random.shuffle(train_negs)
    heldout_negs = _images_in(EVAL_NEG_DIR)
    junk = _make_junk()
    field = _images_in(COLLECTED_DIR)

    print("\n" + "=" * 68)
    print("SENSITIVITY - real plant photos must still be diagnosed")
    print("=" * 68)
    plant_rate = run_group(clf, "Training-set plant photos", plants, "diagnose")
    dusk_rate = run_group(clf, "Plant photos dimmed to dusk (x0.45)", dark_plants, "diagnose")

    print("\n" + "=" * 68)
    print("SPECIFICITY - non-plant photos must be refused")
    print("=" * 68)
    run_group(clf, f"'{NEGATIVE_LABEL}' training images (seen in training)",
              train_negs[:200], "refuse")
    heldout_rate = run_group(clf, "Held-out negatives (categories never trained on)",
                             heldout_negs, "refuse")
    junk_rate = run_group(clf, "Synthetic junk frames", list(junk.values()), "refuse")

    print("\n    per-frame detail:")
    for name, p in junk.items():
        r = clf.predict(p)
        verdict = "REFUSED" if r["status"] in ("not_plant", "unusable") else "*** DIAGNOSED ***"
        print(f"      {name:<20} {verdict:<18} {r['headline']}")

    if field:
        print("\n" + "=" * 68)
        print("REAL FIELD PHOTOS taken on the phone (collected_data/)")
        print("=" * 68)
        for p in field:
            r = clf.predict(p)
            conf = f"{r['confidence'] * 100:.0f}%" if r["label"] else "-"
            print(f"    {os.path.basename(p):<26} {r['status']:<10} {r['headline'][:40]:<42} {conf}")

    print("\n" + "=" * 68)
    print("VERDICT")
    print("=" * 68)
    checks = []
    if plant_rate is not None:
        checks.append(("false refusal of real plants",
                       1 - plant_rate, MAX_FALSE_REFUSAL, "<="))
    if heldout_rate is not None:
        checks.append(("catch rate on unseen non-plants",
                       heldout_rate, MIN_NEGATIVE_CATCH, ">="))
    if junk_rate is not None:
        checks.append(("junk frames given a diagnosis",
                       1 - junk_rate, MAX_CONFIDENT_ON_JUNK, "<="))

    failed = 0
    for name, value, budget, op in checks:
        ok = value <= budget if op == "<=" else value >= budget
        failed += (not ok)
        print(f"    [{'PASS' if ok else 'FAIL'}] {name:<36} "
              f"{value * 100:5.1f}%  (budget {op} {budget * 100:.0f}%)")
    if dusk_rate is not None:
        print(f"    [info] dusk plant photos still diagnosed   {dusk_rate * 100:5.1f}%")

    print()
    if failed:
        print(f"{failed} check(s) FAILED - do not ship this model.")
    else:
        print("All checks passed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
