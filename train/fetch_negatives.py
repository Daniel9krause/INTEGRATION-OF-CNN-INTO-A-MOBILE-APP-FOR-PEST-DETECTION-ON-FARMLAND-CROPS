"""
fetch_negatives.py
-------------------
Downloads a "Not Plant" class for the FDCS dataset from Wikimedia Commons.

WHY THIS CLASS EXISTS
---------------------
A 13-class softmax has to put every photo into one of its 13 boxes. It has
never seen a person, a table, a wall or a phone screen, so those get forced
into whichever class sits closest to the network's default response - in
this model that was "Healthy Leaf", returned at 97%+ confidence for a pure
black frame, a gray wall, skin tone and random noise alike. Raising the
confidence threshold cannot fix that, because the model is genuinely (and
wrongly) confident.

The fix is to teach the model what "not a plant" looks like, by giving it a
14th class full of the things a farmer's phone actually points at by
accident: people, hands, furniture, walls, floors, animals, food, tools,
vehicles, sky, soil, packaging and screens.

Run:
    python train/fetch_negatives.py                 # ~600 images
    python train/fetch_negatives.py --per-source 25 # more per category

Then retrain:
    python train/train_model.py

Images land in data/dataset/Not Plant/ alongside the other class folders,
so train_model.py picks them up with no further changes. Re-running is safe
and resumable: files are content-hashed, so already-downloaded images are
skipped rather than duplicated.

Everything here is downloaded from Wikimedia Commons (the same source as
the rest of this dataset), at thumbnail resolution, with a descriptive
User-Agent and rate limiting, per the Wikimedia API etiquette guidelines.
"""

import argparse
import hashlib
import io
import json
import os
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "data", "dataset", "Not Plant")

API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = ("FDCS-farmland-detect/1.0 (student capstone project; "
              "dataset assembly for offline pest classifier)")
THUMB_WIDTH = 512
REQUEST_DELAY = 0.6          # be polite to the API
MAX_RETRIES = 4

# Saved at 384px: bigger than the model's 224px input (so RandomZoom has
# real pixels to work with) without bloating the repo.
STORE_SIZE = 384

# Commons categories, chosen to cover what a phone camera actually sees
# when it is NOT pointed at a leaf or an insect. Curated categories beat
# free-text search here - a search for "table" returns Old Master still
# lifes full of fruit and foliage, which would poison the negative class.
CATEGORIES = [
    # People - by far the most common accidental subject (selfies, hands)
    "Human faces", "Portrait photographs of men", "Portrait photographs of women",
    "Human hands", "Human arms", "Human legs", "People sitting",
    # Indoor surfaces and furniture
    "Tables", "Chairs", "Wooden floors", "Tiled floors", "Ceilings",
    "Interior of houses", "Doors", "Windows", "Staircases", "Beds",
    # Walls and building materials
    "Brick walls", "Concrete walls", "Plastered walls", "Corrugated iron",
    # Electronics and everyday objects
    "Smartphones", "Laptops", "Computer keyboards", "Television sets",
    "Plastic bottles", "Buckets", "Cutlery", "Cups", "Plates",
    # Textiles, paper, packaging
    "Shoes", "Clothing", "Textiles", "Books", "Handwriting", "Cardboard boxes",
    # Vehicles, roads and outdoor built environment
    "Cars", "Motorcycles", "Bicycles", "Asphalt", "Roads", "Roofs",
    # Animals a farmer may well photograph instead of a pest
    "Dogs", "Domestic cats", "Cattle", "Goats", "Chickens", "Birds",
    # Ground, sky, water - the classic "pointed at nothing" frames
    "Sand", "Gravel", "Soil", "Rocks", "Clouds", "Blue sky", "Water",
    # Food, tools, fire
    "Bread", "Cooked rice", "Meat", "Hammers", "Hand tools", "Fire",
]

# Free-text searches for a few concepts with no clean single category.
SEARCHES = [
    "dark room interior night", "dimly lit room", "person selfie indoors",
    "empty wall paint", "wooden desk surface", "concrete floor texture",
]


def _api_get(params):
    """GET the Commons API with retry/backoff on rate limiting."""
    url = API + "?" + urllib.parse.urlencode(params)
    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=40) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and attempt < MAX_RETRIES - 1:
                wait = 2 ** (attempt + 1)
                print(f"      rate limited, backing off {wait}s ...")
                time.sleep(wait)
                continue
            raise
        except Exception:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
                continue
            raise
    return {}


def _pages(data):
    return list((data.get("query") or {}).get("pages", {}).values())


def list_category(name, limit):
    return _pages(_api_get({
        "action": "query", "format": "json",
        "generator": "categorymembers", "gcmtitle": f"Category:{name}",
        "gcmtype": "file", "gcmlimit": str(min(limit * 3, 200)),
        "prop": "imageinfo", "iiprop": "url|size|mime", "iiurlwidth": str(THUMB_WIDTH),
    }))


def list_search(term, limit):
    return _pages(_api_get({
        "action": "query", "format": "json",
        "generator": "search", "gsrsearch": f"filetype:bitmap {term}",
        "gsrnamespace": "6", "gsrlimit": str(min(limit * 3, 100)),
        "prop": "imageinfo", "iiprop": "url|size|mime", "iiurlwidth": str(THUMB_WIDTH),
    }))


def _looks_like_vegetation(img):
    """Reject anything strongly green-dominant. Commons categories are
    curated but not perfect - a photo of a bicycle leaning on a hedge would
    teach the model that foliage means 'Not Plant', which is exactly
    backwards. Cheap green-dominance test, deliberately conservative."""
    a = np.asarray(img.convert("RGB").resize((96, 96)), dtype=np.int16)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    green = (g > r + 12) & (g > b + 12) & (g > 40)
    return float(green.mean()) > 0.30


def _download(url):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def _existing_hashes(out_dir):
    seen = set()
    if not os.path.isdir(out_dir):
        return seen
    for f in os.listdir(out_dir):
        p = os.path.join(out_dir, f)
        if os.path.isfile(p):
            with open(p, "rb") as fh:
                seen.add(hashlib.sha1(fh.read()).hexdigest())
    return seen


def harvest(sources, per_source, out_dir, seen_hashes):
    saved = 0
    for kind, name in sources:
        print(f"  [{kind}] {name}")
        try:
            pages = list_category(name, per_source) if kind == "cat" \
                else list_search(name, per_source)
        except Exception as e:
            print(f"      skipped ({e})")
            continue
        time.sleep(REQUEST_DELAY)

        got = 0
        for page in pages:
            if got >= per_source:
                break
            info = (page.get("imageinfo") or [{}])[0]
            mime = info.get("mime", "")
            if not mime.startswith("image/") or mime in ("image/svg+xml", "image/gif"):
                continue
            if info.get("width", 0) < 200 or info.get("height", 0) < 200:
                continue
            url = info.get("thumburl") or info.get("url")
            if not url:
                continue

            try:
                raw = _download(url)
            except Exception:
                continue

            digest = hashlib.sha1(raw).hexdigest()
            if digest in seen_hashes:
                continue

            try:
                img = Image.open(io.BytesIO(raw)).convert("RGB")
            except Exception:
                continue
            if _looks_like_vegetation(img):
                continue

            img.thumbnail((STORE_SIZE, STORE_SIZE), Image.LANCZOS)
            slug = "".join(c if c.isalnum() else "_" for c in name.lower())[:24]
            img.save(os.path.join(out_dir, f"{slug}_{digest[:10]}.jpg"),
                     "JPEG", quality=88)
            seen_hashes.add(digest)
            got += 1
            saved += 1
            time.sleep(0.05)

        print(f"      kept {got}")
    return saved


def add_lowlight_variants(out_dir, seen_hashes, count):
    """Darkened / blurred copies of the negatives.

    The quality gate in utils/image_quality.py rejects frames that are
    physically unusable, but a dim-but-legible indoor photo passes it -
    that is exactly the "photo of a person in a dark room came back as
    Healthy Leaf" case from the field. Those images have to be recognisable
    as Not Plant by the model itself, so the negative class needs dim and
    softly-focused examples, not just well-lit stock photography.
    """
    originals = [f for f in os.listdir(out_dir)
                 if f.lower().endswith(".jpg") and not f.startswith("lowlight_")]
    if not originals:
        return 0
    random.shuffle(originals)
    made = 0
    for name in originals:
        if made >= count:
            break
        try:
            img = Image.open(os.path.join(out_dir, name)).convert("RGB")
        except Exception:
            continue
        factor = random.uniform(0.16, 0.42)
        var = ImageEnhance.Brightness(img).enhance(factor)
        var = ImageEnhance.Contrast(var).enhance(random.uniform(0.6, 0.95))
        if random.random() < 0.45:
            var = var.filter(ImageFilter.GaussianBlur(random.uniform(0.6, 1.8)))
        var.save(os.path.join(out_dir, f"lowlight_{made:04d}_{name}"),
                 "JPEG", quality=86)
        made += 1
    return made


def main():
    ap = argparse.ArgumentParser(description="Fetch the 'Not Plant' negative class.")
    ap.add_argument("--per-source", type=int, default=9,
                    help="images to keep per Commons category/search (default 9)")
    ap.add_argument("--lowlight", type=int, default=110,
                    help="dim/blurred variants to synthesise (default 110)")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    seen = _existing_hashes(OUT_DIR)
    print(f"Output: {OUT_DIR}")
    print(f"Already present: {len(seen)} images\n")

    sources = [("cat", c) for c in CATEGORIES] + [("search", s) for s in SEARCHES]
    random.seed(7)

    saved = harvest(sources, args.per_source, OUT_DIR, seen)
    print(f"\nDownloaded {saved} new images.")

    made = add_lowlight_variants(OUT_DIR, seen, args.lowlight)
    print(f"Synthesised {made} low-light / soft-focus variants.")

    total = len([f for f in os.listdir(OUT_DIR) if f.lower().endswith(".jpg")])
    print(f"\n'Not Plant' class now holds {total} images.")
    if total < 250:
        print("That is on the low side - re-run with a larger --per-source "
              "for a more robust negative class.")
    print("\nNext: make sure 'Not Plant' is listed in model/labels.txt, then run"
          "\n    python train/train_model.py")


if __name__ == "__main__":
    sys.exit(main())
