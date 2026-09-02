"""
image_quality
--------------
A cheap, dependency-light gate that runs BEFORE the neural network and
answers one question: "is this photo even usable?"

Why this exists
---------------
A softmax classifier has no way to say "I can't see anything here". Fed a
pitch-black frame it still returns a confident class - the trained FDCS
model returned "Healthy Leaf" at 97% for a pure black image, because
whichever class sits closest to the network's zero-input response absorbs
every off-manifold photo. Farmers scanning at dusk or in a dim store room
were getting "Healthy Leaf" for photos that contained no leaf at all.

So we refuse to classify images that are physically unusable, and say
exactly what is wrong so the farmer can retake the shot:

  * too dark        - shot at night / no flash / lens covered
  * too washed out  - direct sun glare, blown-out highlights
  * too blurry      - motion blur, or the phone never focused
  * featureless     - pointed at a blank wall, the sky, or a surface with
                      no structure at all

Deliberately uses only numpy + Pillow (no OpenCV): OpenCV is not in
buildozer.spec's Android requirements, so anything importing cv2 would
crash on a real phone.

Every threshold here is calibrated against this project's own data - see
train/calibrate_thresholds.py, which prints the metric distribution for
the real dataset alongside known-bad images and reports the headroom each
threshold has.
"""

import numpy as np
from PIL import Image, ImageOps

# Longest edge the metrics are computed at. Small enough to be instant on
# a low-end phone, large enough that fine leaf texture still registers.
ANALYSIS_SIZE = 256

# --- Thresholds (see module docstring; calibrated, not guessed) ----------
# Mean brightness on 0-255. Real dataset photos sit well above this; a
# genuinely unusable night shot sits below it.
MIN_BRIGHTNESS = 32.0
# A photo can have an acceptable *mean* and still be mostly black (e.g. a
# dark room with one bright lamp). Require the bulk of the frame to carry
# some signal too.
MIN_P75_BRIGHTNESS = 45.0
# Blown out: almost everything pinned near white, nothing left to see.
MAX_BRIGHTNESS = 244.0
# Spread between the dark and bright ends of the histogram. Below this the
# frame is essentially one flat tone.
MIN_DYNAMIC_RANGE = 26.0
# Global contrast (std of luma).
MIN_CONTRAST = 11.0
# Contrast-normalised sharpness: Laplacian std divided by luma std. This
# ratio is what separates "blurry" from "merely low-contrast", since a raw
# Laplacian variance drops for dark images even when they are in focus.
MIN_SHARPNESS = 0.055
# Absolute edge energy, as a backstop for smooth gradients (clear sky, a
# painted wall) that are technically sharp but contain no subject.
MIN_EDGE_DENSITY = 0.010


class QualityReport(dict):
    """Plain dict subclass so it stays trivially JSON-serialisable, with a
    couple of conveniences for call sites."""

    @property
    def usable(self):
        return self["usable"]

    @property
    def reason(self):
        return self["reason"]


def _luma(image_path_or_pil):
    """Load an image and return (grayscale float array, PIL RGB image)."""
    if isinstance(image_path_or_pil, Image.Image):
        img = image_path_or_pil
    else:
        img = Image.open(image_path_or_pil)
    # Honour the EXIF orientation tag: phone galleries store the sensor
    # frame plus a rotation flag rather than rotating the pixels, so an
    # un-rotated portrait shot would otherwise be analysed sideways.
    img = ImageOps.exif_transpose(img).convert("RGB")
    small = img.copy()
    small.thumbnail((ANALYSIS_SIZE, ANALYSIS_SIZE), Image.BILINEAR)
    gray = np.asarray(small.convert("L"), dtype=np.float32)
    return gray, img


def _laplacian(gray):
    """3x3 Laplacian via array slicing - no scipy/cv2 needed."""
    return (
        gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
        - 4.0 * gray[1:-1, 1:-1]
    )


def measure(image_path_or_pil):
    """Compute the raw quality metrics for an image. Kept separate from the
    pass/fail logic so train/calibrate_thresholds.py can print the metric
    distribution over a whole folder."""
    gray, _ = _luma(image_path_or_pil)
    if gray.size < 64:
        return {"brightness": 0.0, "p05": 0.0, "p75": 0.0, "p95": 0.0,
                "dynamic_range": 0.0, "contrast": 0.0, "sharpness": 0.0,
                "edge_density": 0.0}

    p05, p75, p95 = np.percentile(gray, [5, 75, 95])
    contrast = float(gray.std())
    lap = _laplacian(gray)
    lap_std = float(lap.std())

    return {
        "brightness": float(gray.mean()),
        "p05": float(p05),
        "p75": float(p75),
        "p95": float(p95),
        "dynamic_range": float(p95 - p05),
        "contrast": contrast,
        # Normalising by contrast makes this a focus measure rather than a
        # contrast measure - a sharp photo of a low-contrast subject still
        # scores well, a blurred photo of a high-contrast one does not.
        "sharpness": lap_std / (contrast + 1e-6),
        # Fraction of pixels sitting on a meaningful edge. Flat surfaces
        # (sky, plaster wall) score near zero however sharp the optics.
        "edge_density": float((np.abs(lap) > 6.0).mean()),
    }


def assess(image_path_or_pil):
    """
    Returns a QualityReport:
        {
          "usable": bool,
          "reason": short headline for the UI ("" when usable),
          "detail": what the farmer should actually do about it,
          "metrics": {...raw numbers, for logging / tuning...},
        }
    Checks run worst-first so the message names the dominant problem rather
    than whichever check happens to be listed first.
    """
    m = measure(image_path_or_pil)

    if m["brightness"] < MIN_BRIGHTNESS or m["p75"] < MIN_P75_BRIGHTNESS:
        return QualityReport(
            usable=False,
            reason="Too dark to identify",
            detail=(
                "This photo is too dark for the app to see anything in it. "
                "Move into daylight or switch on a light, hold the phone "
                "steady, and make sure your finger or a shadow is not "
                "covering the lens."
            ),
            metrics=m,
        )

    if m["brightness"] > MAX_BRIGHTNESS:
        return QualityReport(
            usable=False,
            reason="Too bright / washed out",
            detail=(
                "The photo is overexposed - direct sunlight or a flash has "
                "washed out the detail. Shade the leaf with your hand or your "
                "body, or step out of direct sun, then retake it."
            ),
            metrics=m,
        )

    if m["dynamic_range"] < MIN_DYNAMIC_RANGE or m["contrast"] < MIN_CONTRAST:
        return QualityReport(
            usable=False,
            reason="Nothing clear in the frame",
            detail=(
                "This looks like a flat, empty surface rather than a plant - "
                "the whole frame is one shade. Point the camera at the leaf "
                "or the pest itself and let it fill most of the picture."
            ),
            metrics=m,
        )

    if m["sharpness"] < MIN_SHARPNESS or m["edge_density"] < MIN_EDGE_DENSITY:
        return QualityReport(
            usable=False,
            reason="Too blurry",
            detail=(
                "The photo is out of focus, or was taken while moving. Hold "
                "the phone still, keep about 15-30 cm from the leaf, tap the "
                "screen to focus, then capture again."
            ),
            metrics=m,
        )

    return QualityReport(usable=True, reason="", detail="", metrics=m)
