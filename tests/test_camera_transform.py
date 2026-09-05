"""
test_camera_transform
----------------------
Verifies the camera orientation maths without needing an Android device.

The bug this guards against is subtle and was shipped TWICE. Both times the
correction added a REFLECTION to a problem that is only ever a rotation:

  - the first fix rotated a hardcoded 180 degrees (= a vertical flip AND a
    horizontal one), leaving the sideways sensor uncorrected as well;
  - the second derived the rotation correctly but also flipped the frame
    top-to-bottom, on the belief that Kivy's Fbo inverts it.

Neither is visible in a still photo. A mirrored preview looks like a
perfectly ordinary picture right up until you move the phone and the scene
slides the wrong way - which is exactly why it survived a release, and why
the parity is asserted here directly rather than inferred.

Three properties are checked:

  1. compute_rotation() matches Android's documented formula for the
     sensor/display combinations real phones report.

  2. The pipeline never mirrors. Whatever rotation is in force, the output
     must be a pure rotation of the input - something you could reach by
     turning the original - and never one of its reflections.

  3. The preview transform and the capture transform agree. If they ever
     diverge, the farmer frames a leaf in the preview and the classifier
     receives a differently-oriented image, and nothing in the UI would
     reveal it.

Property 3 is tested by forward-modelling the whole pipeline. Starting from
a known upright image we synthesise what the sensor would produce and what
bytes texture.pixels would hand back - then assert that texture_to_pil()
recovers the original upright image exactly.

Run:  python tests/test_camera_transform.py
"""

import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import camera_transform as ct  # noqa: E402

PASSED = 0
FAILED = 0


def check(name, condition, detail=""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  PASS  {name}")
    else:
        FAILED += 1
        print(f"  FAIL  {name}  {detail}")


class FakeTexture:
    """Stands in for a Kivy camera texture.

    Kivy stores texture rows bottom-first, so `pixels` is the displayed
    image flipped top-to-bottom - reproduced faithfully here, because that
    flip is exactly the kind of detail an implementation can get wrong in a
    way that only shows up on a phone. Note it is a fact about how the bytes
    are handed over, not about the frame's orientation: undoing it is not
    the same thing as the `vflip` correction stage, and conflating the two
    is what mirrored the preview.
    """

    def __init__(self, displayed_rgb, colorfmt="bgr"):
        self.size = displayed_rgb.size
        # Deliberately a NON-rgba colorfmt: Kivy's Texture.pixels always
        # returns RGBA whatever the texture's own format is, so anything
        # that branches on colorfmt to pick a decode mode is wrong and
        # must fail here.
        self.colorfmt = colorfmt
        rgba = displayed_rgb.convert("RGBA")
        bottom_up = rgba.transpose(Image.FLIP_TOP_BOTTOM)
        self.pixels = bottom_up.tobytes()


def make_asymmetric_image(w=64, h=48):
    """An image with no rotational or mirror symmetry, so any wrong
    transform changes the pixels. A symmetric test pattern would pass even
    with the orientation completely wrong."""
    a = np.zeros((h, w, 3), np.uint8)
    a[:, :, 0] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]   # R left->right
    a[:, :, 1] = np.linspace(0, 255, h, dtype=np.uint8)[:, None]   # G top->bottom
    a[:6, :6] = [255, 255, 255]      # unique marker, top-left
    a[-4:, -10:] = [0, 0, 255]       # different marker, bottom-right
    return Image.fromarray(a)


def rotate_cw(img, deg):
    """Rotate clockwise by a multiple of 90, as a person would see it."""
    return img.rotate(-(deg % 360), expand=True)


def test_compute_rotation():
    print("\ncompute_rotation() against Android's formula")
    # The standard case: back sensor mounted at 90, phone held upright in a
    # portrait-locked app. This is what the vast majority of handsets report,
    # and it is why the old hardcoded 180 was wrong.
    check("back sensor 90, portrait -> 90",
          ct.compute_rotation(90, 0) == 90, ct.compute_rotation(90, 0))
    check("back sensor 90, display 90 -> 0",
          ct.compute_rotation(90, 90) == 0, ct.compute_rotation(90, 90))
    check("back sensor 90, display 270 -> 180",
          ct.compute_rotation(90, 270) == 180, ct.compute_rotation(90, 270))
    check("back sensor 270, portrait -> 270",
          ct.compute_rotation(270, 0) == 270, ct.compute_rotation(270, 0))
    check("back sensor 0, portrait -> 0",
          ct.compute_rotation(0, 0) == 0, ct.compute_rotation(0, 0))
    check("front sensor 270, portrait -> 90",
          ct.compute_rotation(270, 0, front_facing=True) == 90,
          ct.compute_rotation(270, 0, front_facing=True))
    check("result is always a legal quarter turn",
          all(ct.compute_rotation(s, d) in (0, 90, 180, 270)
              for s in (0, 90, 180, 270) for d in (0, 90, 180, 270)))


def test_pipeline_never_mirrors():
    """The regression that brought the farmer back twice.

    A rotation can be worked around by turning the phone; a reflection
    cannot be worked around at all, and the user cannot even name what is
    wrong - the picture looks fine, it just refuses to be aimed. So assert
    the parity head-on: whatever the rotation, the output must be one of the
    four rotations of the input and none of its four reflections.
    """
    print("\nthe transform rotates and never reflects")
    img = make_asymmetric_image()
    rotations = [np.asarray(rotate_cw(img, d)) for d in (0, 90, 180, 270)]
    mirrored = img.transpose(Image.FLIP_LEFT_RIGHT)
    reflections = [np.asarray(rotate_cw(mirrored, d)) for d in (0, 90, 180, 270)]

    for rotation in (0, 90, 180, 270):
        out = np.asarray(ct.apply_to_pil(
            img, ct.ANDROID_PREVIEW_IS_VFLIPPED, rotation, False))
        check(f"rotation {rotation:3d} deg output is a rotation of the input",
              any(np.array_equal(out, r) for r in rotations),
              "output is not reachable by turning the original")
        check(f"rotation {rotation:3d} deg output is not mirrored",
              not any(np.array_equal(out, r) for r in reflections),
              "output is a reflection - the preview runs backwards")

    check("the Android default adds no flip of its own",
          ct.ANDROID_PREVIEW_IS_VFLIPPED is False, ct.ANDROID_PREVIEW_IS_VFLIPPED)


def test_capture_matches_preview():
    """Forward-model the Android pipeline and check we invert it exactly."""
    print("\ncapture recovers the upright image the preview showed")
    upright = make_asymmetric_image()

    for rotation in (0, 90, 180, 270):
        # 1. The sensor produces the frame that, turned `rotation` degrees
        #    clockwise, would look upright. So the sensor frame is the
        #    upright image turned back the other way.
        sensor = rotate_cw(upright, -rotation)
        # 2. Kivy hands that frame straight through - its Fbo shader samples
        #    plain tex_coord0, and the Fbo texture shares GL's row convention
        #    with the widget that draws it, so nothing is inverted on the way.
        displayed = sensor
        # 3. texture.pixels hands the rows back bottom-first.
        texture = FakeTexture(displayed)

        recovered = ct.texture_to_pil(
            texture, vflip=ct.ANDROID_PREVIEW_IS_VFLIPPED,
            rotation_cw=rotation, mirror=False,
        )
        same = np.array_equal(np.asarray(recovered), np.asarray(upright.convert("RGB")))
        check(f"rotation {rotation:3d} deg round-trips exactly", same,
              f"got {recovered.size}, expected {upright.size}")


def test_mirror_and_flip_are_distinct():
    """The precise confusion behind the original bug."""
    print("\n180 degree rotation is not a vertical flip")
    img = make_asymmetric_image()
    rot180 = rotate_cw(img, 180)
    vflip = img.transpose(Image.FLIP_TOP_BOTTOM)
    check("rotate(180) != flip_top_bottom",
          not np.array_equal(np.asarray(rot180), np.asarray(vflip)))
    check("rotate(180) == flip_top_bottom + flip_left_right",
          np.array_equal(np.asarray(rot180),
                         np.asarray(vflip.transpose(Image.FLIP_LEFT_RIGHT))))


def test_apply_to_pil_shapes():
    print("\napply_to_pil keeps the whole frame on quarter turns")
    img = make_asymmetric_image(64, 48)
    for rotation, expected in ((0, (64, 48)), (90, (48, 64)),
                               (180, (64, 48)), (270, (48, 64))):
        out = ct.apply_to_pil(img, False, rotation, False)
        check(f"rotation {rotation:3d} -> size {expected}", out.size == expected, out.size)
    check("mirror preserves size",
          ct.apply_to_pil(img, False, 0, True).size == (64, 48))
    check("mirror actually mirrors",
          not np.array_equal(np.asarray(ct.apply_to_pil(img, False, 0, True)),
                             np.asarray(img)))


def main():
    print("=" * 60)
    print("camera transform tests")
    print("=" * 60)
    test_compute_rotation()
    test_pipeline_never_mirrors()
    test_capture_matches_preview()
    test_mirror_and_flip_are_distinct()
    test_apply_to_pil_shapes()
    print("\n" + "=" * 60)
    print(f"{PASSED} passed, {FAILED} failed")
    print("=" * 60)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
