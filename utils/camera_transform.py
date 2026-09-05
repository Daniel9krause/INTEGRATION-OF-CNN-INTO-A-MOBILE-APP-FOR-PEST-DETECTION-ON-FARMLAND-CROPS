"""
camera_transform
-----------------
Makes the Android camera preview come out upright AND the right way round,
and guarantees that the photo we hand to the classifier is oriented exactly
like the preview the farmer framed it in.

THE PROBLEM
-----------
Kivy's Android camera provider does no orientation handling at all. In
kivy/core/camera/camera_android.py the one call that would fix it is
commented out in Kivy's own source:

    # self._android_camera.setDisplayOrientation()

Phone camera sensors are mounted landscape. A back sensor reports
orientation=90, meaning its frames need a 90 degree clockwise turn to look
upright to someone holding the phone in portrait. Kivy hands the raw sensor
frame straight to the widget, so the preview lies on its side and the farmer
has to physically turn the phone to frame a leaf.

That rotation is the whole of the problem. Two earlier fixes each added a
REFLECTION on top of it, which is a nastier class of mistake because a
reflected preview still looks like a perfectly ordinary photo - it just runs
backwards. Nothing is wrong until you move: pan the phone left and the scene
slides right, so you cannot aim it at a leaf.

  - The first rotated a hardcoded 180 degrees. A 180 degree turn is a
    vertical flip PLUS a horizontal one, so it left the sideways sensor
    uncorrected and mirrored the frame.

  - The second derived the rotation properly, but also flipped the frame
    top-to-bottom, on the theory that Kivy's Fbo hands back an upside-down
    frame. It does not. The fragment shader in camera_android.py computes a
    flipped `coord` and then never uses it - it samples plain `tex_coord0` -
    and the Fbo texture and the widget drawing it share GL's bottom-left row
    convention, so nothing is inverted. A lone vertical flip added to a
    rotation is, again, a reflection.

Reflections compose treacherously, which is why this took two goes. A
vertical flip followed by a quarter turn is pixel-for-pixel the same thing
as a horizontal flip; a wrong flip on ANY axis anywhere in the chain
surfaces to the user as "the camera is mirrored". So the only safe rule is
to count them: an upright, un-mirrored preview needs an EVEN number of
flips, and the honest answer here is zero.

THE FIX
-------
Rotate, and only rotate:

    upright = rotate_cw( theta, preview_frame )

where theta comes from Android's own CameraInfo.orientation combined with
the current display rotation - the formula Android's camera documentation
prescribes.

The flip stages stay in the pipeline rather than being deleted, but both now
default to off. Because some vendor ROMs misreport sensor orientation, and
because we cannot test every handset, the computed angle is only a DEFAULT:
the Home screen exposes a Rotate button that adds a persisted 90 degree
offset, and a Flip button that adds a persisted mirror, so any remaining
device quirk is a one-tap fix that survives restarts.

Both the live preview (via a canvas transform, free on the GPU) and the
saved capture (via Pillow, on the texture bytes) go through the same
(vflip, rotation, mirror) triple, so the two can never disagree.
"""

from kivy.graphics import PopMatrix, PushMatrix, Rotate, Scale
from kivy.uix.floatlayout import FloatLayout
from kivy.utils import platform

from utils import app_settings

# Whether the preview frame arrives flipped top-to-bottom. False: neither
# Kivy's Android provider nor the desktop CameraPreview in
# utils/camera_preview.py inverts the frame, so correcting for a flip that
# is not there is what mirrored the preview (see THE PROBLEM above). Kept as
# a named constant rather than being deleted, because it is the one knob to
# reach for if a future Kivy release starts applying the SurfaceTexture
# transform matrix - and because the tests pin the behaviour to it.
ANDROID_PREVIEW_IS_VFLIPPED = False


def compute_rotation(sensor_orientation, display_degrees, front_facing=False):
    """Android's documented preview-orientation formula, as pure arithmetic
    so it can be unit-tested without a handset (see
    tests/test_camera_transform.py).

    `sensor_orientation` is Camera.CameraInfo.orientation - how many degrees
    the sensor is mounted clockwise from the device's natural orientation
    (90 on the overwhelming majority of back cameras). `display_degrees` is
    the current display rotation, 0 while the phone is held upright in a
    portrait-locked app like this one.

    Returns the clockwise rotation to apply to a frame to make it upright.
    """
    if front_facing:
        # Front sensors are mirrored, so the display rotation adds rather
        # than subtracts, and the result is then reflected.
        return (360 - ((sensor_orientation + display_degrees) % 360)) % 360
    return (sensor_orientation - display_degrees + 360) % 360


def android_sensor_rotation(camera_index=0):
    """Clockwise degrees needed to bring this device's camera frames
    upright. Returns 0 off-Android, or a sane guess if the device refuses to
    tell us."""
    if platform != "android":
        return 0
    try:
        from jnius import autoclass

        Camera = autoclass("android.hardware.Camera")
        CameraInfo = autoclass("android.hardware.Camera$CameraInfo")
        PythonActivity = autoclass("org.kivy.android.PythonActivity")

        info = CameraInfo()
        Camera.getCameraInfo(camera_index, info)

        display = PythonActivity.mActivity.getWindowManager().getDefaultDisplay()
        degrees = {0: 0, 1: 90, 2: 180, 3: 270}.get(display.getRotation(), 0)

        rotation = compute_rotation(
            info.orientation, degrees,
            front_facing=(info.facing == CameraInfo.CAMERA_FACING_FRONT),
        )
        print(f"[camera_transform] sensor={info.orientation} display={degrees} "
              f"facing={info.facing} -> rotate {rotation} deg clockwise")
        return rotation
    except Exception as e:
        # No jnius, an unusual ROM, or the camera service not up yet.
        # 90 is the near-universal back-camera case, and a preview the user
        # can still correct with the Rotate button beats a crash.
        print(f"[camera_transform] Could not read sensor orientation: {e}")
        return 90 if platform == "android" else 0


def current_transform(camera_index=0, native_android=True):
    """The (vflip, rotation_cw, mirror) triple in force right now, sensor
    default plus any persisted user correction."""
    if native_android:
        vflip = ANDROID_PREVIEW_IS_VFLIPPED
        base = android_sensor_rotation(camera_index)
    else:
        vflip = False
        base = 0

    offset = int(app_settings.get("camera_rotation_offset", 0) or 0)
    rotation = (base + offset) % 360
    mirror = bool(app_settings.get("camera_mirror", False))
    return vflip, rotation, mirror


def cycle_rotation_offset(step=90):
    """Advance the user's manual correction one quarter turn and persist
    it. Returns the new offset."""
    offset = (int(app_settings.get("camera_rotation_offset", 0) or 0) + step) % 360
    app_settings.set("camera_rotation_offset", offset)
    return offset


def toggle_mirror():
    """Flip the preview left-to-right and persist it. Returns the new state.

    The Flip button behind this is the escape hatch for a mirrored preview -
    the exact fault this module's header describes, which shipped twice
    because it is invisible in a still frame. If it ever comes back on some
    handset, the farmer can undo it in one tap instead of waiting for a
    release.
    """
    mirror = not bool(app_settings.get("camera_mirror", False))
    app_settings.set("camera_mirror", mirror)
    return mirror


def apply_to_pil(img, vflip, rotation_cw, mirror):
    """Apply the same transform the preview shows to a Pillow image.

    Pillow's rotate() is counter-clockwise, so a clockwise angle goes in
    negated. expand=True keeps the full frame when the rotation is an odd
    quarter turn (a 640x480 frame becomes 480x640) instead of cropping the
    corners off.
    """
    from PIL import Image

    if vflip:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    rotation_cw %= 360
    if rotation_cw:
        img = img.rotate(-rotation_cw, expand=True)
    if mirror:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def texture_to_pil(texture, vflip, rotation_cw, mirror):
    """Turn a Kivy camera texture into an upright Pillow RGB image.

    Reading the texture directly beats Widget.export_to_png(), which was
    used before: export_to_png re-renders the widget at *widget* size, so
    it baked in the letterbox bars and threw away sensor resolution. Here
    we get the full frame at native resolution, already oriented.

    Note the flip below: Kivy textures are stored bottom-row-first, so the
    raw bytes are themselves upside down relative to what is drawn. That is
    a texture-memory detail about how the pixels are handed to us, and has
    nothing to do with the `vflip` stage - undoing it is simply what turns
    the byte buffer back into the frame the preview showed, before the same
    (vflip, rotation, mirror) triple is applied to both alike.
    """
    from PIL import Image

    width, height = texture.size
    # Kivy's Texture.pixels is documented as "RGBA format only, unsigned
    # byte. The origin of the image is at bottom left" - it converts
    # internally, so the texture's own colorfmt (bgr for the desktop
    # OpenCV preview, rgba on Android) must NOT be used to pick a mode
    # here. Reading it as anything but RGBA misreads the stride and
    # produces a skewed, colour-swapped image.
    img = Image.frombytes("RGBA", (width, height), bytes(texture.pixels))
    img = img.transpose(Image.FLIP_TOP_BOTTOM)  # bottom-left origin -> top-left

    img = apply_to_pil(img, vflip, rotation_cw, mirror)
    return img.convert("RGB")


class OrientedCamera(FloatLayout):
    """Hosts a camera widget and rotates/flips it to sit upright.

    The transform lives on this wrapper rather than on the camera widget
    itself so the child can be sized to the ROTATED footprint. For a
    quarter turn the child is given the container's swapped dimensions,
    which means that after rotating it lands exactly on the container's
    box - otherwise a landscape preview turned 90 degrees would stick out
    past the sides of a portrait container and get clipped.
    """

    def __init__(self, camera_widget, vflip=False, rotation_cw=0, mirror=False, **kwargs):
        super().__init__(**kwargs)
        self.camera_widget = camera_widget
        self._vflip = vflip
        self._rotation = rotation_cw % 360
        self._mirror = mirror

        camera_widget.size_hint = (None, None)
        camera_widget.allow_stretch = True
        camera_widget.keep_ratio = True

        with self.canvas.before:
            PushMatrix()
            # Listed outermost-first: Kivy multiplies these onto the matrix
            # in order, so the LAST one listed is applied to the geometry
            # FIRST. Reading bottom-up gives the pipeline we want -
            # vflip, then rotate, then mirror.
            self._mirror_instr = Scale(1, 1, 1)
            self._rotate_instr = Rotate(angle=0, axis=(0, 0, 1))
            self._vflip_instr = Scale(1, 1, 1)
        with self.canvas.after:
            PopMatrix()

        self.add_widget(camera_widget)
        self.bind(pos=self._relayout, size=self._relayout)
        self._apply()

    def set_transform(self, vflip, rotation_cw, mirror):
        self._vflip = vflip
        self._rotation = rotation_cw % 360
        self._mirror = mirror
        self._apply()

    @property
    def transform(self):
        return self._vflip, self._rotation, self._mirror

    def _apply(self, *_):
        # Kivy's Rotate is counter-clockwise; our angles are clockwise.
        self._rotate_instr.angle = (-self._rotation) % 360
        self._vflip_instr.y = -1 if self._vflip else 1
        self._mirror_instr.x = -1 if self._mirror else 1
        self._relayout()

    def _relayout(self, *_):
        origin = self.center
        for instr in (self._rotate_instr, self._vflip_instr, self._mirror_instr):
            instr.origin = origin

        if self._rotation in (90, 270):
            self.camera_widget.size = (self.height, self.width)
        else:
            self.camera_widget.size = self.size
        self.camera_widget.center = origin
