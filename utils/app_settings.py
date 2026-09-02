"""
app_settings
-------------
A tiny JSON-backed settings store for the handful of preferences that have
to survive an app restart - currently the camera orientation correction.

Kept deliberately dumb (load-on-read, write-through on set) because there
are only a few keys and they change rarely. Any write failure is swallowed:
a farmer whose storage is full should still get a working camera, just one
that forgets its rotation next launch.
"""

import json
import os

from kivy.utils import platform

if platform == "android":
    from android.storage import app_storage_path
    SETTINGS_DIR = app_storage_path()
else:
    SETTINGS_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "storage"
    )

os.makedirs(SETTINGS_DIR, exist_ok=True)
SETTINGS_PATH = os.path.join(SETTINGS_DIR, "settings.json")

DEFAULTS = {
    # Extra clockwise rotation, in degrees, applied on top of the angle
    # computed from the Android camera sensor - see utils/camera_transform.
    # 0 unless the user has tapped the on-screen Rotate button.
    "camera_rotation_offset": 0,
    # Horizontal mirror toggle, for the rare device whose preview comes
    # back flipped rather than rotated.
    "camera_mirror": False,
}


def _load():
    try:
        with open(SETTINGS_PATH, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            merged = dict(DEFAULTS)
            merged.update(data)
            return merged
    except (OSError, ValueError):
        pass
    return dict(DEFAULTS)


def get(key, default=None):
    return _load().get(key, DEFAULTS.get(key, default))


def set(key, value):  # noqa: A001 - reads naturally as app_settings.set(...)
    data = _load()
    data[key] = value
    try:
        with open(SETTINGS_PATH, "w") as f:
            json.dump(data, f, indent=2)
    except OSError as e:
        print(f"[app_settings] Could not persist {key}: {e}")
    return value
