"""
HomeScreen
-----------
The app's entry screen: live camera preview + capture button, plus a
"pick from gallery" fallback. On capture, the image is saved locally and
handed off to ResultScreen for classification.

Camera handling is platform-aware:
  - On Android: uses Kivy's built-in `Camera` widget (native camera API).
  - On desktop (Windows/Mac/Linux): uses our own `CameraPreview` (OpenCV),
    because Kivy 2.3.0's OpenCV camera provider has a known crash bug on
    Windows. See utils/camera_preview.py for details.
"""

import os
import time
from kivy.uix.screenmanager import Screen
from kivy.utils import platform
from kivy.app import App
from kivy.clock import Clock

if platform == "android":
    from android.storage import app_storage_path
    CAPTURE_DIR = os.path.join(app_storage_path(), "captures")
else:
    CAPTURE_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "storage", "captures"
    )

os.makedirs(CAPTURE_DIR, exist_ok=True)

# Arbitrary but fixed Android activity-result request code for our own
# gallery-picker Intent (see _pick_image_android below) - just needs to
# not collide with any other startActivityForResult call in this app.
_UPLOAD_REQUEST_CODE = 0x1DEA


class HomeScreen(Screen):
    _camera_widget = None
    _using_native_camera = False

    def on_pre_enter(self, *args):
        self._ensure_camera_widget()
        if self._camera_widget is None:
            return
        if self._using_native_camera:
            self._camera_widget.play = True
        else:
            self._camera_widget.start()

    def on_leave(self, *args):
        if self._camera_widget is None:
            return
        if self._using_native_camera:
            self._camera_widget.play = False
        else:
            self._camera_widget.stop()

    def _ensure_camera_widget(self):
        """Lazily create the platform-appropriate camera widget and drop it
        into the `camera_container` placeholder from home.kv."""
        if self._camera_widget is not None:
            return

        container = self.ids.get("camera_container")
        if container is None:
            return

        try:
            if platform == "android":
                from kivy.uix.camera import Camera
                self._camera_widget = Camera(resolution=(640, 480), play=False, index=0)
                self._using_native_camera = True
                self._fix_camera_orientation(self._camera_widget)
            else:
                from utils.camera_preview import CameraPreview
                self._camera_widget = CameraPreview(index=0, fps=24)
                self._using_native_camera = False
            container.add_widget(self._camera_widget)
        except Exception as e:
            # Camera permission denied, no camera hardware, or another app
            # already holding it — don't crash, just fall back to Upload.
            print(f"[HomeScreen] Camera unavailable: {e}")
            self._camera_widget = None
            self._show_status("Camera unavailable — use Upload instead, or check camera permission in Settings.")

    @staticmethod
    def _fix_camera_orientation(widget):
        """Kivy's Camera widget renders upside-down on many Android
        devices - a long-documented Kivy/Android quirk in how the camera
        sensor's native (landscape) orientation ends up in the preview
        texture. Rotating the widget's own canvas 180 degrees corrects
        both the live preview AND export_to_png() captures, since
        export_to_png renders through this same canvas rather than
        reading raw texture bytes directly. (If a particular device
        instead shows a 90-degree sideways rotation rather than upside-
        down, this angle is the one thing to change.)"""
        from kivy.graphics import PushMatrix, PopMatrix, Rotate

        with widget.canvas.before:
            PushMatrix()
            rotate = Rotate(angle=180, origin=widget.center)
        with widget.canvas.after:
            PopMatrix()

        def _update_origin(instance, _value):
            rotate.origin = instance.center

        widget.bind(pos=_update_origin, size=_update_origin)

    def capture_image(self):
        """Grab the current camera frame and save it as a real image file."""
        if self._camera_widget is None:
            self._show_status("Camera not ready yet — try again in a second.")
            return

        filename = f"scan_{int(time.time())}.jpg"
        filepath = os.path.join(CAPTURE_DIR, filename)

        try:
            if self._using_native_camera:
                if self._camera_widget.texture is None:
                    self._show_status("Camera not ready yet — try again in a second.")
                    return
                self._camera_widget.export_to_png(filepath)
            else:
                if not self._camera_widget.capture_to_file(filepath):
                    self._show_status("Camera not ready yet — try again in a second.")
                    return
        except Exception as e:
            # Disk full, storage permission revoked mid-session, etc.
            print(f"[HomeScreen] Capture failed: {e}")
            self._show_status("Couldn't save the photo — check storage space and try again.")
            return

        self._go_to_result(filepath)

    def pick_image(self):
        """Open a native file picker so the user can upload an existing
        photo (e.g. from their gallery) instead of using the live camera.
        On desktop this uses plyer (a real file-open dialog). On Android
        we bypass plyer's own filechooser and drive the picker directly
        (see _pick_image_android) - see that method's docstring for why."""
        if platform == "android":
            self._pick_image_android()
            return

        try:
            from plyer import filechooser
        except Exception as e:
            self._show_status("Image upload isn't available on this device.")
            print(f"[HomeScreen] plyer.filechooser import failed: {e}")
            return

        try:
            filechooser.open_file(
                on_selection=self._on_gallery_selection,
                multiple=False,
                title="Select a photo of the pest/leaf",
                filters=[["Images", "*.jpg", "*.jpeg", "*.png", "*.bmp"]],
            )
        except Exception as e:
            self._show_status("Couldn't open the file picker.")
            print(f"[HomeScreen] filechooser.open_file failed: {e}")

    def _pick_image_android(self):
        """Android's system picker (gallery/Photos/Files) hands back a
        content:// URI, not a real filesystem path. plyer's built-in
        Android filechooser only resolves a handful of known content-
        provider authorities, and does so via the `_data` column - which
        Android 10+'s Scoped Storage leaves null for most providers,
        including the modern system Photo Picker most phones now show by
        default. So plyer silently returns None/an unusable path even
        when the user successfully picked a real photo, and the app
        reports "No image selected". Driving the picker Intent ourselves
        and reading the result via ContentResolver works regardless of
        Android version or which picker the OS shows."""
        try:
            from jnius import autoclass
            from android import activity
        except Exception as e:
            self._show_status("Image upload isn't available on this device.")
            print(f"[HomeScreen] Android filechooser setup failed: {e}")
            return

        Intent = autoclass("android.content.Intent")
        PythonActivity = autoclass("org.kivy.android.PythonActivity")

        intent = Intent(Intent.ACTION_OPEN_DOCUMENT)
        intent.addCategory(Intent.CATEGORY_OPENABLE)
        intent.setType("image/*")

        def on_activity_result(request_code, result_code, data):
            if request_code != _UPLOAD_REQUEST_CODE:
                return
            activity.unbind(on_activity_result=on_activity_result)
            RESULT_OK = -1  # android.app.Activity.RESULT_OK
            filepath = None
            if result_code == RESULT_OK and data is not None:
                try:
                    filepath = self._copy_android_uri(data.getData())
                except Exception as e:
                    print(f"[HomeScreen] Failed to read picked image: {e}")
            Clock.schedule_once(lambda dt: self.choose_from_gallery(filepath))

        activity.bind(on_activity_result=on_activity_result)
        try:
            PythonActivity.mActivity.startActivityForResult(intent, _UPLOAD_REQUEST_CODE)
        except Exception as e:
            activity.unbind(on_activity_result=on_activity_result)
            self._show_status("Couldn't open the file picker.")
            print(f"[HomeScreen] startActivityForResult failed: {e}")

    @staticmethod
    def _copy_android_uri(uri):
        """Copy a content:// Uri's bytes into a real local file, via a
        raw OS-level file descriptor (ParcelFileDescriptor.detachFd())
        rather than a manual Java byte[]-copying loop - this sidesteps
        pyjnius array marshalling entirely, since Python's os.fdopen()
        reads a plain file descriptor natively."""
        if uri is None:
            return None
        from jnius import autoclass

        PythonActivity = autoclass("org.kivy.android.PythonActivity")
        resolver = PythonActivity.mActivity.getContentResolver()
        pfd = resolver.openFileDescriptor(uri, "r")
        if pfd is None:
            return None

        fd = pfd.detachFd()
        dest_path = os.path.join(CAPTURE_DIR, f"upload_{int(time.time())}.jpg")
        with os.fdopen(fd, "rb") as src, open(dest_path, "wb") as out:
            while True:
                chunk = src.read(65536)
                if not chunk:
                    break
                out.write(chunk)
        return dest_path

    def _on_gallery_selection(self, selection):
        """plyer's callback can fire off the Kivy thread (e.g. from an
        Android activity result), so hop back onto the main thread before
        touching any widgets or the screen manager."""
        Clock.schedule_once(lambda dt: self.choose_from_gallery(
            selection[0] if selection else None
        ))

    def choose_from_gallery(self, filepath):
        """Called once the user has picked an existing photo of a pest/leaf
        via pick_image()."""
        if filepath and os.path.exists(filepath):
            self._go_to_result(filepath)
        else:
            self._show_status("No image selected.")

    def _go_to_result(self, filepath):
        app = App.get_running_app()
        result_screen = app.root.get_screen("result")
        result_screen.load_and_classify(filepath)
        app.root.current = "result"

    def _show_status(self, message):
        status_label = self.ids.get("status_label")
        if status_label:
            status_label.text = message
            Clock.schedule_once(lambda dt: setattr(status_label, "text", ""), 3)

    def go_to_history(self):
        App.get_running_app().root.current = "history"

    def go_to_advisory(self):
        App.get_running_app().root.current = "advisory"
