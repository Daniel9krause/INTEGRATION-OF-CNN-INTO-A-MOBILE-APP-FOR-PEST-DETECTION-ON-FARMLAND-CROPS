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
        Uses plyer, which maps to a real file-open dialog on desktop and
        Android's document/gallery picker on-device."""
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
