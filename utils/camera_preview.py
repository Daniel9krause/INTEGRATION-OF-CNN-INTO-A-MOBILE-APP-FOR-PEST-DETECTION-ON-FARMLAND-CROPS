"""
CameraPreview
--------------
A drop-in replacement for Kivy's built-in `Camera` widget, used only for
DESKTOP testing (Windows/Mac/Linux). Kivy 2.3.0's OpenCV camera provider
has a known bug (`AttributeError: 'CameraOpenCV' object has no attribute
'fps'`) that crashes on init on Windows. Driving OpenCV ourselves avoids
that bug entirely and gives us reliable frame capture.

On Android, this class is never used — HomeScreen falls back to Kivy's
native Camera widget instead, which uses Android's own camera API and
doesn't have this bug.
"""

import cv2
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.clock import Clock


class CameraPreview(Image):
    def __init__(self, index=0, fps=30, **kwargs):
        super().__init__(**kwargs)
        self.capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)  # CAP_DSHOW = fast, reliable on Windows
        self._fps = fps
        self._event = None
        self._last_frame = None

    def start(self):
        if self._event is None:
            self._event = Clock.schedule_interval(self._update, 1.0 / self._fps)

    def stop(self):
        if self._event is not None:
            self._event.cancel()
            self._event = None

    def _update(self, dt):
        ret, frame = self.capture.read()
        if not ret:
            return
        self._last_frame = frame

        # OpenCV gives BGR, bottom-up isn't required here but Kivy textures
        # are bottom-left origin, so flip vertically for correct display.
        buf = cv2.flip(frame, 0).tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt="bgr")
        texture.blit_buffer(buf, colorfmt="bgr", bufferfmt="ubyte")
        self.texture = texture

    def last_frame_pil(self):
        """Most recent frame as a Pillow RGB image, or None.

        HomeScreen puts this through the same orientation transform as the
        Android capture path, so the Rotate button behaves identically on
        desktop and the saved photo always matches the preview.
        """
        if self._last_frame is None:
            return None
        from PIL import Image
        return Image.fromarray(cv2.cvtColor(self._last_frame, cv2.COLOR_BGR2RGB))

    def capture_to_file(self, filepath):
        """Save the most recent frame to disk. Returns True on success."""
        if self._last_frame is None:
            return False
        cv2.imwrite(filepath, self._last_frame)
        return True

    def release(self):
        self.stop()
        if self.capture is not None:
            self.capture.release()
