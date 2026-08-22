"""
Farm_land Detect and Classification System (FDCS)
--------------------------------------------------
A mobile pest & disease detection app for farmland, built with Kivy.

Flow:
  HomeScreen (capture/pick image) -> ResultScreen (classification + advisory)
                                   -> HistoryScreen (past scans, SQLite)
                                   -> AdvisoryScreen (browse all known pests)

Author: Daniel Krause (GCTU) - Final Year Capstone Project
"""

import os
from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, SlideTransition
from kivy.lang import Builder
from kivy.core.window import Window
from kivy.utils import platform

from utils.database import ScanDatabase
from model.classifier import PestClassifier

from screens.home_screen import HomeScreen
from screens.result_screen import ResultScreen
from screens.history_screen import HistoryScreen
from screens.advisory_screen import AdvisoryScreen

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load all .kv layout files
Builder.load_file(os.path.join(BASE_DIR, "assets", "home.kv"))
Builder.load_file(os.path.join(BASE_DIR, "assets", "result.kv"))
Builder.load_file(os.path.join(BASE_DIR, "assets", "history.kv"))
Builder.load_file(os.path.join(BASE_DIR, "assets", "advisory.kv"))


def request_android_permissions():
    """Ask for camera + storage permissions at runtime on Android."""
    if platform == "android":
        try:
            from android.permissions import request_permissions, Permission
            request_permissions([
                Permission.CAMERA,
                Permission.WRITE_EXTERNAL_STORAGE,
                Permission.READ_EXTERNAL_STORAGE,
            ])
        except Exception as e:
            print(f"[Permissions] Could not request permissions: {e}")


class FDCSApp(App):
    title = "Farm_land Detect and Classification System"

    def build(self):
        request_android_permissions()

        # Shared, app-wide resources: one DB connection, one loaded model
        self.db = ScanDatabase()
        self.classifier = PestClassifier()

        if platform != "android":
            Window.size = (400, 720)  # simulate a phone screen on desktop

        sm = ScreenManager(transition=SlideTransition())
        sm.add_widget(HomeScreen(name="home"))
        sm.add_widget(ResultScreen(name="result"))
        sm.add_widget(HistoryScreen(name="history"))
        sm.add_widget(AdvisoryScreen(name="advisory"))
        sm.current = "home"
        return sm

    def on_stop(self):
        self.db.close()


if __name__ == "__main__":
    FDCSApp().run()
