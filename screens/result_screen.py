"""
ResultScreen
-------------
Runs the captured image through PestClassifier, displays:
  - the captured photo
  - predicted class + confidence
  - which organism group it belongs to (Insect Pest / Fungal / Viral / Bacterial / Healthy)
  - advisory / recommended action

Then persists the result to SQLite (scan history) and files the raw image
into the growing labeled dataset via DataCollector.
"""

import os
import json
from kivy.uix.screenmanager import Screen
from kivy.app import App
from kivy.properties import StringProperty, NumericProperty
from kivy.clock import Clock

from utils.data_collector import DataCollector

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADVISORY_PATH = os.path.join(BASE_DIR, "data", "advisory_data.json")

with open(ADVISORY_PATH, "r") as f:
    ADVISORY_DB = json.load(f)


class ResultScreen(Screen):
    image_path = StringProperty("")
    label_text = StringProperty("")
    confidence_text = StringProperty("")
    group_text = StringProperty("")
    description_text = StringProperty("")
    advisory_text = StringProperty("")
    confidence_value = NumericProperty(0)

    def on_pre_enter(self, *args):
        self.data_collector = DataCollector()

    def load_and_classify(self, filepath):
        self.image_path = filepath
        self.label_text = "Analyzing..."
        self.confidence_text = ""
        self.advisory_text = ""
        # Slight delay so the UI paints the "Analyzing..." state before the
        # (potentially CPU-heavy) inference call blocks the main thread.
        Clock.schedule_once(lambda dt: self._run_inference(filepath), 0.1)

    def _run_inference(self, filepath):
        app = App.get_running_app()
        result = app.classifier.predict(filepath)

        label = result["label"]
        confidence = result["confidence"]
        info = ADVISORY_DB.get(label, {
            "group": "Unknown",
            "type": "Unclassified organism",
            "description": "This does not closely match any organism in our current 13-class database.",
            "advisory": "Consider consulting a local agricultural extension officer. This image has been saved for expert review and future model retraining.",
        })

        self.label_text = label
        self.confidence_text = f"{confidence * 100:.1f}% confidence"
        self.confidence_value = confidence * 100
        self.group_text = info["group"]
        self.description_text = info["description"]
        self.advisory_text = info["advisory"]

        self._current_result = {
            "label": label,
            "confidence": confidence,
            "group": info["group"],
        }

        # Persist: 1) scan history record, 2) raw image into class-labeled dataset
        app.db.add_scan(filepath, label, confidence, info["group"], flagged_new=(info["group"] == "Unknown"))
        self.data_collector.save(filepath, label, flagged_new=(info["group"] == "Unknown"))

    def flag_as_new_organism(self):
        """User says 'this isn't actually what the model thinks it is' —
        re-file the image as an unclassified new discovery for expert review
        and future dataset expansion, and update the history record."""
        if not self.image_path:
            return
        self.data_collector.save(self.image_path, "unclassified", flagged_new=True)
        self.advisory_text = ("Thanks — this image has been saved separately as a potential "
                               "new pest/disease not yet in our 13-class model. It will be "
                               "reviewed for inclusion in a future model update.")

    def go_home(self):
        App.get_running_app().root.current = "home"

    def go_to_history(self):
        App.get_running_app().root.current = "history"
