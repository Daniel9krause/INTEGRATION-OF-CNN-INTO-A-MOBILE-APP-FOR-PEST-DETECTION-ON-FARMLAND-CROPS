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
from kivy.properties import StringProperty, NumericProperty, ListProperty, BooleanProperty
from kivy.clock import Clock

from utils.data_collector import DataCollector

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADVISORY_PATH = os.path.join(BASE_DIR, "data", "advisory_data.json")

with open(ADVISORY_PATH, "r") as f:
    ADVISORY_DB = json.load(f)

# A softmax classifier always has to pick SOMETHING from its known classes —
# even a photo of a person, a wall, or a car gets forced into whichever of
# the 13 classes activates most. A low top confidence is the only signal
# we get that the image probably isn't one of our known classes at all, so
# below this we refuse to present the raw guess as a real match. Tune this
# based on real testing: too many legit photos coming back "Uncertain" ->
# lower it; obviously wrong photos still confidently labeled -> raise it.
CONFIDENCE_THRESHOLD = 0.60

_CONFIDENT_COLOR = [0.2, 0.6, 0.2, 1]   # green
_UNCERTAIN_COLOR = [0.85, 0.55, 0.1, 1]  # amber


def build_result_info(label, confidence):
    """Turns a raw (label, confidence) prediction into what the UI should
    show — shared by a fresh classification (ResultScreen) and reopening a
    saved one from History, so both stay consistent.
    Returns (display_label, info_dict, confidence_color).
    """
    if confidence < CONFIDENCE_THRESHOLD:
        display_label = f"Uncertain (closest guess: {label})"
        info = {
            "group": "Unknown",
            "type": "Unclassified organism",
            "description": (
                "The model isn't confident this matches any of our 13 known "
                "pest/disease classes — it may be an unrelated photo, poor "
                "lighting/framing, or a genuinely new organism."
            ),
            "advisory": (
                "Try recapturing with the leaf/pest filling more of the frame "
                "in good lighting. If it still comes back uncertain, this may "
                "be worth a local agricultural extension officer's review — "
                "the image has been saved for that and for future retraining."
            ),
        }
        return display_label, info, _UNCERTAIN_COLOR

    info = ADVISORY_DB.get(label, {
        # Safety net only — shouldn't trigger unless labels.txt and
        # advisory_data.json ever drift out of sync with each other.
        "group": "Unknown",
        "type": "Unclassified organism",
        "description": "This does not closely match any organism in our current 13-class database.",
        "advisory": "Consider consulting a local agricultural extension officer. This image has been saved for expert review and future model retraining.",
    })
    return label, info, _CONFIDENT_COLOR


class ResultScreen(Screen):
    image_path = StringProperty("")
    label_text = StringProperty("")
    confidence_text = StringProperty("")
    group_text = StringProperty("")
    description_text = StringProperty("")
    advisory_text = StringProperty("")
    confidence_value = NumericProperty(0)
    confidence_color = ListProperty(_CONFIDENT_COLOR)
    # Hidden when the result is already auto-flagged as "Unknown" — the
    # manual flag button is for overriding a *confident* wrong answer;
    # showing it on top of an already-uncertain result is redundant.
    show_flag_button = BooleanProperty(True)

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

        try:
            result = app.classifier.predict(filepath)
            label = result["label"]
            confidence = result["confidence"]
            display_label, info, color = build_result_info(label, confidence)
        except Exception as e:
            # A real farmer's phone will eventually feed this a corrupted
            # file, a screenshot, a HEIC/weird format PIL can't open, or a
            # 0-byte file from an interrupted save. None of that should
            # crash the app — show a clear recoverable message instead.
            print(f"[ResultScreen] Classification failed for {filepath}: {e}")
            self.label_text = "Couldn't analyze this image"
            self.confidence_text = ""
            self.confidence_value = 0
            self.confidence_color = _UNCERTAIN_COLOR
            self.group_text = ""
            self.description_text = (
                "This file couldn't be read as an image — it may be corrupted, "
                "in an unsupported format, or something went wrong saving it."
            )
            self.advisory_text = "Try capturing or selecting a different photo (JPG or PNG work best)."
            self.show_flag_button = False
            self._current_result = None
            return

        self.label_text = display_label
        self.confidence_text = f"{confidence * 100:.1f}% confidence"
        self.confidence_value = confidence * 100
        self.confidence_color = color
        self.group_text = info["group"]
        self.description_text = info["description"]
        self.advisory_text = info["advisory"]
        self.show_flag_button = (info["group"] != "Unknown")

        self._current_result = {
            "label": label,
            "confidence": confidence,
            "group": info["group"],
        }

        # Persist: 1) scan history record, 2) raw image into class-labeled
        # dataset. A failure here (disk full, storage permission revoked)
        # shouldn't hide a successful classification result from the user —
        # log it and move on rather than crash.
        # Note: predicted_label stores the model's raw top guess even when
        # uncertain (useful for review/retraining) — group_name="Unknown" is
        # what actually marks it as not a trusted match.
        try:
            app.db.add_scan(filepath, label, confidence, info["group"], flagged_new=(info["group"] == "Unknown"))
            self.data_collector.save(filepath, label, flagged_new=(info["group"] == "Unknown"))
        except Exception as e:
            print(f"[ResultScreen] Failed to persist scan (result still shown): {e}")

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
        self.show_flag_button = False

    def go_home(self):
        App.get_running_app().root.current = "home"

    def go_to_history(self):
        App.get_running_app().root.current = "history"
