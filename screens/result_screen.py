"""
ResultScreen
-------------
Runs the captured image through PestClassifier and shows the outcome:
  - the captured photo
  - what it is (or an explicit refusal to say)
  - which organism group it belongs to
  - the advisory / recommended action

Then persists the result to SQLite (scan history) and files the raw image
into the growing labeled dataset via DataCollector.

A REFUSAL IS A RESULT
---------------------
This screen branches on `status` from the classifier, never on the
confidence number. Four outcomes are displayed differently on purpose:

    ok          green  - a real diagnosis, safe to act on
    uncertain   amber  - looks like a plant, but which pest is a coin flip
    not_plant   red    - no plant in the photo, nothing diagnosed
    unusable    red    - too dark / blurred / blank to analyse at all

The last two show NO pest name and NO treatment advice. The old screen
printed "Uncertain (closest guess: Beetle)" for a photo of a person, which
still put a pest name and its pesticide recommendation in front of the
farmer - the guess was the harm, not the wording around it.
"""

from kivy.app import App
from kivy.clock import Clock
from kivy.properties import (BooleanProperty, ListProperty, NumericProperty,
                             StringProperty)
from kivy.uix.screenmanager import Screen

from utils import result_presentation
from utils.data_collector import DataCollector
from utils.result_presentation import REFUSED_COLOR

# Re-exported so screens/history_screen.py keeps a single import site for
# the presentation logic.
present = result_presentation.present


class ResultScreen(Screen):
    image_path = StringProperty("")
    label_text = StringProperty("")
    confidence_text = StringProperty("")
    group_text = StringProperty("")
    description_text = StringProperty("")
    advisory_text = StringProperty("")
    confidence_value = NumericProperty(0)
    confidence_color = ListProperty(result_presentation.CONFIDENT_COLOR)
    # The confidence bar is hidden for refusals: there is no meaningful
    # percentage to show when no diagnosis was made, and a "0%" bar reads as
    # "definitely healthy" to someone skimming.
    show_confidence = BooleanProperty(True)
    show_advisory = BooleanProperty(True)
    # The manual flag button overrides a *confident wrong answer*. On a
    # result that already refused to diagnose, it is redundant.
    show_flag_button = BooleanProperty(True)

    def on_pre_enter(self, *args):
        self.data_collector = DataCollector()

    def load_and_classify(self, filepath):
        self.image_path = filepath
        self.label_text = "Analyzing..."
        self.confidence_text = ""
        self.advisory_text = ""
        self.description_text = ""
        self.group_text = ""
        self.confidence_value = 0
        self.show_confidence = False
        self.show_flag_button = False
        # Slight delay so the UI paints the "Analyzing..." state before the
        # (potentially CPU-heavy) inference call blocks the main thread.
        Clock.schedule_once(lambda dt: self._run_inference(filepath), 0.1)

    def _run_inference(self, filepath):
        app = App.get_running_app()

        try:
            result = app.classifier.predict(filepath)
        except Exception as e:
            # A real farmer's phone will eventually feed this a corrupted
            # file, a screenshot, a HEIC/weird format PIL can't open, or a
            # 0-byte file from an interrupted save. None of that should
            # crash the app - show a clear recoverable message instead.
            print(f"[ResultScreen] Classification failed for {filepath}: {e}")
            self._show_load_error()
            return

        self._apply(result)
        self._persist(filepath, result)

    def _apply(self, result):
        """Bind a classifier verdict onto the screen's properties."""
        view = present(
            result["status"], result.get("label"), result.get("confidence", 0.0),
            result.get("headline", ""), result.get("detail", ""),
        )

        self.label_text = view["display_label"]
        self.group_text = view["group_text"]
        self.description_text = view["description_text"]
        self.advisory_text = view["advisory_text"]
        self.confidence_color = view["confidence_color"]
        self.show_confidence = view["show_confidence"]
        self.show_advisory = view["show_advisory"]
        self.show_flag_button = view["show_flag_button"]

        confidence = result.get("confidence", 0.0)
        if view["show_confidence"]:
            self.confidence_text = f"{confidence * 100:.1f}% confidence"
            self.confidence_value = confidence * 100
        else:
            self.confidence_text = ""
            self.confidence_value = 0

        self._current_result = result

    def _show_load_error(self):
        self.label_text = "Couldn't analyze this image"
        self.confidence_text = ""
        self.confidence_value = 0
        self.confidence_color = REFUSED_COLOR
        self.group_text = "No result"
        self.description_text = (
            "This file couldn't be read as an image - it may be corrupted, "
            "in an unsupported format, or something went wrong saving it."
        )
        self.advisory_text = "Try capturing or selecting a different photo (JPG or PNG work best)."
        self.show_confidence = False
        self.show_advisory = True
        self.show_flag_button = False
        self._current_result = None

    def _persist(self, filepath, result):
        """Record the scan and file the image for future retraining.

        A failure here (disk full, storage permission revoked) shouldn't
        hide a successful classification from the user - log it and move on
        rather than crash.
        """
        app = App.get_running_app()
        status = result["status"]
        label = result.get("label") or ""
        view = present(status, result.get("label"), result.get("confidence", 0.0),
                       result.get("headline", ""), result.get("detail", ""))
        try:
            app.db.add_scan(
                filepath,
                # Store the raw top guess even on a refusal - it is useful
                # for review and retraining. `status` is what marks the row
                # as not a trusted match, so History can never redisplay it
                # as one.
                label or view["display_label"],
                result.get("confidence", 0.0),
                view["group_text"],
                flagged_new=(status in ("not_plant", "uncertain")),
                status=status,
                headline=result.get("headline", ""),
                detail=result.get("detail", ""),
            )
            self.data_collector.save_result(filepath, status, label)
        except Exception as e:
            print(f"[ResultScreen] Failed to persist scan (result still shown): {e}")

    def flag_as_new_organism(self):
        """User says 'this isn't actually what the model thinks it is' -
        re-file the image as an unclassified new discovery for expert review
        and future dataset expansion."""
        if not self.image_path:
            return
        self.data_collector.save(self.image_path, "unclassified", flagged_new=True)
        self.advisory_text = ("Thanks - this image has been saved separately as a potential "
                              "new pest/disease not yet in our model. It will be "
                              "reviewed for inclusion in a future model update.")
        self.show_flag_button = False

    def go_home(self):
        App.get_running_app().root.current = "home"

    def go_to_history(self):
        App.get_running_app().root.current = "history"
