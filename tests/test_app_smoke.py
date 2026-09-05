"""
test_app_smoke
---------------
Builds the real screens against the real .kv files, headlessly.

The decision-logic tests prove the classifier refuses correctly; this proves
the app can still be assembled and that a refusal actually reaches the
screen. Between them they cover the seam where this change is riskiest: the
Result screen now hides widgets based on `status`, and the .kv files bind
properties (`show_confidence`, `show_advisory`) that did not exist before -
a typo there is invisible to Python's compiler and would only surface as a
blank or broken screen on the phone.

Uses Kivy's mock GL backend so no graphics driver is needed. Note it does
NOT set KIVY_WINDOW=mock - Kivy 2.3.1 has no such window provider, and
asking for one makes Kivy fail to find any window at all and abort before a
single test runs.

Run:  python tests/test_app_smoke.py
"""

import os
import sys

import numpy as np

os.environ.setdefault("KIVY_GL_BACKEND", "mock")
os.environ.setdefault("KIVY_NO_ARGS", "1")
os.environ.setdefault("KIVY_NO_CONSOLELOG", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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


def _test_end_to_end():
    """Drive the real path a capture takes: image -> classifier -> screen ->
    SQLite -> collected_data.

    The unit tests stub the network and the smoke tests stub the classifier;
    this one runs the actual trained model through the actual ResultScreen
    and then checks what landed in the database and on disk. It is the only
    test that would catch _persist() and DataCollector.save_result() being
    wired up wrongly - the two places a refusal could still leak out as a
    pest label after everything upstream got it right.
    """
    import tempfile

    from kivy.app import App

    from model.classifier import PestClassifier
    from screens.result_screen import ResultScreen
    from utils.data_collector import DataCollector
    from utils.database import ScanDatabase

    print("\nend-to-end: real model -> screen -> database -> collected_data")

    tmp = tempfile.mkdtemp()

    class _HeadlessApp(App):
        pass

    app = _HeadlessApp()
    app.db = ScanDatabase(db_path=os.path.join(tmp, "scans.db"))
    app.classifier = PestClassifier()
    # Kivy resolves App.get_running_app() from this attribute; setting it
    # directly is what lets the screen run without an event loop.
    App._running_app = app

    screen = ResultScreen(name="result")
    screen.data_collector = DataCollector(base_dir=os.path.join(tmp, "collected"))

    def run(path):
        # Call _run_inference directly: load_and_classify defers it onto the
        # Clock, which never ticks without a running event loop.
        screen._run_inference(path)
        return app.db.get_all_scans()[0]

    # 1. A real leaf photo should still get a diagnosis end to end.
    leaf = None
    for folder in ("Leaf Blight", "Leaf Spot", "Mosaic Virus"):
        d = os.path.join(BASE_DIR, "data", "dataset", folder)
        if os.path.isdir(d):
            for name in sorted(os.listdir(d)):
                if name.lower().endswith((".jpg", ".jpeg", ".png")):
                    leaf = os.path.join(d, name)
                    break
        if leaf:
            break

    if leaf:
        row = run(leaf)
        check("leaf photo reaches the screen with a label",
              bool(screen.label_text) and screen.label_text != "Analyzing...",
              screen.label_text)
        check("leaf photo is recorded in history", row["image_path"] == leaf)
        check("history row carries a status", row["status"] in
              ("ok", "uncertain", "not_plant", "unusable"), row["status"])

    # 2. A pitch-black frame must be refused, must NOT be filed under a pest
    #    name, and must not be stored as image data at all.
    dark = os.path.join(tmp, "dark.jpg")
    from PIL import Image as PILImage
    PILImage.fromarray(np.zeros((240, 320, 3), np.uint8)).save(dark)

    row = run(dark)
    check("black frame refused end to end", row["status"] == "unusable", row["status"])
    check("refusal shows no confidence", not screen.show_confidence)
    check("refusal offers no flag button", not screen.show_flag_button)

    collected = os.path.join(tmp, "collected")
    folders = sorted(os.listdir(collected)) if os.path.isdir(collected) else []
    check("unusable frame not filed into any class folder",
          "_low_quality" not in folders and "Healthy_Leaf" not in folders, folders)

    # 3. A non-plant photo must be filed under _not_plant, never a pest.
    def _pest_file_count():
        """Images filed under real class folders (leading underscore = a
        refusal bucket). Compared before and after, so the folder the
        correctly-diagnosed leaf in step 1 legitimately created does not
        count against us - only a NEW pest-filed image would."""
        if not os.path.isdir(collected):
            return 0
        total = 0
        for f in os.listdir(collected):
            d = os.path.join(collected, f)
            if os.path.isdir(d) and not f.startswith("_"):
                total += len(os.listdir(d))
        return total

    neg_dir = os.path.join(BASE_DIR, "data", "dataset", "Not Plant")
    if os.path.isdir(neg_dir):
        before = _pest_file_count()
        negs = [f for f in sorted(os.listdir(neg_dir)) if f.lower().endswith(".jpg")]
        filed = None
        for name in negs[:25]:
            row = run(os.path.join(neg_dir, name))
            if row["status"] == "not_plant":
                filed = row
                break
        check("a non-plant photo is refused end to end", filed is not None,
              "none of the first 25 negatives were refused")
        folders = sorted(os.listdir(collected)) if os.path.isdir(collected) else []
        check("non-plant filed under _not_plant", "_not_plant" in folders, folders)
        check("refusals add nothing to any pest folder",
              _pest_file_count() == before,
              f"{before} -> {_pest_file_count()}")

    app.db.close()
    App._running_app = None


def _test_history_removal():
    """Removing one scan takes its row and its photo, and nothing else.

    The dangerous half of this feature is the file deletion, not the SQL:
    History stores whatever path the row carries, and an image that came
    from outside the app's own captures folder is somebody's original
    photo. Deleting one of those while "tidying up a list" is not a bug the
    user would forgive, so the guard is tested directly rather than trusted.
    """
    import tempfile

    from kivy.app import App

    from screens.history_screen import HistoryScreen, _discard_capture
    from screens.home_screen import CAPTURE_DIR
    from utils.database import ScanDatabase

    print("\nhistory: removing a single scan")

    tmp = tempfile.mkdtemp()

    class _HeadlessApp(App):
        pass

    app = _HeadlessApp()
    app.db = ScanDatabase(db_path=os.path.join(tmp, "scans.db"))
    App._running_app = app

    ours = os.path.join(CAPTURE_DIR, f"_test_remove_{os.getpid()}.jpg")
    from PIL import Image as PILImage
    PILImage.fromarray(np.zeros((8, 8, 3), np.uint8)).save(ours)

    # An "uploaded from the gallery" path that was never copied in - the
    # case the guard exists for.
    theirs = os.path.join(tmp, "somebody_elses_photo.jpg")
    PILImage.fromarray(np.zeros((8, 8, 3), np.uint8)).save(theirs)

    keep_id = app.db.add_scan(theirs, "Leaf Spot", 0.91, "Fungal Disease", status="ok")
    drop_id = app.db.add_scan(ours, "Aphids", 0.88, "Insect Pest", status="ok")

    screen = HistoryScreen(name="history")
    doomed = app.db.get_scan(drop_id)
    screen._delete_scan(doomed)

    remaining = [row["id"] for row in app.db.get_all_scans()]
    check("the removed scan is gone from history", drop_id not in remaining, remaining)
    check("the other scan is untouched", keep_id in remaining, remaining)
    check("the removed scan's photo is deleted", not os.path.exists(ours))

    # Same call, on a path this app never created.
    _discard_capture(theirs)
    check("a photo outside the captures folder is left alone",
          os.path.exists(theirs))

    check("a missing file is not an error", _discard_capture(ours) is None)
    check("an empty path is not an error", _discard_capture("") is None)

    # The confirmation dialog, driven without a window: Popup.open is stubbed
    # so the test never needs a display, and the buttons are dispatched
    # directly. Cancel is the case worth pinning - a dialog that deletes
    # whichever button you press is worse than no dialog at all, and nothing
    # in the UI would show it.
    from kivy.uix.popup import Popup

    real_open = Popup.open
    opened = {}
    Popup.open = lambda self, *a, **kw: opened.__setitem__("popup", self)
    try:
        survivor = app.db.get_scan(keep_id)
        screen._confirm_delete(survivor)
        popup = opened.get("popup")
        check("Remove asks before deleting", popup is not None)

        if popup is not None:
            buttons = {b.text: b for b in popup.content.children[0].children}
            check("the dialog offers both answers",
                  set(buttons) == {"Remove", "Cancel"}, sorted(buttons))

            buttons["Cancel"].dispatch("on_release")
            check("cancelling keeps the scan",
                  keep_id in [r["id"] for r in app.db.get_all_scans()])

            opened.clear()
            screen._confirm_delete(survivor)
            popup = opened["popup"]
            {b.text: b for b in popup.content.children[0].children}["Remove"] \
                .dispatch("on_release")
            check("confirming removes the scan",
                  keep_id not in [r["id"] for r in app.db.get_all_scans()])
            check("cancelled-then-confirmed still spares the outside photo",
                  os.path.exists(theirs))
    finally:
        Popup.open = real_open

    app.db.close()
    App._running_app = None
    if os.path.exists(ours):
        os.remove(ours)


def main():
    print("=" * 60)
    print("app smoke tests")
    print("=" * 60)

    from kivy.lang import Builder

    print("\nkv layouts parse")
    for name in ("home", "result", "history", "advisory"):
        path = os.path.join(BASE_DIR, "assets", f"{name}.kv")
        try:
            Builder.load_file(path)
            check(f"{name}.kv loads", True)
        except Exception as e:
            check(f"{name}.kv loads", False, repr(e))

    print("\nscreens instantiate")
    from screens.advisory_screen import AdvisoryScreen
    from screens.history_screen import HistoryScreen
    from screens.home_screen import HomeScreen
    from screens.result_screen import ResultScreen

    screens = {}
    for cls, key in ((HomeScreen, "home"), (ResultScreen, "result"),
                     (HistoryScreen, "history"), (AdvisoryScreen, "advisory")):
        try:
            screens[key] = cls(name=key)
            check(f"{cls.__name__} constructs", True)
        except Exception as e:
            check(f"{cls.__name__} constructs", False, repr(e))

    print("\nhome.kv exposes the widgets home_screen.py looks up")
    home = screens.get("home")
    if home is not None:
        check("camera_container id present", "camera_container" in home.ids,
              list(home.ids))
        check("status_label id present", "status_label" in home.ids, list(home.ids))
        check("rotate_camera() is callable", callable(getattr(home, "rotate_camera", None)))
        check("flip_camera() is callable", callable(getattr(home, "flip_camera", None)))

    print("\nresult screen reflects each verdict")
    result = screens.get("result")
    if result is not None:
        from utils.result_presentation import ADVISORY_DB

        result._apply({"status": "ok", "label": "Leaf Blight", "confidence": 0.93,
                       "headline": "Leaf Blight", "detail": ""})
        check("ok: names the disease", result.label_text == "Leaf Blight",
              result.label_text)
        check("ok: shows confidence", result.show_confidence)
        check("ok: shows the real advisory",
              result.advisory_text == ADVISORY_DB["Leaf Blight"]["advisory"])

        result._apply({"status": "not_plant", "label": None, "confidence": 0.0,
                       "headline": "No plant or leaf detected",
                       "detail": "This photo does not appear to show a plant."})
        check("not_plant: says so plainly",
              result.label_text == "No plant or leaf detected", result.label_text)
        check("not_plant: confidence hidden", not result.show_confidence)
        check("not_plant: confidence bar zeroed", result.confidence_value == 0)
        check("not_plant: flag button hidden", not result.show_flag_button)
        check("not_plant: no pest advisory",
              all(result.advisory_text != v["advisory"]
                  for k, v in ADVISORY_DB.items() if k != "Not Plant"))

        result._apply({"status": "unusable", "label": None, "confidence": 0.0,
                       "headline": "Too dark to identify",
                       "detail": "This photo is too dark."})
        check("unusable: names the problem",
              result.label_text == "Too dark to identify", result.label_text)
        check("unusable: advisory box hidden", not result.show_advisory)

        result._apply({"status": "uncertain", "label": "Beetle", "confidence": 0.41,
                       "headline": "Not sure - possible Beetle", "detail": ""})
        check("uncertain: names the guess", "Beetle" in result.label_text,
              result.label_text)
        check("uncertain: confidence shown", result.show_confidence)
        check("uncertain: no Beetle treatment advice",
              result.advisory_text != ADVISORY_DB["Beetle"]["advisory"])

    print("\nOrientedCamera lays out a rotated preview")
    from kivy.uix.image import Image as KivyImage

    from utils.camera_transform import OrientedCamera

    inner = KivyImage()
    box = OrientedCamera(inner, vflip=True, rotation_cw=90, mirror=False)
    box.size = (400, 700)
    box.pos = (0, 0)
    box._relayout()
    check("quarter turn swaps the child's dimensions",
          tuple(inner.size) == (700, 400), inner.size)
    check("child stays centred on the container",
          tuple(inner.center) == tuple(box.center), (inner.center, box.center))
    check("transform triple round-trips", box.transform == (True, 90, False),
          box.transform)

    box.set_transform(False, 180, True)
    box._relayout()
    check("half turn keeps the child's dimensions",
          tuple(inner.size) == (400, 700), inner.size)
    check("Kivy angle is the counter-clockwise complement",
          box._rotate_instr.angle == 180, box._rotate_instr.angle)

    box.set_transform(False, 90, False)
    box._relayout()
    check("90 deg clockwise becomes 270 deg counter-clockwise in Kivy",
          box._rotate_instr.angle == 270, box._rotate_instr.angle)

    print("\ndatabase migration keeps old rows readable")
    import sqlite3
    import tempfile

    from utils.database import ScanDatabase

    tmp = os.path.join(tempfile.mkdtemp(), "old.db")
    legacy = sqlite3.connect(tmp)
    legacy.execute("""CREATE TABLE scans (
        id INTEGER PRIMARY KEY AUTOINCREMENT, image_path TEXT NOT NULL,
        predicted_label TEXT NOT NULL, confidence REAL NOT NULL,
        group_name TEXT, flagged_new INTEGER DEFAULT 0, created_at TEXT NOT NULL)""")
    legacy.execute("INSERT INTO scans (image_path, predicted_label, confidence, "
                   "group_name, flagged_new, created_at) VALUES "
                   "('/old.jpg', 'Aphids', 0.88, 'Insect Pest', 0, '2026-01-01 00:00:00')")
    legacy.commit()
    legacy.close()

    db = ScanDatabase(db_path=tmp)
    rows = db.get_all_scans()
    check("pre-upgrade row survives", len(rows) == 1 and rows[0]["predicted_label"] == "Aphids")
    check("new status column added", "status" in rows[0].keys(), list(rows[0].keys()))
    check("legacy row has no status", rows[0]["status"] is None)

    db.add_scan("/new.jpg", "Leaf Spot", 0.91, "Fungal Disease",
                status="ok", headline="Leaf Spot", detail="")
    newest = db.get_all_scans()[0]
    check("new row records its status", newest["status"] == "ok", newest["status"])

    from screens.history_screen import _row_value
    check("_row_value defaults a NULL status to ok",
          _row_value(rows[0], "status", "ok") == "ok")
    db.close()

    _test_history_removal()

    _test_end_to_end()

    print("\n" + "=" * 60)
    print(f"{PASSED} passed, {FAILED} failed")
    print("=" * 60)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
