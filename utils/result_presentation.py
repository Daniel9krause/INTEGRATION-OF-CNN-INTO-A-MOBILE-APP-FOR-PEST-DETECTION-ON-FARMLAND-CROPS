"""
result_presentation
--------------------
Turns a classifier verdict into the fields the Result screen displays.

Two callers share this: ResultScreen after a fresh scan, and HistoryScreen
reopening a stored one. Keeping the mapping here is what stops them
drifting - previously History re-derived "is this trustworthy?" from the
confidence number alone, which meant a refused scan could reappear later as
a confident diagnosis simply because it was stored under a real class name.

The rule this module enforces: a pest name and its advisory are shown ONLY
for status "ok" or "uncertain". A refusal never gets a treatment
recommendation attached, because acting on one is exactly the harm the
refusal exists to prevent.
"""

import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADVISORY_PATH = os.path.join(BASE_DIR, "data", "advisory_data.json")

with open(ADVISORY_PATH, "r", encoding="utf-8") as f:
    ADVISORY_DB = json.load(f)

CONFIDENT_COLOR = [0.16, 0.60, 0.24, 1]   # green  - trust this
UNCERTAIN_COLOR = [0.85, 0.55, 0.10, 1]   # amber  - take with care
REFUSED_COLOR = [0.70, 0.25, 0.20, 1]     # red    - no diagnosis made

_UNKNOWN_ADVISORY = {
    "group": "Unknown",
    "type": "Unclassified organism",
    "description": ("This does not closely match any organism in our current "
                    "database."),
    "advisory": ("Consider consulting a local agricultural extension officer. "
                 "This image has been saved for expert review and future model "
                 "retraining."),
}


def present(status, label=None, confidence=0.0, headline="", detail=""):
    """
    Map a verdict onto everything the Result screen needs.

    `headline` / `detail` come from PestClassifier (and are persisted with
    the scan, so History replays the exact wording the farmer first saw).
    Both fall back to sensible defaults when absent - old history rows
    written before those columns existed have neither.

    Returns a dict of display fields; see ResultScreen for how each is bound.
    """
    status = status or "ok"

    if status == "unusable":
        return {
            "display_label": headline or "Couldn't use this photo",
            "group_text": "No result",
            "description_text": detail or (
                "The photo could not be analysed. Retake it in better light, "
                "holding the phone steady."),
            "advisory_text": "",
            "confidence_color": REFUSED_COLOR,
            "show_confidence": False,
            "show_advisory": False,
            "show_flag_button": False,
            "is_diagnosis": False,
        }

    if status == "not_plant":
        info = ADVISORY_DB.get("Not Plant", _UNKNOWN_ADVISORY)
        return {
            "display_label": headline or "No plant or leaf detected",
            "group_text": info["group"],
            "description_text": detail or info["description"],
            # Guidance on retaking the photo, NOT a treatment recommendation.
            "advisory_text": info["advisory"],
            "confidence_color": REFUSED_COLOR,
            "show_confidence": False,
            "show_advisory": True,
            "show_flag_button": False,
            "is_diagnosis": False,
        }

    if status == "uncertain":
        return {
            "display_label": headline or (
                f"Not sure - possible {label}" if label else "Not sure"),
            "group_text": "Uncertain",
            "description_text": detail or (
                "This looks like a plant, but the app cannot confidently tell "
                "which pest or disease it is."),
            "advisory_text": (
                "Do not treat based on this result alone. Retake the photo with "
                "the affected leaf filling most of the frame in good daylight. "
                "If it stays unsure, show it to an agricultural extension "
                "officer - the image has been saved for review."
            ),
            "confidence_color": UNCERTAIN_COLOR,
            "show_confidence": True,
            "show_advisory": True,
            "show_flag_button": True,
            "is_diagnosis": False,
        }

    # status == "ok"
    info = ADVISORY_DB.get(label, _UNKNOWN_ADVISORY)
    return {
        "display_label": label or "Result",
        "group_text": info["group"],
        "description_text": info["description"],
        "advisory_text": info["advisory"],
        "confidence_color": CONFIDENT_COLOR,
        "show_confidence": True,
        "show_advisory": True,
        "show_flag_button": True,
        "is_diagnosis": True,
    }


def collection_folder(status, label):
    """Where DataCollector should file the image.

    Refused scans are still worth keeping - they are exactly the training
    data the negative class needs to keep improving - but they must never be
    filed under a pest name, or the next retraining run learns that photos
    of walls are Healthy Leaf and the original bug comes straight back.
    """
    if status == "ok":
        return label
    if status == "uncertain":
        return "_uncertain"
    if status == "not_plant":
        return "_not_plant"
    return "_low_quality"
