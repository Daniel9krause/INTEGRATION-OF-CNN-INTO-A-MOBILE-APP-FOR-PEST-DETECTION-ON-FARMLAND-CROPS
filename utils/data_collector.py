"""
DataCollector
--------------
Every image captured in the field is real-world data. Instead of throwing
it away after classification, we file it into a class-labeled folder so
the dataset keeps growing for future model retraining:

    collected_data/
        Aphids/
            2026-08-21_14-03-11.jpg
        Whitefly/
            2026-08-21_14-05-40.jpg
        _unclassified_new/          <- images the user flags as "not in this list"
            2026-08-21_14-10-02.jpg

This mirrors how PlantVillage-style datasets are structured (one folder
per class), so collected images can be merged straight into training data
later without any extra reshaping.
"""

import os
import shutil
from datetime import datetime
from kivy.utils import platform

if platform == "android":
    from android.storage import app_storage_path
    ROOT_DIR = app_storage_path()
else:
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

COLLECTED_DIR = os.path.join(ROOT_DIR, "collected_data")
UNCLASSIFIED_DIR_NAME = "_unclassified_new"


class DataCollector:
    def __init__(self, base_dir=COLLECTED_DIR):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def _safe_folder_name(self, label):
        return label.strip().replace(" ", "_")

    def save(self, image_path, predicted_label, flagged_new=False):
        """
        Copies the captured image into its class folder.
        If the user flags it as an unrecognized / new organism, it's routed
        to `_unclassified_new` instead, ready for expert review before being
        added as a brand-new class.
        Returns the new stored path.
        """
        folder_name = UNCLASSIFIED_DIR_NAME if flagged_new else self._safe_folder_name(predicted_label)
        target_dir = os.path.join(self.base_dir, folder_name)
        os.makedirs(target_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        ext = os.path.splitext(image_path)[1] or ".jpg"
        target_path = os.path.join(target_dir, f"{timestamp}{ext}")

        shutil.copy2(image_path, target_path)
        return target_path

    def dataset_summary(self):
        """Returns {class_name: image_count} for everything collected so far —
        useful for showing the app is actively growing its own dataset."""
        summary = {}
        if not os.path.exists(self.base_dir):
            return summary
        for folder in sorted(os.listdir(self.base_dir)):
            folder_path = os.path.join(self.base_dir, folder)
            if os.path.isdir(folder_path):
                summary[folder] = len([
                    f for f in os.listdir(folder_path)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ])
        return summary
