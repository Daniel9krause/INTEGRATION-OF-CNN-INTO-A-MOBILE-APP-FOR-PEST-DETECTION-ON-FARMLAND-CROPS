"""
PestClassifier
---------------
Wraps the MobileNetV2 TFLite model and - just as importantly - decides when
NOT to trust it.

Files it uses (all under model/):
    pest_model.tflite   the trained network
    labels.txt          class names, one per line, in output order
    ood_stats.json      out-of-distribution reference stats, written by
                        train/train_model.py (optional; see below)

WHY THIS IS MORE THAN A THIN WRAPPER
------------------------------------
A softmax layer is a forced choice. Whatever you show it - a person, a
table, a wall, a pitch-black frame - it must distribute 100% of its belief
across the classes it knows, and it will happily do so at 97% confidence.
Measured on the previous build of this model:

    pure black    -> Healthy Leaf 97.4%      gray wall  -> Healthy Leaf 98.7%
    skin tone     -> Healthy Leaf 99.3%      random noise -> Healthy Leaf 99.1%
    photo of a person in a dark room -> Healthy Leaf 62.8%

Every one of those cleared the old 0.60 confidence threshold, which is why
raising that threshold was never going to help: the model was not unsure,
it was confidently wrong. Telling a farmer their crop is healthy when the
camera was pointing at a wall is the single worst failure this app can
have, so refusing to answer is treated as a first-class outcome here.

THREE INDEPENDENT LAYERS OF DEFENCE
-----------------------------------
1. QUALITY GATE (utils/image_quality) - runs before the network. Rejects
   frames that are too dark, blown out, blurred or featureless. Catches the
   "scanned at night, got Healthy Leaf" report outright.

2. NEGATIVE CLASS - the model is trained with a "Not Plant" class full of
   people, furniture, walls, animals, food, roads and dim indoor shots, so
   it can express "this is not a plant" directly instead of having to pick
   a disease. This is what stops Healthy Leaf acting as a sink for
   everything unfamiliar.

3. EMBEDDING DISTANCE (open-set / OOD check) - the negative class can only
   cover things we thought to include. So we also compare the image's
   penultimate-layer feature vector against the average feature vector of
   each training class. Something genuinely unlike anything the model was
   trained on lands far from every class centroid, and gets refused even if
   the softmax was confident and the negative class missed it. This is what
   generalises to the endless list of objects nobody put in the dataset.

Only if all three pass does a prediction get shown as a real diagnosis, and
even then a low margin between the top two classes downgrades it to
"uncertain". predict() returns a `status` describing which of these applied
- callers should branch on `status`, not on the confidence number.
"""

import hashlib
import json
import os

import numpy as np
from PIL import Image, ImageOps

from utils import image_quality

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "pest_model.tflite")
LABELS_PATH = os.path.join(BASE_DIR, "labels.txt")
OOD_STATS_PATH = os.path.join(BASE_DIR, "ood_stats.json")
INPUT_SIZE = (224, 224)  # MobileNetV2 default input

# Class name used for the negative/"none of the above" category. Must match
# the folder name under data/dataset/ and the line in labels.txt exactly.
NEGATIVE_LABEL = "Not Plant"

# --- Decision thresholds -------------------------------------------------
# Below this the top class is shown as "uncertain" rather than a diagnosis.
CONFIDENCE_THRESHOLD = 0.55
# The top two classes must be separated by at least this much. Two diseases
# at 0.34 / 0.31 is a coin flip dressed up as an answer, and acting on the
# wrong one costs a farmer real money in the wrong treatment.
MIN_MARGIN = 0.12
# "Not Plant" does not have to win outright to veto a diagnosis. If the
# model gives this much belief to "not a plant", refuse - a real leaf photo
# puts almost nothing here.
NEGATIVE_VETO_PROB = 0.30
# Fallback confidence gate used when ood_stats.json is absent and the
# embedding check cannot run, so a model built by an older training run is
# still held to a stricter standard rather than silently losing a layer.
NO_OOD_CONFIDENCE_FLOOR = 0.70


class PestClassifier:
    def __init__(self, model_path=MODEL_PATH, labels_path=LABELS_PATH,
                 ood_stats_path=OOD_STATS_PATH):
        self.labels = self._load_labels(labels_path)
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self._probs_output = None
        self._embed_output = None

        self.centroids = None          # (num_classes, dim), L2-normalised
        self.ood_threshold = None
        self.has_negative_class = NEGATIVE_LABEL in self.labels

        if os.path.exists(model_path):
            self._load_model(model_path)
        else:
            print(f"[PestClassifier] No model found at {model_path}. "
                  f"Running in MOCK inference mode until pest_model.tflite is added.")

        self._load_ood_stats(ood_stats_path, model_path)
        self._warn_about_missing_layers()

    # ------------------------------------------------------------------ setup

    def _load_labels(self, path):
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]
        return [f"Class {i}" for i in range(13)]

    def _load_model(self, model_path):
        """Load the TFLite interpreter. Tries tflite_runtime first (lighter,
        used on-device / Android), falls back to full tensorflow (desktop
        dev)."""
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            from tensorflow.lite.python.interpreter import Interpreter

        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self._identify_outputs()

    def _identify_outputs(self):
        """The trained model exports two heads: class probabilities and the
        pooled feature embedding the OOD check needs.

        They are matched by SHAPE, never by index - TFLite does not promise
        to preserve the order outputs were declared in, and silently
        swapping a 14-vector for a 1280-vector would produce garbage
        predictions rather than an error.
        """
        num_classes = len(self.labels)
        for detail in self.output_details:
            size = int(detail["shape"][-1])
            if size == num_classes and self._probs_output is None:
                self._probs_output = detail
            elif size > num_classes:
                self._embed_output = detail

        if self._probs_output is None:
            # Single-output model, or labels.txt out of sync with the model.
            self._probs_output = self.output_details[0]

        # Reconcile labels with what the model actually emits. This happens
        # for real during an upgrade - labels.txt gains "Not Plant" before
        # the retrained .tflite is dropped in - and indexing a 14-name list
        # into a 13-wide probability vector raised IndexError and took the
        # whole scan down. A loud warning plus a usable app beats a crash;
        # the mismatched name is dropped rather than silently trusted.
        emitted = int(self._probs_output["shape"][-1])
        if emitted != num_classes:
            print(f"[PestClassifier] WARNING: model outputs {emitted} classes "
                  f"but labels.txt lists {num_classes}. Retrain (see "
                  f"train/train_model.py) - until then the extra names are "
                  f"ignored and predictions may be mislabelled.")
            if emitted < num_classes:
                self.labels = self.labels[:emitted]
            else:
                self.labels = self.labels + [
                    f"Class {i}" for i in range(num_classes, emitted)
                ]
            self.has_negative_class = NEGATIVE_LABEL in self.labels

    @staticmethod
    def _file_sha256(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    def _load_ood_stats(self, path, model_path=None):
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                stats = json.load(f)

            # The centroids are coordinates in ONE model's feature space.
            # Swap the .tflite without regenerating them and the numbers
            # still have the right shape while meaning nothing - the check
            # would then reject real leaves and wave junk through, with no
            # visible symptom. Tie the two files together explicitly.
            recorded = stats.get("model_sha256")
            if recorded and model_path and os.path.exists(model_path):
                actual = self._file_sha256(model_path)
                if actual != recorded:
                    print("[PestClassifier] WARNING: model/ood_stats.json was "
                          "generated for a different pest_model.tflite. Ignoring "
                          "it and falling back to a stricter confidence floor. "
                          "Re-run train/train_model.py to regenerate both together.")
                    return

            centroids = np.asarray(stats["centroids"], dtype=np.float32)
            # Stored already normalised, but renormalising is cheap and
            # makes the cosine similarity below correct regardless.
            norms = np.linalg.norm(centroids, axis=1, keepdims=True)
            self.centroids = centroids / np.maximum(norms, 1e-8)
            self.ood_threshold = float(stats["similarity_threshold"])
            self._ood_labels = stats.get("labels", self.labels)
        except (OSError, ValueError, KeyError) as e:
            print(f"[PestClassifier] Ignoring unreadable {path}: {e}")
            self.centroids = None
            self.ood_threshold = None

    def _warn_about_missing_layers(self):
        if self.interpreter is None:
            return
        if not self.has_negative_class:
            print(f"[PestClassifier] NOTE: no '{NEGATIVE_LABEL}' class in labels.txt. "
                  f"Run train/fetch_negatives.py then retrain so the model can "
                  f"reject non-plant photos directly.")
        if self.centroids is None:
            print("[PestClassifier] NOTE: model/ood_stats.json missing - the "
                  "embedding-distance check is disabled and a stricter "
                  "confidence floor is used instead. Retrain to restore it.")

    # -------------------------------------------------------------- inference

    def _preprocess(self, image_path):
        img = Image.open(image_path)
        # Respect EXIF rotation: gallery uploads from a phone store the
        # sensor frame plus an orientation flag, and classifying a
        # sideways leaf costs real accuracy.
        img = ImageOps.exif_transpose(img).convert("RGB").resize(INPUT_SIZE)
        arr = np.asarray(img, dtype=np.float32)
        # MobileNetV2's ImageNet-pretrained weights expect inputs scaled to
        # [-1, 1], not [0, 1] - train/train_model.py normalizes the exact
        # same way, so this MUST stay in sync with that script.
        arr = (arr / 127.5) - 1.0
        return np.expand_dims(arr, axis=0)

    def _infer(self, image_path):
        """Run the network once. Returns (probs, embedding_or_None)."""
        input_data = self._preprocess(image_path)
        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()

        raw = self.interpreter.get_tensor(self._probs_output["index"])[0]
        probs = self._to_probabilities(raw)

        embedding = None
        if self._embed_output is not None:
            embedding = self.interpreter.get_tensor(self._embed_output["index"])[0]
        return probs, embedding

    @staticmethod
    def _to_probabilities(output):
        """train/train_model.py's exported model already ends in a
        Dense(..., activation="softmax"), so `output` is normally already a
        probability distribution. Re-applying softmax on top of that
        over-smooths confident predictions toward uniform - a genuine
        [0.97, 0.01, ...] collapses to about [0.19, 0.07, ...], which used
        to push every real prediction below threshold and label it
        "Uncertain" no matter how sure the model was. Only fall back to
        softmax for outputs that are not already probabilities (e.g. a
        differently-exported model ending in raw logits).
        """
        output = np.asarray(output, dtype=np.float32)
        if np.all(output >= -1e-6) and abs(float(output.sum()) - 1.0) < 1e-3:
            probs = np.clip(output, 0.0, None)
            return probs / max(float(probs.sum()), 1e-8)
        exp = np.exp(output - np.max(output))
        return exp / exp.sum()

    def _ood_similarity(self, embedding):
        """Cosine similarity to the NEAREST training-class centroid.

        High means "this looks like something the model has seen"; low means
        the image sits off the manifold the model was fit on, whatever the
        softmax claims. Returns None when stats are unavailable.
        """
        if embedding is None or self.centroids is None:
            return None
        vec = np.asarray(embedding, dtype=np.float32).ravel()
        if vec.shape[0] != self.centroids.shape[1]:
            return None
        vec = vec / max(float(np.linalg.norm(vec)), 1e-8)
        return float(np.max(self.centroids @ vec))

    # ---------------------------------------------------------------- predict

    def predict(self, image_path):
        """
        Classify an image, refusing when the evidence does not support an
        answer. Returns:

          {
            "status":     "ok" | "uncertain" | "not_plant" | "unusable",
            "label":      best in-distribution class name (may be None),
            "confidence": float 0-1 for that label,
            "top3":       [(label, prob), ...] over plant classes only,
            "headline":   short user-facing title
            "detail":     plain-language explanation / what to do next
            "quality":    raw metrics from the quality gate
            "diagnostics": model-internal numbers, for logging and tuning
          }

        `status` is the contract - callers must branch on it. Anything other
        than "ok" means no diagnosis should be presented or acted on.
        """
        quality = image_quality.assess(image_path)
        if not quality["usable"]:
            return {
                "status": "unusable",
                "label": None,
                "confidence": 0.0,
                "top3": [],
                "headline": quality["reason"],
                "detail": quality["detail"],
                "quality": quality["metrics"],
                "diagnostics": {},
            }

        if self.interpreter is None:
            return self._mock_predict(quality)

        probs, embedding = self._infer(image_path)
        similarity = self._ood_similarity(embedding)

        # Split the distribution into the real classes and the negative one.
        neg_index = self.labels.index(NEGATIVE_LABEL) if self.has_negative_class else None
        neg_prob = float(probs[neg_index]) if neg_index is not None else 0.0

        plant_probs = probs.copy()
        if neg_index is not None:
            plant_probs[neg_index] = 0.0
        plant_total = float(plant_probs.sum())
        # Renormalise so "how sure are we WHICH pest, given it is a plant"
        # is scored separately from "is this a plant at all". Without this,
        # a strong negative score would drag every plant class below the
        # confidence threshold and everything would read as "uncertain"
        # instead of the more useful "not a plant".
        plant_norm = plant_probs / plant_total if plant_total > 1e-8 else plant_probs

        order = np.argsort(plant_norm)[::-1]
        top3 = [(self.labels[i], float(plant_norm[i])) for i in order[:3]]
        top_label, top_conf = top3[0]
        margin = top_conf - (top3[1][1] if len(top3) > 1 else 0.0)

        diagnostics = {
            "negative_prob": neg_prob,
            "ood_similarity": similarity,
            "ood_threshold": self.ood_threshold,
            "margin": float(margin),
            "raw_top": float(probs.max()),
        }

        # --- Layer 2: the model's own "not a plant" verdict ---------------
        if neg_index is not None and (int(np.argmax(probs)) == neg_index
                                      or neg_prob >= NEGATIVE_VETO_PROB):
            return self._not_plant_result(top_label, quality, diagnostics)

        # --- Layer 3: does this look like anything we were trained on? ----
        if similarity is not None and similarity < self.ood_threshold:
            return self._not_plant_result(top_label, quality, diagnostics)

        # No embedding check available: demand more of the softmax instead.
        floor = CONFIDENCE_THRESHOLD if similarity is not None else NO_OOD_CONFIDENCE_FLOOR

        if top_conf < floor or margin < MIN_MARGIN:
            return {
                "status": "uncertain",
                "label": top_label,
                "confidence": top_conf,
                "top3": top3,
                "headline": "Not sure - possible " + top_label,
                "detail": (
                    "This does look like a plant, but the app cannot tell "
                    "confidently which pest or disease it is. Retake the photo "
                    "with the affected leaf filling most of the frame, in good "
                    "daylight, holding the phone steady. If it stays unsure, "
                    "show it to an agricultural extension officer before "
                    "treating - the photo has been saved for review."
                ),
                "quality": quality["metrics"],
                "diagnostics": diagnostics,
            }

        return {
            "status": "ok",
            "label": top_label,
            "confidence": top_conf,
            "top3": top3,
            "headline": top_label,
            "detail": "",
            "quality": quality["metrics"],
            "diagnostics": diagnostics,
        }

    def _not_plant_result(self, closest, quality, diagnostics):
        return {
            "status": "not_plant",
            "label": None,
            "confidence": 0.0,
            "top3": [],
            "headline": "No plant or leaf detected",
            "detail": (
                "This photo does not appear to show a plant, leaf or crop pest, "
                "so no diagnosis was made. Point the camera at the leaf, stem or "
                "insect you want checked and let it fill most of the frame, "
                "about 15-30 cm away in good light."
            ),
            "quality": quality["metrics"],
            "diagnostics": dict(diagnostics, closest_plant_class=closest),
        }

    def _mock_predict(self, quality):
        """Used only when no .tflite file is present, so the UI, database and
        advisory lookup stay testable during development.

        Deliberately returns "uncertain" rather than inventing a diagnosis:
        a mock that confidently names a disease is indistinguishable from a
        real prediction in screenshots and demos, and that is exactly how a
        placeholder ends up trusted in the field.
        """
        return {
            "status": "uncertain",
            "label": self.labels[0] if self.labels else None,
            "confidence": 0.0,
            "top3": [],
            "headline": "Model not installed",
            "detail": (
                "No trained model file was found on this device, so no real "
                "diagnosis can be made. Rebuild the app with "
                "model/pest_model.tflite included."
            ),
            "quality": quality["metrics"],
            "diagnostics": {"mock": True},
        }
