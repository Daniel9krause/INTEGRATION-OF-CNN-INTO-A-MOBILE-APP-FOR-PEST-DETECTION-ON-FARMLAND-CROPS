"""
PestClassifier
---------------
Wraps a MobileNetV2 TFLite model trained on 13 pest/disease classes.

Drop your trained model file at: model/pest_model.tflite
Labels live at:                  model/labels.txt

If no .tflite model is present (e.g. during early development), the
classifier falls back to a deterministic mock prediction so the rest of
the app (UI, database, advisory lookup) remains fully testable.
"""

import os
import random
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "pest_model.tflite")
LABELS_PATH = os.path.join(BASE_DIR, "labels.txt")
INPUT_SIZE = (224, 224)  # MobileNetV2 default input


class PestClassifier:
    def __init__(self, model_path=MODEL_PATH, labels_path=LABELS_PATH):
        self.labels = self._load_labels(labels_path)
        self.interpreter = None
        self.input_details = None
        self.output_details = None

        if os.path.exists(model_path):
            self._load_model(model_path)
        else:
            print(f"[PestClassifier] No model found at {model_path}. "
                  f"Running in MOCK inference mode until pest_model.tflite is added.")

    def _load_labels(self, path):
        if os.path.exists(path):
            with open(path, "r") as f:
                return [line.strip() for line in f if line.strip()]
        return [f"Class {i}" for i in range(13)]

    def _load_model(self, model_path):
        """Load the TFLite interpreter. Tries tflite_runtime first (lighter,
        used on-device / Android), falls back to full tensorflow (desktop dev)."""
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            from tensorflow.lite.python.interpreter import Interpreter

        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def _preprocess(self, image_path):
        img = Image.open(image_path).convert("RGB").resize(INPUT_SIZE)
        arr = np.asarray(img, dtype=np.float32)
        # MobileNetV2's ImageNet-pretrained weights expect inputs scaled to
        # [-1, 1], not [0, 1] — train/train_model.py normalizes the exact
        # same way, so this MUST stay in sync with that script.
        arr = (arr / 127.5) - 1.0
        arr = np.expand_dims(arr, axis=0)
        return arr

    def predict(self, image_path):
        """
        Returns a dict:
          {
            "label": str,
            "confidence": float (0-1),
            "top3": [(label, confidence), ...]
          }
        """
        if self.interpreter is None:
            return self._mock_predict()

        input_data = self._preprocess(image_path)
        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]["index"])[0]

        # train/train_model.py's exported model already ends in a
        # Dense(..., activation="softmax") layer, so `output` is normally
        # already a valid probability distribution (non-negative, sums to
        # ~1). Re-applying softmax on top of that (as this used to do
        # unconditionally) over-smooths already-confident probabilities
        # toward a near-uniform distribution - e.g. a genuine [0.97, 0.01,
        # ...] prediction collapses to roughly [0.19, 0.07, ...], which
        # silently pushed every real prediction below result_screen.py's
        # CONFIDENCE_THRESHOLD and labeled it "Uncertain" regardless of
        # how sure the model actually was. Only fall back to softmax for
        # outputs that don't already look like probabilities (e.g. a
        # differently-exported model whose last layer is raw logits).
        if np.all(output >= -1e-6) and abs(float(output.sum()) - 1.0) < 1e-3:
            probs = np.clip(output, 0, None)
            probs = probs / probs.sum()
        else:
            exp = np.exp(output - np.max(output))
            probs = exp / exp.sum()

        top_indices = probs.argsort()[-3:][::-1]
        top3 = [(self.labels[i], float(probs[i])) for i in top_indices]

        return {
            "label": top3[0][0],
            "confidence": top3[0][1],
            "top3": top3,
        }

    def _mock_predict(self):
        """Deterministic-ish mock so the app is fully demoable before the
        trained .tflite model is dropped in."""
        label = random.choice(self.labels)
        confidence = round(random.uniform(0.72, 0.97), 4)
        others = random.sample([l for l in self.labels if l != label], 2)
        top3 = [(label, confidence)] + [
            (o, round(random.uniform(0.05, 0.2), 4)) for o in others
        ]
        return {"label": label, "confidence": confidence, "top3": top3}
