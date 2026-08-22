"""
train_model.py
----------------
Trains the REAL MobileNetV2 pest/disease classifier for FDCS and exports it
straight to model/pest_model.tflite — the file model/classifier.py auto-
detects at startup. Until that file exists, the app runs in a mock mode
that guesses a random label instead of actually looking at your photo;
this script is what turns that off.

HOW TO USE
----------
1. Install training deps (same file as desktop testing — tensorflow-cpu
   is already in there):
       pip install -r requirements-desktop.txt

2. Gather images and sort them into one folder per class, matching
   model/labels.txt EXACTLY (same spelling/spacing), under:
       data/dataset/<Class Name>/*.jpg
   See data/dataset/README.md for where to get images for each class.
   Aim for 150+ images per class minimum; 300+ is much better.

3. Run this script from the project root:
       python train/train_model.py
   On a CPU-only laptop this can take anywhere from ~15 min to a couple
   hours depending on dataset size — that's normal for CPU training. If
   you have a Google account, Google Colab gives free GPU time: upload
   data/dataset there, run this same script, then download the resulting
   model/pest_model.tflite back into this project.

4. When it finishes, model/pest_model.tflite is written automatically.
   Just restart the app (python main.py) — no other code changes needed.

WHAT IT DOES
------------
Two-phase transfer learning on top of ImageNet-pretrained MobileNetV2:
    Phase 1 — freeze the pretrained base, train only the new
              classification head (fast).
    Phase 2 — unfreeze the top of the base and fine-tune everything at a
              low learning rate (slower, meaningfully improves accuracy).
Then converts the trained Keras model to a quantized .tflite file, small
and fast enough to run on a phone.
"""

import os
import sys

import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "data", "dataset")
LABELS_PATH = os.path.join(BASE_DIR, "model", "labels.txt")
OUT_MODEL_PATH = os.path.join(BASE_DIR, "model", "pest_model.tflite")

IMG_SIZE = (224, 224)          # must match model/classifier.py's INPUT_SIZE
BATCH_SIZE = 16
VAL_SPLIT = 0.2
SEED = 42
HEAD_EPOCHS = 8                 # phase 1: frozen base
FINE_TUNE_EPOCHS = 10           # phase 2: fine-tuning
FINE_TUNE_AT_LAYER = 100        # unfreeze base layers from this index onward


def load_labels():
    with open(LABELS_PATH) as f:
        labels = [line.strip() for line in f if line.strip()]
    if len(labels) < 2:
        sys.exit(f"model/labels.txt needs at least 2 class names, found {len(labels)}.")
    return labels


def class_counts(labels):
    """Image count per class folder — used to sanity-check the dataset and
    to weight underrepresented classes during training."""
    counts = {}
    for label in labels:
        folder = os.path.join(DATASET_DIR, label)
        if not os.path.isdir(folder):
            counts[label] = 0
            continue
        counts[label] = len([
            f for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ])
    return counts


def mobilenet_preprocess(x, y):
    # Must exactly match model/classifier.py's runtime preprocessing:
    # MobileNetV2 expects inputs scaled to [-1, 1].
    return (x / 127.5) - 1.0, y


def build_datasets(labels):
    if not os.path.isdir(DATASET_DIR):
        sys.exit(
            f"No dataset found at {DATASET_DIR}\n"
            f"Create one subfolder per class (named exactly like the lines "
            f"in model/labels.txt) and fill each with images before running "
            f"this script. See data/dataset/README.md."
        )

    counts = class_counts(labels)
    missing = [label for label, n in counts.items() if n == 0]
    if missing:
        sys.exit(
            "These classes have no images yet under data/dataset/:\n  "
            + "\n  ".join(missing)
            + "\nEvery class in model/labels.txt needs at least a few "
              "images (150+ recommended) before training. See "
              "data/dataset/README.md for sources."
        )

    print("[train] Images per class:")
    for label, n in counts.items():
        flag = "  <- low, consider adding more" if n < 50 else ""
        print(f"    {label}: {n}{flag}")

    # class_names=labels forces the model's output index order to match
    # model/labels.txt exactly (otherwise Keras defaults to alphabetical
    # folder order, which would silently mismatch classifier.py's labels
    # and scramble every prediction).
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=VAL_SPLIT,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        class_names=labels,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=VAL_SPLIT,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        class_names=labels,
    )

    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.15),
        tf.keras.layers.RandomContrast(0.15),
    ])

    train_ds = train_ds.map(lambda x, y: (augment(x, training=True), y))
    train_ds = train_ds.map(mobilenet_preprocess).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(mobilenet_preprocess).prefetch(tf.data.AUTOTUNE)

    class_weight = {
        i: max(counts.values()) / counts[label]
        for i, label in enumerate(labels)
    }
    return train_ds, val_ds, class_weight


def build_model(num_classes):
    base = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet"
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs), base


def main():
    labels = load_labels()
    print(f"[train] {len(labels)} classes: {labels}")

    train_ds, val_ds, class_weight = build_datasets(labels)
    model, base = build_model(len(labels))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("\n[train] Phase 1: training classifier head (MobileNetV2 base frozen)...")
    model.fit(train_ds, validation_data=val_ds, epochs=HEAD_EPOCHS, class_weight=class_weight)

    print("\n[train] Phase 2: fine-tuning top layers of MobileNetV2...")
    base.trainable = True
    for layer in base.layers[:FINE_TUNE_AT_LAYER]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(train_ds, validation_data=val_ds, epochs=FINE_TUNE_EPOCHS, class_weight=class_weight)

    val_loss, val_acc = model.evaluate(val_ds)
    print(f"\n[train] Final validation accuracy: {val_acc * 100:.1f}%")
    if val_acc < 0.6:
        print("[train] That's low — usually means too few images per class, "
              "or classes that look too similar. More data fixes this fastest.")

    print("\n[train] Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]  # shrink for mobile
    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(OUT_MODEL_PATH), exist_ok=True)
    with open(OUT_MODEL_PATH, "wb") as f:
        f.write(tflite_model)

    print(f"\n[train] Done! Model written to {OUT_MODEL_PATH}")
    print("Restart the app (python main.py) — it will auto-detect the model "
          "and stop using mock predictions.")


if __name__ == "__main__":
    main()
