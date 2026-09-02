"""
train_model.py
----------------
Trains the MobileNetV2 pest/disease classifier for FDCS and exports it to
model/pest_model.tflite, plus model/ood_stats.json - the two files
model/classifier.py loads at startup.

HOW TO USE
----------
1. Install training deps:
       pip install -r requirements-desktop.txt

2. Gather images into one folder per class, matching model/labels.txt
   EXACTLY (same spelling/spacing), under data/dataset/<Class Name>/*.jpg
   Aim for 150+ images per class minimum; 300+ is much better.

   The "Not Plant" class is special - fetch it automatically:
       python train/fetch_negatives.py

3. Run from the project root:
       python train/train_model.py
   CPU-only takes roughly 15-60 min depending on dataset size. Google Colab
   gives free GPU time if you want it faster.

4. model/pest_model.tflite and model/ood_stats.json are written. Restart the
   app - no other code changes needed.

WHAT IT DOES
------------
Two-phase transfer learning on ImageNet-pretrained MobileNetV2:
    Phase 1 - freeze the pretrained base, train only the new head (fast).
    Phase 2 - unfreeze the top of the base, fine-tune at a low learning
              rate (slower, meaningfully better).

Then two export steps that exist to stop the model bluffing:

  * TWO OUTPUTS. The exported graph returns both the class probabilities
    and the 1280-d pooled feature vector ("embedding") they were computed
    from. Both come out of the same single forward pass, so the extra head
    costs no inference time on the phone.

  * OOD STATS. After conversion, every training image is pushed through the
    *converted TFLite model* (not the Keras one - quantisation shifts the
    features slightly, and the app runs the TFLite version) to get the
    average feature vector of each plant class. model/classifier.py
    compares a new photo's features against those centroids and refuses to
    diagnose anything that sits far from all of them. That is what lets a
    closed-set classifier say "this is not a plant" for objects nobody
    thought to put in the training set.

The negative "Not Plant" class is deliberately EXCLUDED from those
centroids: the question the check needs to answer is "does this look like a
known plant or pest?", and including a negative centroid would let unusual
non-plant photos pass by being similar to other non-plant photos.
"""

import hashlib
import json
import os
import sys

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "data", "dataset")
LABELS_PATH = os.path.join(BASE_DIR, "model", "labels.txt")
OUT_MODEL_PATH = os.path.join(BASE_DIR, "model", "pest_model.tflite")
OUT_OOD_PATH = os.path.join(BASE_DIR, "model", "ood_stats.json")

IMG_SIZE = (224, 224)          # must match model/classifier.py's INPUT_SIZE
BATCH_SIZE = 16
VAL_SPLIT = 0.2
SEED = 42
HEAD_EPOCHS = 8                 # phase 1: frozen base
FINE_TUNE_EPOCHS = 10           # phase 2: fine-tuning
FINE_TUNE_AT_LAYER = 100        # unfreeze base layers from this index onward

# Must match model/classifier.py's NEGATIVE_LABEL.
NEGATIVE_LABEL = "Not Plant"

# Fraction of genuine plant photos we accept losing to the embedding check.
# The threshold is set at this percentile of in-distribution similarity, so
# 3.0 means "keep 97% of real leaf photos". Raising it rejects more junk but
# starts turning away real scans, which farmers experience as the app being
# broken - the negative class and the quality gate are the cheaper places to
# buy strictness.
OOD_PERCENTILE = 3.0

# Ceiling on any single class's training weight - see build_datasets().
MAX_CLASS_WEIGHT = 4.0


def load_labels():
    with open(LABELS_PATH, encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]
    if len(labels) < 2:
        sys.exit(f"model/labels.txt needs at least 2 class names, found {len(labels)}.")
    return labels


def class_counts(labels):
    """Image count per class folder - used to sanity-check the dataset and
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
        hint = ""
        if NEGATIVE_LABEL in missing:
            hint = (f"\n\nThe '{NEGATIVE_LABEL}' class is what lets the app reject "
                    f"photos that are not plants at all. Populate it with:\n"
                    f"    python train/fetch_negatives.py")
        sys.exit(
            "These classes have no images yet under data/dataset/:\n  "
            + "\n  ".join(missing)
            + "\nEvery class in model/labels.txt needs at least a few images "
              "(150+ recommended) before training. See data/dataset/README.md."
            + hint
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
        DATASET_DIR, validation_split=VAL_SPLIT, subset="training", seed=SEED,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="categorical",
        class_names=labels,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR, validation_split=VAL_SPLIT, subset="validation", seed=SEED,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="categorical",
        class_names=labels,
    )

    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.15),
        tf.keras.layers.RandomContrast(0.15),
        # Farmers scan in every light there is - dawn, overcast, deep shade,
        # harsh noon sun. Without brightness jitter the model learns the
        # even studio lighting of the reference photos and gets brittle in
        # the field, which is part of how dim shots ended up as confident
        # "Healthy Leaf". Applied before normalisation, so on the 0-255
        # scale image_dataset_from_directory produces.
        tf.keras.layers.RandomBrightness(0.25, value_range=(0, 255)),
    ])

    train_ds = train_ds.map(lambda x, y: (augment(x, training=True), y))
    train_ds = train_ds.map(mobilenet_preprocess).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(mobilenet_preprocess).prefetch(tf.data.AUTOTUNE)

    # Balanced weighting: total / (num_classes * count), the standard
    # "inverse frequency, mean weight 1" formula.
    #
    # This replaced a plain max(counts)/count, which produced the same
    # RATIOS but scaled every weight up by max/mean. That was harmless while
    # the largest class held 108 images; adding the 500+ image "Not Plant"
    # class multiplied every weight roughly fivefold, which - since the loss
    # is weight-scaled - quietly amounts to a fivefold learning-rate rise
    # and destabilises the low-LR fine-tuning phase. Normalising to mean 1
    # keeps the effective learning rate fixed no matter how the dataset
    # grows.
    total = sum(counts.values())
    n = len(labels)
    class_weight = {}
    for i, label in enumerate(labels):
        w = total / (n * counts[label])
        # Cap the rarest classes. Stem Borer has 40 images; letting it pull
        # an unbounded weight makes the model chase a handful of examples
        # and lose accuracy everywhere else. More images is the real fix -
        # see data/dataset/README.md.
        class_weight[i] = min(w, MAX_CLASS_WEIGHT)

    print("[train] Class weights (balanced, capped at "
          f"{MAX_CLASS_WEIGHT}):")
    for i, label in enumerate(labels):
        capped = " (capped)" if total / (n * counts[label]) > MAX_CLASS_WEIGHT else ""
        print(f"    {label:<16} {class_weight[i]:.2f}{capped}")

    return train_ds, val_ds, class_weight


def build_model(num_classes):
    """Returns (train_model, export_model, base).

    Both models share the very same layer objects, so training `train_model`
    trains `export_model` too. They differ only in what they return:
    training needs one output to compute a loss against, while the app needs
    the embedding as well for its out-of-distribution check.
    """
    base = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet"
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = base(inputs, training=False)
    embedding = tf.keras.layers.GlobalAveragePooling2D(name="embedding")(x)
    x = tf.keras.layers.Dropout(0.3)(embedding)
    probs = tf.keras.layers.Dense(num_classes, activation="softmax", name="probs")(x)

    train_model = tf.keras.Model(inputs, probs)
    export_model = tf.keras.Model(inputs, [probs, embedding])
    return train_model, export_model, base


def per_class_report(model, val_ds, labels):
    """Accuracy per class on the validation split. Worth reading closely for
    the negative class specifically: if 'Not Plant' recall is poor, the app
    will go back to guessing diseases for photos of walls and people."""
    y_true, y_pred = [], []
    for xb, yb in val_ds:
        p = model.predict(xb, verbose=0)
        y_true.extend(np.argmax(yb.numpy(), axis=1))
        y_pred.extend(np.argmax(p, axis=1))
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    print("\n[train] Per-class validation accuracy:")
    for i, label in enumerate(labels):
        mask = y_true == i
        if not mask.any():
            print(f"    {label:<16} (no validation samples)")
            continue
        acc = float((y_pred[mask] == i).mean())
        flag = "  <- weak" if acc < 0.6 else ""
        marker = "  *negative class*" if label == NEGATIVE_LABEL else ""
        print(f"    {label:<16} {acc * 100:5.1f}%  (n={int(mask.sum())}){flag}{marker}")
    return y_true, y_pred


def convert_to_tflite(export_model):
    print("\n[train] Converting to TFLite (2 outputs: probabilities + embedding)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(export_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]  # shrink for mobile
    tflite_model = converter.convert()
    os.makedirs(os.path.dirname(OUT_MODEL_PATH), exist_ok=True)
    with open(OUT_MODEL_PATH, "wb") as f:
        f.write(tflite_model)
    size_mb = len(tflite_model) / 1e6
    print(f"[train] Wrote {OUT_MODEL_PATH} ({size_mb:.1f} MB)")
    return tflite_model


# --------------------------------------------------------------------------
# Out-of-distribution reference stats
# --------------------------------------------------------------------------

def _dataset_files(labels):
    """Deterministic (path, class_index) listing of the whole dataset."""
    items = []
    for idx, label in enumerate(labels):
        folder = os.path.join(DATASET_DIR, label)
        if not os.path.isdir(folder):
            continue
        for name in sorted(os.listdir(folder)):
            if name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                items.append((os.path.join(folder, name), idx))
    return items


def _is_holdout(path):
    """Stable 80/20 split keyed on the filename hash.

    Deliberately independent of Keras's own shuffling: the centroids must
    come from one set of images and the acceptance threshold from a
    different one, or the threshold is measured on the very images that
    defined the centroids and comes out optimistically tight - which in the
    field means real leaf scans getting refused.
    """
    digest = hashlib.md5(os.path.basename(path).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 100 < 20


def _embed_all(interpreter, paths):
    in_detail = interpreter.get_input_details()[0]
    outs = interpreter.get_output_details()
    embed_detail = max(outs, key=lambda d: int(d["shape"][-1]))
    probs_detail = min(outs, key=lambda d: int(d["shape"][-1]))

    embeddings, predictions = [], []
    for i, path in enumerate(paths):
        if i % 200 == 0:
            print(f"      {i}/{len(paths)} ...")
        try:
            img = Image.open(path)
            img = ImageOps.exif_transpose(img).convert("RGB").resize(IMG_SIZE)
        except Exception:
            embeddings.append(None)
            predictions.append(None)
            continue
        arr = (np.asarray(img, dtype=np.float32) / 127.5) - 1.0
        interpreter.set_tensor(in_detail["index"], np.expand_dims(arr, 0))
        interpreter.invoke()
        embeddings.append(interpreter.get_tensor(embed_detail["index"])[0].copy())
        predictions.append(interpreter.get_tensor(probs_detail["index"])[0].copy())
    return embeddings, predictions


def _normalise(v):
    v = np.asarray(v, dtype=np.float32)
    return v / max(float(np.linalg.norm(v)), 1e-8)


def write_ood_stats(tflite_model, labels):
    """Push the whole dataset through the CONVERTED model and record what
    'looks like a known plant class' means in feature space."""
    print("\n[train] Computing out-of-distribution reference stats...")
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    if len(interpreter.get_output_details()) < 2:
        print("[train] WARNING: converted model has a single output - the "
              "embedding head did not survive conversion, so ood_stats.json "
              "will not be written and classifier.py will fall back to a "
              "stricter confidence floor.")
        return None

    items = _dataset_files(labels)
    paths = [p for p, _ in items]
    class_idx = [c for _, c in items]
    embeddings, _ = _embed_all(interpreter, paths)

    neg_index = labels.index(NEGATIVE_LABEL) if NEGATIVE_LABEL in labels else -1
    plant_indices = [i for i in range(len(labels)) if i != neg_index]

    # --- centroids, from the 80% "reference" side of the split -----------
    sums = {i: [] for i in plant_indices}
    for path, cls, emb in zip(paths, class_idx, embeddings):
        if emb is None or cls == neg_index or _is_holdout(path):
            continue
        sums[cls].append(_normalise(emb))

    centroids, centroid_labels = [], []
    for i in plant_indices:
        if not sums[i]:
            print(f"      skipping '{labels[i]}' - no reference images")
            continue
        centroids.append(_normalise(np.mean(np.stack(sums[i]), axis=0)))
        centroid_labels.append(labels[i])
    if not centroids:
        print("[train] WARNING: no centroids could be built; skipping ood_stats.json")
        return None
    centroids = np.stack(centroids)

    def max_similarity(emb):
        return float(np.max(centroids @ _normalise(emb)))

    # --- threshold, from the untouched 20% ------------------------------
    holdout_plant = [max_similarity(e) for p, c, e in zip(paths, class_idx, embeddings)
                     if e is not None and c != neg_index and _is_holdout(p)]
    negatives = [max_similarity(e) for c, e in zip(class_idx, embeddings)
                 if e is not None and c == neg_index]

    if not holdout_plant:
        print("[train] WARNING: empty holdout; skipping ood_stats.json")
        return None

    threshold = float(np.percentile(holdout_plant, OOD_PERCENTILE))

    hp = np.array(holdout_plant)
    print(f"      plant holdout similarity : min {hp.min():.3f}  "
          f"p{OOD_PERCENTILE:g} {threshold:.3f}  median {np.median(hp):.3f}  max {hp.max():.3f}")
    if negatives:
        ng = np.array(negatives)
        print(f"      'Not Plant' similarity   : min {ng.min():.3f}  "
              f"median {np.median(ng):.3f}  max {ng.max():.3f}")
        print(f"      -> embedding check alone rejects "
              f"{float((ng < threshold).mean()) * 100:.1f}% of non-plant images, "
              f"while keeping {float((hp >= threshold).mean()) * 100:.1f}% of real plant photos")

    stats = {
        "created_by": "train/train_model.py",
        # Binds these centroids to the exact .tflite they were measured in.
        # classifier.py refuses to use them against any other model file -
        # feature-space coordinates from a different network have the right
        # shape but no meaning, and would silently corrupt the OOD check.
        "model_sha256": hashlib.sha256(tflite_model).hexdigest(),
        "embedding_dim": int(centroids.shape[1]),
        "labels": centroid_labels,
        "centroids": centroids.tolist(),
        "similarity_threshold": threshold,
        "percentile": OOD_PERCENTILE,
        "notes": ("Cosine similarity to the nearest plant-class centroid. "
                  "Images scoring below similarity_threshold are refused as "
                  "out-of-distribution. Regenerate whenever the model is retrained "
                  "- centroids are only valid for the exact .tflite they came from."),
    }
    with open(OUT_OOD_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f)
    print(f"[train] Wrote {OUT_OOD_PATH} "
          f"({len(centroid_labels)} class centroids, threshold {threshold:.3f})")
    return stats


def main():
    labels = load_labels()
    print(f"[train] {len(labels)} classes: {labels}")
    if NEGATIVE_LABEL not in labels:
        print(f"[train] WARNING: '{NEGATIVE_LABEL}' is not in model/labels.txt. "
              f"Without it the model cannot say 'this is not a plant' and will "
              f"force every photo into a pest class. See train/fetch_negatives.py.")

    train_ds, val_ds, class_weight = build_datasets(labels)
    model, export_model, base = build_model(len(labels))

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="categorical_crossentropy", metrics=["accuracy"])

    print("\n[train] Phase 1: training classifier head (MobileNetV2 base frozen)...")
    model.fit(train_ds, validation_data=val_ds, epochs=HEAD_EPOCHS,
              class_weight=class_weight)

    print("\n[train] Phase 2: fine-tuning top layers of MobileNetV2...")
    base.trainable = True
    for layer in base.layers[:FINE_TUNE_AT_LAYER]:
        layer.trainable = False

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=FINE_TUNE_EPOCHS,
              class_weight=class_weight)

    val_loss, val_acc = model.evaluate(val_ds)
    print(f"\n[train] Final validation accuracy: {val_acc * 100:.1f}%")
    if val_acc < 0.6:
        print("[train] That's low - usually means too few images per class, "
              "or classes that look too similar. More data fixes this fastest.")

    per_class_report(model, val_ds, labels)

    tflite_model = convert_to_tflite(export_model)
    write_ood_stats(tflite_model, labels)

    print(f"\n[train] Done. Wrote:\n    {OUT_MODEL_PATH}\n    {OUT_OOD_PATH}")
    print("Verify the refusal behaviour before shipping:\n"
          "    python train/evaluate_rejection.py")


if __name__ == "__main__":
    main()
