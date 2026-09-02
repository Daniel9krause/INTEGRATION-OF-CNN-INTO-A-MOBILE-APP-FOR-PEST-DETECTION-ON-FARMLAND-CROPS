"""
test_decision_logic
--------------------
Tests the refusal pipeline in model/classifier.py and the display rules in
utils/result_presentation.py, with the network stubbed out.

Why stub the network? Because these are the rules that decide whether a
farmer is shown a diagnosis at all, and they must hold for probability
vectors the current trained model may never happen to produce. Testing them
only through the real model would mean the day a retrain shifts its outputs,
the guarantee quietly stops being tested.

The property that matters most is the last one: a refusal must never carry a
pest name or a treatment recommendation. That was the original bug - the old
screen printed "Uncertain (closest guess: Beetle)" for a photo of a person
and attached Beetle's pesticide advisory underneath.

Run:  python tests/test_decision_logic.py
"""

import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import classifier as C  # noqa: E402
from utils import result_presentation as RP  # noqa: E402

PASSED = 0
FAILED = 0

LABELS = ["Healthy Leaf", "Aphids", "Leaf Blight", "Beetle", "Not Plant"]
NEG = LABELS.index("Not Plant")
DIM = 8


def check(name, condition, detail=""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  PASS  {name}")
    else:
        FAILED += 1
        print(f"  FAIL  {name}  {detail}")


def make_classifier(centroids=True, threshold=0.5):
    """A PestClassifier with no model file, wired up by hand.

    Sidesteps TFLite entirely: predict() is exercised through a stubbed
    _infer, so the decision logic is what is under test.
    """
    clf = C.PestClassifier(model_path="/nonexistent", labels_path="/nonexistent",
                           ood_stats_path="/nonexistent")
    clf.labels = list(LABELS)
    clf.has_negative_class = True
    clf.interpreter = object()          # non-None: "a model is loaded"
    if centroids:
        # Four unit centroids on distinct axes; an embedding pointing along
        # one of them scores 1.0, an unrelated direction scores ~0.
        cents = np.zeros((4, DIM), np.float32)
        for i in range(4):
            cents[i, i] = 1.0
        clf.centroids = cents
        clf.ood_threshold = threshold
    else:
        clf.centroids = None
        clf.ood_threshold = None
    return clf


def stub(clf, probs, embedding=None):
    probs = np.asarray(probs, np.float32)
    if embedding is None:
        embedding = np.zeros(DIM, np.float32)
        embedding[0] = 1.0          # squarely in-distribution
    clf._infer = lambda path: (probs, np.asarray(embedding, np.float32))


def good_photo(tmpdir):
    """A frame that passes the quality gate, so the gate is never the thing
    under test in this file.

    Uses a real dataset photo rather than synthetic noise: noise is sharp
    and high-contrast but has no structure, which is exactly what the gate's
    edge-density check is built to reject, so a synthetic fixture would fail
    for the right reason and make these tests unrunnable.
    """
    path = os.path.join(tmpdir, "ok.jpg")
    if os.path.exists(path):
        return path

    dataset = os.path.join(os.path.dirname(tmpdir), "..", "data", "dataset")
    for folder in ("Healthy Leaf", "Leaf Blight", "Leaf Spot"):
        d = os.path.normpath(os.path.join(dataset, folder))
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if not name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            candidate = os.path.join(d, name)
            from utils import image_quality
            if image_quality.assess(candidate)["usable"]:
                Image.open(candidate).convert("RGB").save(path, quality=95)
                return path
    raise RuntimeError("no dataset image passed the quality gate to use as a fixture")


def test_quality_gate_short_circuits(img):
    print("\nquality gate runs before the network")
    clf = make_classifier()
    stub(clf, [0.99, 0.003, 0.003, 0.003, 0.001])

    dark = os.path.join(os.path.dirname(img), "dark.jpg")
    Image.fromarray(np.zeros((240, 320, 3), np.uint8)).save(dark)
    r = clf.predict(dark)
    check("pitch-black frame -> unusable", r["status"] == "unusable", r["status"])
    check("no label offered for an unusable frame", r["label"] is None, r["label"])
    check("headline names the problem", "dark" in r["headline"].lower(), r["headline"])


def test_negative_class(img):
    print("\nnegative class veto")
    clf = make_classifier()

    stub(clf, [0.10, 0.05, 0.05, 0.05, 0.75])
    check("'Not Plant' wins -> not_plant", clf.predict(img)["status"] == "not_plant")

    # It does not have to win outright. A real leaf photo puts almost
    # nothing on the negative class, so a large minority share is already
    # strong evidence the subject is not a plant.
    stub(clf, [0.55, 0.05, 0.03, 0.02, 0.35])
    r = clf.predict(img)
    check("'Not Plant' at 0.35 vetoes a 0.55 top class",
          r["status"] == "not_plant", f"{r['status']}")

    stub(clf, [0.90, 0.04, 0.03, 0.02, 0.01])
    check("small negative share does not veto",
          clf.predict(img)["status"] == "ok")


def test_embedding_ood(img):
    print("\nembedding distance check")
    clf = make_classifier(threshold=0.5)

    # Confident softmax, but the features sit nowhere near any class the
    # model was trained on. This is the case the negative class cannot
    # cover - objects nobody thought to include.
    far = np.zeros(DIM, np.float32)
    far[DIM - 1] = 1.0
    stub(clf, [0.97, 0.01, 0.01, 0.005, 0.005], embedding=far)
    r = clf.predict(img)
    check("confident softmax + far embedding -> not_plant",
          r["status"] == "not_plant", f"{r['status']} {r['diagnostics']}")
    check("closest class recorded for later review",
          r["diagnostics"].get("closest_plant_class") == "Healthy Leaf")

    near = np.zeros(DIM, np.float32)
    near[1] = 1.0
    stub(clf, [0.05, 0.90, 0.02, 0.02, 0.01], embedding=near)
    check("confident softmax + near embedding -> ok",
          clf.predict(img)["status"] == "ok")


def test_confidence_and_margin(img):
    print("\nconfidence and margin")
    clf = make_classifier()

    stub(clf, [0.30, 0.28, 0.22, 0.19, 0.01])
    r = clf.predict(img)
    check("low confidence -> uncertain", r["status"] == "uncertain", r["status"])
    check("uncertain still names the closest guess", r["label"] == "Healthy Leaf")

    # Two diseases neck and neck. High enough individually to clear the
    # confidence floor once renormalised, but choosing between them is a
    # coin flip - and the wrong pesticide costs money.
    stub(clf, [0.02, 0.50, 0.46, 0.01, 0.01])
    r = clf.predict(img)
    check("near-tie between two classes -> uncertain",
          r["status"] == "uncertain", f"{r['status']} margin={r['diagnostics']['margin']:.3f}")

    stub(clf, [0.02, 0.92, 0.03, 0.02, 0.01])
    check("clear winner -> ok", clf.predict(img)["status"] == "ok")


def test_probabilities_renormalised(img):
    print("\nplant probabilities exclude the negative class")
    clf = make_classifier()
    stub(clf, [0.02, 0.60, 0.13, 0.05, 0.20])
    r = clf.predict(img)
    total = sum(p for _, p in r["top3"])
    check("negative class absent from top3",
          all(lbl != "Not Plant" for lbl, _ in r["top3"]),
          r["top3"])
    check("reported confidence is share among plants (0.60/0.80=0.75)",
          abs(r["confidence"] - 0.75) < 1e-3, r["confidence"])
    check("top3 probabilities do not exceed 1", total <= 1.0 + 1e-6, total)


def test_no_ood_stats_is_stricter(img):
    """Losing a defence layer must tighten the remaining one, not silently
    weaken the whole pipeline. The same probabilities are fed to both
    configurations so the only variable is whether the OOD check exists."""
    print("\nmissing ood_stats.json raises the confidence floor")
    probs = [0.62, 0.20, 0.10, 0.06, 0.02]

    with_ood = make_classifier(centroids=True)
    stub(with_ood, probs)
    check("0.63 confidence is accepted while the OOD check is available",
          with_ood.predict(img)["status"] == "ok", with_ood.predict(img)["status"])

    without_ood = make_classifier(centroids=False)
    stub(without_ood, probs)
    r = without_ood.predict(img)
    check("the same result becomes uncertain without the OOD check",
          r["status"] == "uncertain", r["status"])


def test_presentation_never_advises_on_refusal():
    print("\na refusal never carries a pest name or treatment advice")
    for status in ("not_plant", "unusable"):
        view = RP.present(status, label="Beetle", confidence=0.97,
                          headline="No plant or leaf detected", detail="")
        check(f"{status}: pest name not shown",
              "Beetle" not in view["display_label"], view["display_label"])
        check(f"{status}: not marked as a diagnosis", not view["is_diagnosis"])
        check(f"{status}: confidence hidden", not view["show_confidence"])
        beetle_advice = RP.ADVISORY_DB["Beetle"]["advisory"]
        check(f"{status}: Beetle's advisory not shown",
              view["advisory_text"] != beetle_advice)

    view = RP.present("uncertain", label="Beetle", confidence=0.4)
    check("uncertain: names the guess but is not a diagnosis",
          "Beetle" in view["display_label"] and not view["is_diagnosis"])
    check("uncertain: advisory says not to treat on this alone",
          "not treat" in view["advisory_text"].lower(), view["advisory_text"])

    view = RP.present("ok", label="Beetle", confidence=0.93)
    check("ok: shows the real advisory",
          view["advisory_text"] == RP.ADVISORY_DB["Beetle"]["advisory"])
    check("ok: is a diagnosis", view["is_diagnosis"])


def test_collection_folders():
    print("\nrefused images are never filed under a pest name")
    check("ok -> the class folder", RP.collection_folder("ok", "Beetle") == "Beetle")
    check("not_plant -> _not_plant",
          RP.collection_folder("not_plant", "Beetle") == "_not_plant")
    check("uncertain -> _uncertain",
          RP.collection_folder("uncertain", "Beetle") == "_uncertain")
    check("unusable -> _low_quality",
          RP.collection_folder("unusable", "Beetle") == "_low_quality")


def main():
    tmpdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tmp")
    os.makedirs(tmpdir, exist_ok=True)
    img = good_photo(tmpdir)

    from utils import image_quality
    assert image_quality.assess(img)["usable"], "test fixture must pass the quality gate"

    print("=" * 60)
    print("decision logic tests")
    print("=" * 60)
    test_quality_gate_short_circuits(img)
    test_negative_class(img)
    test_embedding_ood(img)
    test_confidence_and_margin(img)
    test_probabilities_renormalised(img)
    test_no_ood_stats_is_stricter(img)
    test_presentation_never_advises_on_refusal()
    test_collection_folders()
    print("\n" + "=" * 60)
    print(f"{PASSED} passed, {FAILED} failed")
    print("=" * 60)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
