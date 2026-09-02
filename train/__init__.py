"""Training and evaluation scripts (not packaged into the APK).

buildozer.spec excludes this whole directory from the build - these run on
a desktop/Colab machine to produce model/pest_model.tflite and
model/ood_stats.json, which are what actually ship.
"""
