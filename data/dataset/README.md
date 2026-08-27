# Training dataset layout

`train/train_model.py` expects one folder per class **directly under this
directory**, named *exactly* like the corresponding line in
[`model/labels.txt`](../../model/labels.txt) (same spelling/spacing,
case-sensitive):

```
data/dataset/
├── Healthy Leaf/
│   ├── img001.jpg
│   └── ...
├── Aphids/
├── Armyworm/
├── Bollworm/
├── Grasshopper/
├── Stem Borer/
├── Whitefly/
├── Leaf Blight/
├── Leaf Spot/
├── Mosaic Virus/
├── Beetle/
├── Moth/
└── Weevil/
```

Nothing else goes in this folder — the training script scans exactly these
13 subfolders and **errors out immediately if it finds an extra folder
that isn't in `labels.txt`**, so don't drop a whole downloaded dataset in
here as-is — only copy over the class folders you actually need, renamed
to match the list above.

Put image files **directly inside** each class folder — no subfolders.
Aim for **150+ images per class minimum, 300+ is much better**; very
uneven counts between classes are fine (the script auto-weights
underrepresented classes), but a class with only a handful of images will
just perform badly no matter what. `.jpg`/`.jpeg`/`.png`/`.bmp`, any
resolution — the script resizes everything to 224×224 automatically.

## Current status

All 13 classes are populated and a model has been trained on them
(80.9% validation accuracy — see the main [README](../../README.md)'s
"Dataset provenance and known weak spots" section for exact per-class
counts, sources, and the deduplication/quality-filtering that was done
first). Most classes have 85–108 real images; four (Aphids, Armyworm,
Bollworm, Stem Borer) are thinner (40–74) after removing duplicate-file
padding. **Stem Borer (40 images) is the best candidate to grow next**
if you want to improve accuracy — add real images below, then rerun
`python train/train_model.py`.

## Where to get more images (if you want to grow a class later)

- **Insect classes** (Aphids, Armyworm, Bollworm, Grasshopper, Stem Borer,
  Whitefly, Beetle, Moth, Weevil):
  [Agricultural Pests Image Dataset](https://www.kaggle.com/datasets/vencerlanz09/agricultural-pests-image-dataset),
  [Pest Dataset](https://www.kaggle.com/datasets/simranvolunesia/pest-dataset),
  [Dangerous Farm Insects Dataset](https://www.kaggle.com/datasets/tarundalal/dangerous-insects-dataset).
- **Disease + healthy classes** (Healthy Leaf, Leaf Blight, Leaf Spot,
  Mosaic Virus):
  [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
  — pick and merge matching subclasses (e.g. `Tomato___Tomato_mosaic_virus`
  → `Mosaic Virus`, `Potato___Early_blight` + `Potato___Late_blight` →
  `Leaf Blight`).

None of these need the Kaggle CLI/API — click **Download** on the dataset
page in your browser, unzip, and copy the relevant images into the
matching folder above.

## Next step

Run `python train/train_model.py` from the project root.
