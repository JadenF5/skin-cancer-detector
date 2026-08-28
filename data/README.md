# Data

Download HAM10000 from Kaggle:
https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

Expected layout after downloading and unzipping:

```
data/
  raw/
    HAM10000_metadata.csv
    HAM10000_images_part_1/
      ISIC_0024306.jpg
      ...
    HAM10000_images_part_2/
      ISIC_0029306.jpg
      ...
```

The metadata CSV has one row per image with (among other columns) an `image_id`
and a `dx` column — `dx` is the diagnosis label you'll be classifying
(e.g. `nv`, `mel`, `bkl`, `bcc`, `akiec`, `vasc`, `df`).

You'll want to write a small one-time script (or just do this in `dataset.py`'s
`__init__`) that maps each `image_id` to whichever of the two image folders it's
actually in, since the images are split across two directories.

This folder is gitignored — don't commit the raw images to your repo.
