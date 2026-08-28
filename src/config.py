"""
Central place for paths, hyperparameters, and the class list.

Fill in the values below. Keeping them here (instead of scattered as magic
numbers through train.py / dataset.py / evaluate.py) means you change one
file when you want to try a different learning rate or image size.
"""

from pathlib import Path

# ---- Paths ----
# TODO: point these at wherever you unzipped the HAM10000 download
DATA_DIR = Path("data/raw")
METADATA_CSV = DATA_DIR / "HAM10000_metadata.csv"
IMAGE_DIRS = [
    # TODO: list both image folders here, e.g.
    DATA_DIR / "HAM10000_images_part_1",
    DATA_DIR / "HAM10000_images_part_2",
]

CHECKPOINT_DIR = Path("checkpoints")

# ---- Classes ----
# Binary cancer-detection task: every HAM10000 image gets regrouped from its
# original 7-way `dx` label into one of these two.
CLASSES = ["benign", "malignant"]

# Which of the original `dx` codes count as malignant. Everything else in the
# dataset (nv, bkl, df, vasc) is treated as benign.
#
# This is the most common grouping in published work on this dataset. Some
# papers instead treat "akiec" as benign since it's precancerous rather than
# invasive — if you want to try that convention, just remove it from this set.
MALIGNANT_DX = {"mel", "bcc", "akiec"}


def dx_to_label(dx: str) -> int:
    """
    Map a raw `dx` string from the metadata CSV to a binary class index
    (0 = benign, 1 = malignant), using MALIGNANT_DX above.

    TODO: implement this — it's one line, but dataset.py depends on it, so
    write it now. Return CLASSES.index("malignant") if dx is in
    MALIGNANT_DX, else CLASSES.index("benign").
    """
    if dx in MALIGNANT_DX:
        return CLASSES.index("malignant")
    else:
        return CLASSES.index("benign")

# ---- Model ----
# TODO: pick a torchvision backbone, e.g. "resnet18" or "efficientnet_b0"
BACKBONE = "efficientnet_b0"
IMAGE_SIZE = 224  # TODO: e.g. 224 (must match what your backbone expects)

# ---- Training ----
BATCH_SIZE = 32         # TODO
NUM_EPOCHS = 15         # TODO
LEARNING_RATE = .0001   # TODO
VAL_SPLIT = .15         # TODO: fraction of data held out for validation, e.g. 0.15
TEST_SPLIT = .15        # TODO: fraction held out for test, e.g. 0.15
RANDOM_SEED = 42

# ---- Weights & Biases ----
WANDB_PROJECT = "skin-cancer-classifier"
WANDB_ENTITY = "JadenF5"  # TODO: your wandb username or team
