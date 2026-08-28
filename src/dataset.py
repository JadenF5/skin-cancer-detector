"""
PyTorch Dataset for HAM10000.

Goal: given the metadata CSV and the two image folders, produce a Dataset
that returns (image_tensor, label_index) pairs for a DataLoader.

New concept if you haven't done this before: a torch.utils.data.Dataset is
just a class that implements __len__ and __getitem__. DataLoader wraps it
to handle batching, shuffling, and parallel loading for you.
"""

from pathlib import Path
from typing import Optional, Callable
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from . import config


class HAM10000Dataset(Dataset):
    def __init__(
        self,
        metadata_df: pd.DataFrame,
        image_dirs: list[Path],
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            metadata_df: a (already filtered, e.g. train/val/test split) slice
                of the HAM10000 metadata CSV as a DataFrame.
            image_dirs: the list of folders to search for each image_id.
            transform: an Albumentations transform pipeline (see transforms.py)
                to apply to each image before returning it.
        - Store whatever you need from the args on self.
        - Consider building a dict of {image_id: full_path} once here in
          __init__ (by scanning image_dirs) rather than re-searching the
          folders on every __getitem__ call — much faster.
        """
        self.metadata_df = metadata_df
        self.transform = transform
        self.image_path_map = {}
        for directory in image_dirs:
            for file_path in directory.glob("*.jpg"):
                image_id = file_path.stem
                self.image_path_map[image_id] = file_path

    def __len__(self) -> int:
        """
        return the number of rows in your metadata slice.
        """
        return len(self.metadata_df)

    def __getitem__(self, idx: int):
        """
        1. Look up the row at position `idx` in your metadata.
        2. Find the corresponding image file on disk (use the image_id -> path
           mapping you built in __init__).
        3. Load it (PIL.Image.open(...).convert("RGB")).
        4. Look up the row's `dx` label and convert it to a binary class
           index using config.dx_to_label(dx_string) — this is where the
           7-way diagnosis gets collapsed into benign (0) / malignant (1).
        5. If self.transform is set, apply it to the image. Albumentations
           transforms expect a numpy array, not a PIL Image — you'll need
           `import numpy as np; np.array(pil_image)` first, and Albumentations
           returns a dict, so you'll pull the tensor out with `result["image"]`.
        6. Return (image_tensor, label_index).
        """
        row = self.metadata_df.iloc[idx]
        image_id = row["image_id"]
        path = self.image_path_map.get(image_id)
        pil_image = Image.open(path).convert("RGB")
        dx_string = row["dx"]
        label_index = config.dx_to_label(dx_string)
        if self.transform != None:
            image_np = np.array(pil_image)
            augmented = self.transform(image=image_np)
            image_tensor = augmented["image"]
        else:
            image_tensor = pil_image 
        return (image_tensor, label_index)


def build_splits(
    metadata_csv: Path = config.METADATA_CSV,
    val_frac: float = config.VAL_SPLIT,
    test_frac: float = config.TEST_SPLIT,
    seed: int = config.RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the metadata CSV and split it into train/val/test DataFrames.

    TODO:
    - Read the CSV with pandas.
    - Add a column applying config.dx_to_label to `dx`, so you can stratify
      on the *binary* label rather than the original 7-way `dx` — this
      matters more here than in a balanced multiclass setup, since malignant
      cases are only ~20% of the data and you want that ratio preserved in
      every split.
    - Do the stratified split (sklearn.model_selection.train_test_split has
      a `stratify=` argument — use it twice: once to carve off test, once
      more on the remainder to carve off val).
    - Return (train_df, val_df, test_df).

    Note: HAM10000 has some patients with multiple lesion images
    (`lesion_id` repeats). For a rigorous split you'd group by `lesion_id`
    so the same lesion never appears in both train and test — worth doing
    once your basic pipeline works, to avoid a leakage-inflated accuracy.
    """
    df = pd.read_csv(metadata_csv)
    df['new_label_column'] = df['dx'].apply()

    remaining_df, test_df = train_test_split(
    df,
    test_size=test_frac,          
    random_state=seed,    
    stratify=df['new_label_column'])

    adjusted_val_size = val_frac / (1.0 - test_frac)

    train_df, val_df = train_test_split(
    remaining_df,
    test_size=adjusted_val_size,         
    random_state=seed,
    stratify=remaining_df['new_label_column'])

    return train_df, val_df, test_df
