"""
Image augmentation pipelines using Albumentations.

New library: Albumentations is faster and has a larger transform library than
torchvision.transforms, and it's what you'll see used in most real-world CV
pipelines. Docs: https://albumentations.ai/docs/

Two pipelines:
- get_train_transforms(): augmentation + normalization, used only on the
  training set (helps the model generalize instead of memorizing).
- get_val_transforms(): resize + normalization only, no random augmentation
  (used on val/test — you want a stable, repeatable evaluation).
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

from . import config

def get_train_transforms() -> A.Compose:
    """
    build an A.Compose([...]) pipeline. At minimum you'll want:
    - A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE)
    - Some augmentations appropriate for dermatoscopic images, e.g.
      A.HorizontalFlip(), A.VerticalFlip(), A.RandomRotate90(),
      A.ColorJitter(...), A.ShiftScaleRotate(...)
      (skin lesion images don't have a "correct orientation," so flips/
      rotations are safe and effective here — unlike, say, photos of text)
    - A.Normalize(...) — use ImageNet mean/std if you're using an
      ImageNet-pretrained backbone: mean=[0.485, 0.456, 0.406],
      std=[0.229, 0.224, 0.225]
    - ToTensorV2() at the end, to convert to a torch tensor
    """
    return A.Compose([
      A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
      A.HorizontalFlip(p=0.5),
      A.VerticalFlip(p=0.5),
      A.RandomRotate90(p=0.5),
      A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
      A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5,),
      A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
      ToTensorV2(),
    ])

def get_val_transforms() -> A.Compose:
    """
    same as above but WITHOUT the random augmentations — just
    Resize -> Normalize -> ToTensorV2.
    """
    return A.Compose([
      A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
      A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
      ToTensorV2(),
    ])
