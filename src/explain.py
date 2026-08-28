"""
Model interpretability with Grad-CAM.

New library: pytorch-grad-cam (`pip install grad-cam`). Grad-CAM produces a
heatmap over the input image showing which regions most influenced the
model's "malignant" or "benign" call — useful for sanity-checking that your
model is actually looking at the lesion, not e.g. a ruler or marker pen
visible in some dermatoscopic images (a known real failure mode in skin
lesion datasets — worth mentioning if you find it happening to your model,
it's a genuinely interesting result, and especially worth checking for here
since a model that's right for the wrong reason is a real risk in a
cancer-detection framing).

Docs / examples: https://github.com/jacobgil/pytorch-grad-cam
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from . import config
from .model import build_model, load_checkpoint
from .transforms import get_val_transforms


def generate_gradcam(model, image_path: Path, target_layer, device: str = "cpu") -> np.ndarray:
    """
    TODO:
    1. Load and preprocess the image at image_path the same way your val
       transform does (resize + normalize), and also keep an unnormalized
       resized copy around (0-1 float array) for overlaying the heatmap on.
    2. Build `cam = GradCAM(model=model, target_layers=[target_layer])`
       (for a ResNet backbone, a good target layer is typically
       `model.layer4[-1]`).
    3. Run `grayscale_cam = cam(input_tensor=your_preprocessed_batch)[0]`
    4. Use `show_cam_on_image(rgb_img_0_to_1, grayscale_cam, use_rgb=True)`
       to get the heatmap overlaid on the original image.
    5. Return that overlaid image array.
    """
    raise NotImplementedError


def save_gradcam_examples(model, image_paths: list[Path], output_dir: Path, device: str = "cpu") -> None:
    """
    TODO: loop over image_paths, call generate_gradcam() on each, and save
    the result (e.g. with PIL.Image.fromarray(...).save(...)) into
    output_dir. Pick a handful of correctly-classified AND a handful of
    misclassified examples from your evaluate.py results — the
    misclassified ones are often the most interesting to look at.
    """
    raise NotImplementedError


if __name__ == "__main__":
    """
    TODO: load your trained model + checkpoint, pick some sample images
    from your test set, call save_gradcam_examples().
    """
    raise NotImplementedError
