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
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget 
from . import config
from .model import build_model, load_checkpoint
from .transforms import get_val_transforms

def generate_gradcam(model, image_path: Path, target_layer, device: str = "cpu") -> np.ndarray:
   """
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
   pil_img = Image.open(image_path).convert("RGB")
   pil_img_resized = pil_img.resize((config.IMAGE_SIZE, config.IMAGE_SIZE))
   img_np = np.array(pil_img_resized)
   rgb_img_0_to_1 = img_np.astype(np.float32) / 255.0
   transformed = get_val_transforms()(image=img_np)
   input_tensor = transformed["image"].unsqueeze(0).to(device)
   with GradCAM(model=model, target_layers=[target_layer]) as cam:
      grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(1)])[0]
   visualization = show_cam_on_image(rgb_img_0_to_1, grayscale_cam, use_rgb=True)
   return visualization

def save_gradcam_examples(model, image_paths: list[Path], output_dir: Path, device: str = "cpu") -> None:
   """
   loop over image_paths, call generate_gradcam() on each, and save
   the result (e.g. with PIL.Image.fromarray(...).save(...)) into
   output_dir. Pick a handful of correctly-classified AND a handful of
   misclassified examples from your evaluate.py results — the
   misclassified ones are often the most interesting to look at.
   """
   output_dir.mkdir(parents=True, exist_ok=True)
   if config.BACKBONE == "resnet18":
      target_layer = model.layer4[-1]
   else:
      target_layer = model.features[-1]
   for img_path in image_paths:
      visualization = generate_gradcam(model, img_path, target_layer, device)
      out_path = output_dir / f"cam_{img_path.stem}.jpg"
      Image.fromarray(visualization).save(out_path)


if __name__ == "__main__":
   """
   load your trained model + checkpoint, pick some sample images
   from your test set, call save_gradcam_examples().
   """
   if torch.cuda.is_available():
      device = "cuda"
   else:
      device = "cpu"
   model = build_model()
   checkpoint_path = config.CHECKPOINT_DIR / "best_model.pth"
   if checkpoint_path.exists():
      load_checkpoint(model, checkpoint_path, device)
      model = model.to(device)
      samples = [
         config.DATA_DIR / "HAM10000_images_part_1" / "ISIC_0024325.jpg",
         config.DATA_DIR / "HAM10000_images_part_2" / "ISIC_0029350.jpg"
      ]
      output_destination = Path("outputs/gradcam")
      print(f"Generating diagnostic heatmaps inside: {output_destination}")
      save_gradcam_examples(model, samples, output_destination, device)
      print("Finished")
   else:
      print("Could not find model weights, please complete training first.")
