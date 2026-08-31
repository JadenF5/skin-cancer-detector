"""
FastAPI inference service — you already know this framework from the audio
analyzer project, so this file reuses that skill rather than teaching a new
one. The new part here is Dockerizing it (see ../Dockerfile).

Returns a benign/malignant call with a confidence score — not a diagnosis.
Keep that distinction in the response shape and in any UI built on top of it.

Run locally with: uvicorn api.main:app --reload
Then visit http://localhost:8000/docs for the interactive API docs.
"""

from io import BytesIO

from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
import numpy as np
from src import config
from src.model import build_model, load_checkpoint
from src.transforms import get_val_transforms

app = FastAPI(title="Skin Cancer Classifier API")

# load trained model once at startup (not per-request — that would be slow). Something like:
#   device = "cpu"
#   model = build_model(pretrained=False)
#   model = load_checkpoint(model, config.CHECKPOINT_DIR / "best_model.pth", device)
#   model.eval()
device = "cpu"
model = build_model(pretrained=False)
model = load_checkpoint(model, config.CHECKPOINT_DIR / "best_model.pth", device)
model.eval()

@app.get("/health")
def health():
    """Simple endpoint to confirm the service is up. Already implemented for you."""
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
   """
   1. Read the uploaded file into a PIL Image:
      contents = await file.read()
      image = Image.open(BytesIO(contents)).convert("RGB")
   2. Apply get_val_transforms() to it (remember: Albumentations wants a
      numpy array, and returns a dict — see the note in dataset.py).
   3. Add a batch dimension (unsqueeze(0)) and run it through `model`.
   4. Softmax the logits to get [p_benign, p_malignant].
   5. Return a JSON-serializable dict, e.g.:
      {"predicted_class": "malignant", "confidence": 0.87,
         "p_benign": 0.13, "p_malignant": 0.87,
         "disclaimer": "Not a diagnostic tool. For demonstration purposes only."}
      Keep that disclaimer field in the response — this isn't just UI
      copy, it's part of the contract of the API.
   """
   contents = await file.read()
   image = Image.open(BytesIO(contents)).convert("RGB")
   img_np = np.array(image)
   transformed = get_val_transforms()(image=img_np)
   input_tensor = transformed["image"].unsqueeze(0).to(device)
   with torch.no_grad():
      logits = model(input_tensor)
      probabilities = torch.softmax(logits, dim=1)
      p_benign = probabilities[0, 0].item()
      p_malignant = probabilities[0, 1].item()
      if p_malignant > p_benign:
         classification = "malignant"
         confidence = p_malignant
      else:
         classification = "benign"
         confidence = p_benign
      return {"predicted_class": classification,
              "confidence": confidence,
              "p_benign": p_benign,
              "p_malignant": p_malignant,
              "disclaimer": "Not a diagnostic tool. For demonstration purposes only."
      }
