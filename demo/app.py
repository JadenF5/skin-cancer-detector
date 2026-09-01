"""
Using Render.
"""
import gradio as gr
import torch
import os
from PIL import Image
import numpy as np
from src import config
from src.model import build_model, load_checkpoint
from src.transforms import get_val_transforms

# load your trained model once, same as in api/main.py
device = "cuda" if torch.cuda.is_available() else "cpu"
model = build_model(pretrained=False)  # TODO
model = load_checkpoint(model, config.CHECKPOINT_DIR / "best_model.pth", device)
model.eval()

def predict(image: Image.Image) -> dict:
    """
    TODO:
    1. Apply get_val_transforms() to the input PIL image.
    2. Run it through `model`, softmax the output to get
       [p_benign, p_malignant].
    3. Return {"benign": p_benign, "malignant": p_malignant} — Gradio's
       gr.Label output component renders a dict of {class_name: probability}
       as a nice bar chart automatically.
    """
    img_np = np.array(image)
    transformed = get_val_transforms()(image=img_np)
    input_tensor = transformed["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)
        p_benign = probabilities[0, 0].item()
        p_malignant = probabilities[0, 1].item()
    return {"benign": p_benign, "malignant": p_malignant}


# TODO: build the Render interface. Something like:
# demo = gr.Interface(
#     fn=predict,
#     inputs=gr.Image(type="pil"),
#     outputs=gr.Label(num_top_classes=2),
#     title="Skin Cancer Detector",
#     description=(
#         "Portfolio project — trained on the public HAM10000 dataset to "
#         "distinguish malignant from benign skin lesions. "
#         "NOT A DIAGNOSTIC TOOL. For demonstration purposes only — see a "
#         "dermatologist for any real skin concern."
#     ),
# )
# Keep that disclaimer in the description verbatim — don't soften or
# remove it even for a portfolio demo.
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=2),
    title="Skin Cancer Detector",
    description=(
        "Portfolio project — trained on the public HAM10000 dataset to "
        "distinguish malignant from benign skin lesions. "
        "NOT A DIAGNOSTIC TOOL. For demonstration purposes only — see a "
        "dermatologist for any real skin concern."
    ),
)

if __name__ == "__main__":
    if demo is not None:
        demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
