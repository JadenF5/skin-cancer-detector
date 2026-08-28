# Skin Cancer Detector — Health AI Project

A deep-learning binary classifier that flags dermatoscopic skin lesion images as
**malignant (cancer)** or **benign**, with model interpretability (Grad-CAM), an
inference API, and an interactive demo.

This started as a 7-class lesion-type classifier and was deliberately narrowed to
a binary cancer/benign task — that's the more common and more clinically
meaningful framing you'll see in published work on this dataset, and it's a
better story for a portfolio piece: "detects likely skin cancer" is a clearer
pitch than "sorts lesions into 7 diagnostic categories."

This is a **scaffold, not a finished project**. Every function you need is stubbed out
with a docstring describing exactly what it should do. Your job is to fill them in.
That's on purpose — you'll actually understand every piece if you write it yourself,
and you'll be able to talk through it confidently in an interview.

## Why this project (and why these frameworks)

Your current resume is strong on classical ML (Gradient Boosting, Random Forest),
audio signal processing, and chatbot/RAG pipelines. This project is deliberately
picked to fill in the gaps that Health AI / ML Engineer job postings ask for most
often and that you don't have proof of yet:

| Framework | What it's for here | Why it matters for job apps |
|---|---|---|
| **PyTorch (transfer learning)** | Fine-tune a pretrained CNN (ResNet18/EfficientNet) on medical images | You list PyTorch on your resume but don't have a deep-learning *vision* project to back it up — this closes that gap |
| **Albumentations** | Image augmentation (rotation, color jitter, etc.) | Industry-standard augmentation library, shows up constantly in CV job descriptions |
| **Weights & Biases (wandb)** | Experiment tracking — logging loss curves, comparing runs | One of the most commonly requested "have you used an experiment tracker" tools in ML interviews |
| **Grad-CAM** (via `pytorch-grad-cam`) | Visualize *what the model looked at* to make a prediction | Model interpretability — a differentiator, especially for health-adjacent ML where "why did the model say this" matters |
| **Docker** | Containerize the inference API | Almost every ML engineering job wants "can you ship a model," and Docker is the baseline answer |
| **Gradio** | Quick interactive web demo, deployable to Hugging Face Spaces | Fast way to make a shareable, clickable demo without hand-building a frontend |

You already know FastAPI and React — this project reuses FastAPI for the "real"
inference API, and uses Gradio (something new) for the fast demo, so you're not
just re-doing what you've already shown.

## Dataset

**HAM10000** ("Human Against Machine with 10000 training images") — 10,015
dermatoscopic images, originally labeled across 7 skin lesion classes. For this
project you'll regroup those 7 labels into 2:

- **Malignant (cancer):** `mel` (melanoma), `bcc` (basal cell carcinoma),
  `akiec` (actinic keratosis / early carcinoma)
- **Benign:** `nv`, `bkl`, `df`, `vasc`

This grouping is defined in one place — `config.MALIGNANT_DX` — so it's a
one-line change if you want to try the alternative convention some papers use
(treating `akiec` as benign since it's precancerous rather than invasive).

Heads up: this split is also **imbalanced in a new way**. Malignant cases are
only ~20% of the dataset, which matters a lot for a cancer-detection task — a
model that just predicts "benign" every time would already score ~80%
accuracy while being clinically useless. `evaluate.py` is written to make you
report sensitivity/specificity and ROC-AUC precisely because of this — don't
lean on plain accuracy as your headline number.

- Kaggle: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
- Download the images and the `HAM10000_metadata.csv` file, then unzip into `data/raw/`
  (see `data/README.md` for the exact expected layout).
- **Note:** this dataset is for non-commercial / educational use — fine for a
  portfolio project, just don't productize it commercially without checking licensing.

## Suggested build order

Don't try to fill in every file at once. Work top to bottom — each step only needs
the one before it to work:

1. **`data/README.md`** — download the dataset, confirm the folder layout
2. **`src/config.py`** — fill in your paths and hyperparameters
3. **`src/dataset.py`** — load the CSV, build a PyTorch `Dataset` that returns (image, label)
4. **`src/transforms.py`** — build your Albumentations train/val transform pipelines
5. **`src/model.py`** — load a pretrained backbone, swap the final layer for 7 classes
6. **`src/train.py`** — training loop + wandb logging (get this working before anything else)
7. **`src/evaluate.py`** — confusion matrix, per-class precision/recall/F1
8. **`src/explain.py`** — Grad-CAM heatmaps on a few sample images
9. **`api/main.py`** — FastAPI endpoint that loads your trained model and serves predictions
10. **`demo/app.py`** — Gradio UI that calls your model directly (or hits the API)
11. **`Dockerfile`** — containerize the FastAPI service

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

You'll also want a free [Weights & Biases account](https://wandb.ai) (`wandb login`
from your terminal once you've signed up) before running `train.py`.

## A note on scope

This is a cancer-detection task, which raises the stakes on how you frame it.
Don't present this project (in interviews, your portfolio, or the demo UI
itself) as something that could actually diagnose skin cancer. Frame it as
what it is: a portfolio project demonstrating a medical imaging pipeline,
evaluated on a public benchmark dataset, with honest reporting of its
accuracy, sensitivity/specificity, and limitations (class imbalance, small
dataset, no clinical validation, no dermatologist review). Put an explicit
"not a diagnostic tool" disclaimer in the Gradio demo's description — it's
already stubbed in for you in `demo/app.py`, just don't remove it.

That framing is itself a good signal in an interview — it shows you
understand the difference between a research/portfolio result and a
deployable clinical tool, which matters more in health AI than in most other
ML domains.
