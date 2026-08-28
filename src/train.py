"""
Training loop, with Weights & Biases logging.

New tool: wandb (Weights & Biases) tracks your experiments — loss/accuracy
curves, hyperparameters, even sample predictions — so you can compare runs
without manually tracking numbers in a spreadsheet. Run `wandb login` once
(free account at https://wandb.ai) before running this file.

Run with: python -m src.train
"""

import torch
from torch.utils.data import DataLoader
import wandb

from . import config
from .dataset import HAM10000Dataset, build_splits
from .transforms import get_train_transforms, get_val_transforms
from .model import build_model, save_checkpoint


def train_one_epoch(model, dataloader, optimizer, criterion, device) -> float:
    """
    TODO:
    - model.train()
    - loop over dataloader batches: zero grads, forward pass, compute loss,
      loss.backward(), optimizer.step()
    - track and return the average loss over the epoch (useful to log to wandb)
    """
    raise NotImplementedError


def validate(model, dataloader, criterion, device) -> tuple[float, float]:
    """
    TODO:
    - model.eval(), with torch.no_grad():
    - loop over dataloader, compute loss and accuracy
    - return (avg_val_loss, val_accuracy)
    """
    raise NotImplementedError


def main():
    """
    TODO, roughly in this order:
    1. wandb.init(project=config.WANDB_PROJECT, entity=config.WANDB_ENTITY,
                   config={...your hyperparameters...})
    2. Set device = "cuda" if torch.cuda.is_available() else "cpu"
    3. Build train/val/test splits with build_splits()
    4. Build HAM10000Dataset instances for train and val, with the
       appropriate transforms from transforms.py
    5. Wrap each in a DataLoader (batch_size=config.BATCH_SIZE, shuffle=True
       for train, shuffle=False for val)
    6. Build the model with build_model(), move it to device
    7. Set up an optimizer (torch.optim.Adam is a reasonable default) and a
       loss function. Use nn.CrossEntropyLoss(weight=...) here — with the
       binary regrouping, malignant cases are only ~20% of the data, so an
       unweighted loss will happily learn to just predict "benign" most of
       the time. Compute weight as, roughly, inverse class frequency in your
       train split (e.g. torch.tensor([1.0, n_benign/n_malignant])), and
       pass it to CrossEntropyLoss.
    8. Loop over config.NUM_EPOCHS:
         - call train_one_epoch(), then validate()
         - wandb.log({"train_loss": ..., "val_loss": ..., "val_acc": ...,
           "val_sensitivity": ...}) — track sensitivity (recall on the
           malignant class) alongside accuracy, since it's the metric that
           actually matters for a cancer-detection task: missing a
           malignant case (false negative) is far worse than a false alarm
         - print progress
         - keep track of the best val accuracy seen so far, and
           save_checkpoint() whenever it improves
    9. wandb.finish()
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
