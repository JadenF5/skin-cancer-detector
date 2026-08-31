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
import torch.nn as nn
import wandb

from . import config
from .dataset import HAM10000Dataset, build_splits
from .transforms import get_train_transforms, get_val_transforms
from .model import build_model, save_checkpoint


def train_one_epoch(model, dataloader, optimizer, criterion, device) -> float:
    """
    - model.train()
    - loop over dataloader batches: zero grads, forward pass, compute loss,
      loss.backward(), optimizer.step()
    - track and return the average loss over the epoch (useful to log to wandb)
    """
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
      optimizer.zero_grad()
      images = images.to(device)
      labels = labels.to(device)
      outputs = model(images)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      running_loss += loss.item()
    return running_loss / len(dataloader)

def validate(model, dataloader, criterion, device) -> tuple[float, float, float]:
    """
    - model.eval(), with torch.no_grad():
    - loop over dataloader, compute loss and accuracy
    - return (avg_val_loss, val_accuracy)
    """
    model.eval()
    avg_val_loss = 0.0
    val_accuracy = 0.0
    val_sensitivity = 0.0
    total_samples = 0.0
    running_loss = 0.0
    total_correct = 0.0
    true_positives = 0
    actual_positives = 0
    with torch.no_grad():
      for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        running_loss += criterion(outputs, labels).item()
        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
        true_positives += ((preds == 1) & (labels == 1)).sum().item()
        actual_positives += (labels == 1).sum().item()
    val_accuracy = total_correct / total_samples
    avg_val_loss = running_loss / len(dataloader)
    val_sensitivity = true_positives / actual_positives if actual_positives > 0 else 0.0
    return avg_val_loss, val_accuracy, val_sensitivity


def main():
    """
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
    wandb.init(entity=config.WANDB_ENTITY, project=config.WANDB_PROJECT,
      config={
        "architecture": config.BACKBONE,
        "learning_rate": config.LEARNING_RATE,
        "epochs": config.NUM_EPOCHS,
        "batch_size": config.BATCH_SIZE,
        "seed": config.RANDOM_SEED,
      }
    )
    if torch.cuda.is_available():
      device = "cuda"
    else:
      device = "cpu"
    train_df, val_df, test_df = build_splits()
    train_dataset = HAM10000Dataset(
       metadata_df=train_df,
       image_dirs=config.IMAGE_DIRS,
       transform=get_train_transforms()
    )
    val_dataset = HAM10000Dataset(
       metadata_df=val_df,
       image_dirs=config.IMAGE_DIRS,
       transform=get_val_transforms()
    )
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    n_benign = (train_df['new_label_column'] == 0).sum()
    n_malignant = (train_df['new_label_column'] == 1).sum()
    weight_multipler = n_benign / n_malignant
    class_weights = torch.tensor([1.0, weight_multipler], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    best_composite_score = 0.0
    config.CHECKPOINT_DIR.mkdir(exist_ok=True)
    checkpoint_path = config.CHECKPOINT_DIR / "best_model.pth"
    for epoch in range(config.NUM_EPOCHS):
      avg_train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
      avg_val_loss, val_accuracy, val_sensitivity = validate(model, val_loader, criterion, device)
      composite_score = (val_accuracy + val_sensitivity) / 2.0
      wandb.log({
        "epoch": epoch + 1,
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
        "val_acc": val_accuracy,
        "val_sensitivity": val_sensitivity,
        "val_composite_score": composite_score
      })
      print(f"Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.4f} | Val Sens: {val_sensitivity:.4f}")
      if composite_score > best_composite_score:
        best_composite_score = composite_score
        print(f"New Best Composite Score: {best_composite_score:.4f}! Saving Checkpoint.")
        save_checkpoint(model, checkpoint_path)
    wandb.finish()


if __name__ == "__main__":
    main()
