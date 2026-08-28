"""
Evaluation on the held-out test set: confusion matrix + per-class metrics.

Run with: python -m src.evaluate
"""

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

from . import config
from .dataset import HAM10000Dataset, build_splits
from .transforms import get_val_transforms
from .model import build_model, load_checkpoint


def get_predictions(model, dataloader, device) -> tuple[list[int], list[int], list[float]]:
    """
    TODO:
    - model.eval(), with torch.no_grad():
    - loop over dataloader, run the model to get logits
    - softmax the logits and take:
        - argmax -> predicted class index (0/1)
        - the probability of class 1 (malignant) specifically -> you'll
          need this as a continuous score for the ROC curve below, not
          just the final 0/1 decision
    - collect and return (all_true_labels, all_predicted_labels,
      all_malignant_probabilities) as three lists
    """
    raise NotImplementedError


def print_classification_report(y_true, y_pred) -> None:
    """
    TODO: use sklearn.metrics.classification_report(y_true, y_pred,
    target_names=config.CLASSES) and print it. Read precision/recall for
    the "malignant" row specifically — recall on that row is your
    sensitivity (what fraction of actual cancer cases you caught), which
    matters more here than overall accuracy.
    """
    raise NotImplementedError


def print_sensitivity_specificity(y_true, y_pred) -> None:
    """
    TODO:
    - Get the confusion matrix: tn, fp, fn, tp = confusion_matrix(y_true,
      y_pred).ravel() (this only works cleanly for binary — which is exactly
      what you have now).
    - sensitivity (recall on malignant) = tp / (tp + fn)
    - specificity (recall on benign) = tn / (tn + fp)
    - Print both, clearly labeled. In a cancer-screening context, a low
      sensitivity means you're missing real cancer cases — call that out
      explicitly if it happens, don't just report the number.
    """
    raise NotImplementedError


def plot_roc_curve(y_true, y_scores, save_path=None) -> None:
    """
    TODO:
    - fpr, tpr, thresholds = roc_curve(y_true, y_scores)  (y_scores = your
      malignant-class probabilities from get_predictions, NOT the 0/1 preds)
    - auc = roc_auc_score(y_true, y_scores)
    - plot fpr vs tpr with matplotlib, label the AUC value in the title
    - if save_path is given, plt.savefig(save_path); otherwise plt.show()

    ROC-AUC is a standard headline metric for binary medical classifiers
    because it summarizes performance across every possible decision
    threshold, not just the default 0.5 cutoff — worth knowing why, in case
    it comes up in an interview.
    """
    raise NotImplementedError


def plot_confusion_matrix(y_true, y_pred, save_path=None) -> None:
    """
    TODO:
    - sklearn.metrics.confusion_matrix(y_true, y_pred)
    - visualize with seaborn.heatmap, labeling axes with config.CLASSES
    - if save_path is given, plt.savefig(save_path); otherwise plt.show()
    """
    raise NotImplementedError


def main():
    """
    TODO:
    1. Load your best checkpoint into a fresh model via build_model() +
       load_checkpoint()
    2. Build the test split + DataLoader (reuse build_splits(), but this
       time use the test_df and get_val_transforms() — no augmentation
       at test time)
    3. Call get_predictions(), then:
       - print_classification_report()
       - print_sensitivity_specificity()
       - plot_confusion_matrix()
       - plot_roc_curve()
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
