"""
Model definition: transfer learning on a torchvision backbone.

Concept: instead of training a CNN from scratch (which needs way more data
than 10k images), you start from a model already trained on ImageNet
(millions of images) and just retrain the final classification layer — and
optionally fine-tune some of the earlier layers too. This is the standard
approach for small/medium medical imaging datasets.
"""

import torch
import torch.nn as nn
import torchvision.models as models

from . import config


def build_model(num_classes: int = None, pretrained: bool = True) -> nn.Module:
    """
    TODO:
    1. Load a torchvision backbone matching config.BACKBONE, e.g.:
         torchvision.models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
       or for EfficientNet:
         torchvision.models.efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)
    2. Replace the final classification layer so it outputs `num_classes`
       logits instead of the original 1000 ImageNet classes.
       - For ResNet: replace `model.fc` (a single nn.Linear).
       - For EfficientNet: replace `model.classifier[-1]`.
    3. Return the model.

    Args:
        num_classes: defaults to len(config.CLASSES) if not passed — that's
            2 here (benign/malignant).
        pretrained: whether to load ImageNet weights (almost always True
            for a dataset this size).

    Note on binary setups: with 2 classes you could instead use a single
    output logit + nn.BCEWithLogitsLoss instead of 2 logits + CrossEntropyLoss.
    Both work; 2-logit CrossEntropy is what the rest of this scaffold assumes
    (it keeps train.py/evaluate.py identical to a general multiclass setup),
    but swapping to single-logit BCE is a reasonable thing to try and mention
    in an interview if you want to show you understand the tradeoff.
    """
    if num_classes is None:
        num_classes = len(config.CLASSES)
    if config.BACKBONE == "efficientnet_b0":
        model = models.efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif config.BACKBONE == "resnet18":
        model = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported backbone: {config.BACKBONE}")
    return model


def freeze_backbone(model: nn.Module) -> nn.Module:
    """
    Optional but worth trying: freeze every layer except the new final
    classification layer, so only that layer trains. Faster, and can work
    well when your dataset is small relative to the backbone.

    TODO: set `.requires_grad = False` on every parameter except the ones
    belonging to the layer you replaced in build_model(). Compare results
    against training the whole network (this function unused / not called)
    to see which generalizes better on your val set.
    """
    for params in model.parameters():
        params.requires_grad = False
    for params in model.classifier.parameters():
        params.requires_grad = True
    return model


def save_checkpoint(model: nn.Module, path) -> None:
    """
    TODO: torch.save(model.state_dict(), path)
    """
    torch.save(model.state_dict(), path)


def load_checkpoint(model: nn.Module, path, device: str = "cpu") -> nn.Module:
    """
    TODO: model.load_state_dict(torch.load(path, map_location=device));
    return model
    """
    model.load_state_dict(torch.load(path, map_location=device))
    return model
