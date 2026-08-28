"""
Small shared helpers. Add to this file as you find yourself repeating code
across train.py / evaluate.py / explain.py — that's the signal something
belongs here instead.

Suggested starting point:
"""

import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    TODO: set random.seed(seed), np.random.seed(seed), torch.manual_seed(seed)
    (and torch.cuda.manual_seed_all(seed) if using a GPU) so your train/val
    splits and weight initialization are reproducible between runs.
    """
    raise NotImplementedError


def get_device() -> str:
    """
    TODO: return "cuda" if torch.cuda.is_available() else "cpu"
    """
    raise NotImplementedError
