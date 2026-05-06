from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducible NumPy/PyTorch experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(prefer_cuda: bool = True) -> str:
    """Return ``cuda`` when available and requested, otherwise ``cpu``."""
    return "cuda" if prefer_cuda and torch.cuda.is_available() else "cpu"
