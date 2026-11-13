from __future__ import annotations

import os
import random
from typing import List

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def repeat_seeds(seed_start: int = 42, n_repeats: int = 5) -> List[int]:
    """Protocol seeds: seed_start, seed_start+1, ..., seed_start+n_repeats-1."""
    if n_repeats < 1:
        raise ValueError("n_repeats must be >= 1")
    return list(range(int(seed_start), int(seed_start) + int(n_repeats)))
