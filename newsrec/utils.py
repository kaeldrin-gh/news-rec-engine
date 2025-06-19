"""
Utility functions for the project.
"""
import random
import numpy as np
import torch

from newsrec import config

def set_seed(seed: int = config.RANDOM_SEED):
    """
    Sets the random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
