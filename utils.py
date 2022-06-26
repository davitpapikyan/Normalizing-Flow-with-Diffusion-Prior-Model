import logging
import os
import random

import numpy as np
import torch


def setup_logger(name: str = __name__):
    """Creates a logger with the specified name.

    Args:
        name: The name of the logger.

    Returns:
        Logger.
    """
    return logging.getLogger(name)


def set_seeds(seed: int = 0):
    """Sets random seeds of Python, NumPy and PyTorch.
    Args:
        seed: Seed value.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
