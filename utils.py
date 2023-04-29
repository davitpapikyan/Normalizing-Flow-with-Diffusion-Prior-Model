import logging
import os
import random
import sys
from typing import List, Dict

import numpy as np
import pkg_resources
import torch


def setup_logger(name: str = __name__):
    """Creates a logger with the specified name.

    Args:
        name: The name of the logger.

    Returns:
        Logger.
    """
    return logging.getLogger(name)


def log_environment(logger):
    """Logs environment information for reproducability.

    Args:
        logger: The logger.
    """
    # Logging Python version.
    logger.info(f"Python version: {sys.version}")

    # Python packages.
    logger.info("Installed packages and their versions.")
    installed_packages = pkg_resources.working_set
    installed_packages_list = sorted(["%s==%s" % (pack.key, pack.version) for pack in installed_packages])
    logger.info(f"{installed_packages_list}")

    # Logging OS environment variables.
    logger.info("OS environment variables")
    for env_var in os.environ:
        logger.info(f"{env_var}: {os.environ[env_var]}")


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
    try:
        torch.use_deterministic_algorithms(True)
    except:
        torch.set_deterministic(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def set_mode(*models, mode):
    for model in models:
        if mode == "train":
            model.train()
        elif mode == "eval":
            model.eval()
        else:
            raise ValueError(f"Unknown value for mode={mode}, supported values are ['train', 'eval'].")


def parse_metric(metric_info: dict, metric_type="fid") -> List[Dict[str, str]]:
    """Returns a list of kwargs to be used for model evaluation."""
    if metric_type in ("fid", "kid"):
        assert len(metric_info["mode"]) == len(metric_info["model_name"]), \
            "Make sure that mode and model_name lists have equal lengths."
        kwargs = [] if len(metric_info["mode"]) == 0 else \
            [{"mode": metric_info["mode"][i], "model_name": metric_info["model_name"][i]}
             for i in range(len(metric_info["mode"]))]
        return kwargs
