from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class Transform(nn.Module, ABC):
    """Base class for transformations with learnable parameters.
    """

    def __init__(self):
        super(Transform, self).__init__()

    @abstractmethod
    def transform(self, x: torch.tensor):
        """Computes f(x) and log_abs_det_jac(x)."""
        ...

    @abstractmethod
    def invert(self, y: torch.tensor):
        """Computes f^-1(y) and inv_log_abs_det_jac(y)."""
        ...
