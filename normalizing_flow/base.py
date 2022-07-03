from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class Transform(nn.Module, ABC):
    """Base class for transformations with learnable parameters.
    """

    def __init__(self):
        super(Transform, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def transform(self, x: torch.tensor, log_det_jac: torch.tensor):
        """Computes f(x) and log_abs_det_jac(x)."""
        ...

    @abstractmethod
    def invert(self, y: torch.tensor, inv_log_det_jac: torch.tensor):
        """Computes f^-1(y) and inv_log_abs_det_jac(y)."""
        ...
