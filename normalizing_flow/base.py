from abc import ABC, abstractmethod
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


class Transform(nn.Module, ABC):
    """Base class for normalizing flow transformation f: x -> y.
    """

    def __init__(self):
        super(Transform, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def transform(self, x: Tensor, log_det_jac: Tensor) -> Tuple[Tensor, Tensor]:
        """Computes f(x) and log_abs_det_jac(x).

        Args:
            x: Input tensor.
            log_det_jac: Ongoing log abs determinant of jacobian.

        Returns:
            y: Forward transformed input.
            log_det_jac: log abs determinant of jacobian matrix of the transformation.
        """
        ...

    @abstractmethod
    def invert(self, y: Tensor) -> Tensor:
        """Computes f^{-1}(y).

        Args:
            y: Input tensor.

        Returns:
            inv_y: Inverse transformed input.
        """
        ...


class Prior(ABC):
    """Base class for the prior distribution.
    """

    def __init__(self):
        super(Prior, self).__init__()

    @abstractmethod
    def sample(self, shape: tuple) -> Tensor:
        """Samples from the prior distribution.

        Args:
            shape: The size of tensor to be sampled.

        Returns:
            Sampled tensor of given shape.
        """
        ...

    @abstractmethod
    def compute_log_prob(self, x: Tensor) -> Tensor:
        """Computes the log density of the given tensor of shape [B, D] where B is the batch size.

        Args:
             x: Input tensor.

        Returns:
            Log density value.
        """
        ...
