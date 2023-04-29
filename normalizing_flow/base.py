from abc import ABC, abstractmethod
from typing import Tuple, Any

import torch
import torch.nn as nn
from torch import Tensor


class Transform(nn.Module, ABC):
    """Base class for invertible transformation f: x -> y.

    Attributes:
        device: Device.
    """

    def __init__(self):
        super(Transform, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def transform(self, x: Tensor, log_det_jac: Tensor, logp: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Computes f(x), log_abs_det_jac(x) and logp if applicable.

        Args:
            x: Input tensor.
            log_det_jac: Ongoing log abs determinant of jacobian.
            logp: Ongoing latent variable log probability calculated using prior distribution.

        Returns:
            y: Forward transformed input.
            log_det_jac: Log abs determinant of jacobian matrix of the transformation.
            logp: Latent variable log probability calculated using prior distribution.
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


class Prior(nn.Module, ABC):
    """Base class for the prior distribution.

    Attributes:
        device: Device.
    """

    def __init__(self):
        super(Prior, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def sample(self, shape: Tuple[int, Any]) -> Tensor:
        """Samples from the prior distribution.

        Args:
            shape: The size of tensor to be sampled. The first entry identifies the number of samples. The following
            entries are dimensions.

        Returns:
            Sampled tensor of the given shape.
        """
        ...

    @abstractmethod
    def compute_log_prob(self, x: Tensor) -> Tensor:
        """Computes the log density of the given tensor of shape [B, *] where B is the batch size.

        Args:
             x: Input tensor.

        Returns:
            Log density value for each sample, a vector of shape [B, ].
        """
        ...
