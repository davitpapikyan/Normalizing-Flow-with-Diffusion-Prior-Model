import os

import numpy as np
import torch
from torch import Tensor

from .base import Prior
from .utils import ZeroConv2d


class IsotropicGaussian(Prior):
    """Multivariate isotropic Gaussian prior.

    Attributes:
        Log2PI: Constant used to compute log probability.
        mean: Mean.
        logsd: Log covariance.
    """

    def __init__(self, mean: Tensor, logsd: Tensor):
        """Initializes the parameters."""
        super(IsotropicGaussian, self).__init__()
        self.Log2PI = float(np.log(2 * np.pi))
        self.mean = mean
        self.logsd = logsd

    def compute_log_prob(self, x: Tensor) -> Tensor:
        """Computes the log density of the given tensor of shape [B, D] where B is the batch size.

        Args:
             x: Input tensor.

        Returns:
            Log density value.
        """
        logps = -0.5 * (self.Log2PI + 2.0 * self.logsd + ((x - self.mean) ** 2.0) * torch.exp(-2.0 * self.logsd))
        return logps.view(x.size(0), -1).sum(dim=1)

    def sample(self, shape: tuple = None, temperature: float = 1.0) -> Tensor:
        """Samples from the prior distribution. Note that sample dimensions are defined based on mean parameter.

        Args:
            shape: Unused parameter. Shape is uniquely defined by mean.
            temperature: The temperature parameter.

        Returns:
            Sampled tensor of given shape.
        """
        eps = torch.zeros_like(self.mean, device=self.device).normal_()
        return self.mean + (torch.exp(self.logsd) * temperature) * eps


class GaussianPrior(Prior):
    """Multivariate isotropic Gaussian prior with leanable mean and covariance.
    """

    def __init__(self, in_channels: int, learn_prior_mean_logs: bool = True):
        """Initializes the parameters.

        Args:
            in_channels: The number of channels of an input image.
            learn_prior_mean_logs: Whether to learn mean and covariance of Gaussian prior.
        """
        super(GaussianPrior, self).__init__()
        self.__conv = ZeroConv2d(2 * in_channels, 2 * in_channels, padding=(3 - 1) // 2) \
            if learn_prior_mean_logs else None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__conv.to(self.device)

    def compute_log_prob(self, x: Tensor) -> Tensor:
        """Computes the log density of the given tensor of shape [B, D] where B is the batch size.

        Args:
             x: Input tensor.

        Returns:
            Log density value.
        """
        h = torch.zeros(size=(x.shape[0], 2*x.shape[1], *x.shape[2:]), device=self.device)
        h = self.__conv(h) if self.__conv else h
        mean, logs = torch.chunk(h, 2, dim=1)
        prior = IsotropicGaussian(mean, logs)
        return prior.compute_log_prob(x)

    def sample(self, shape: tuple = None, temperature: float = 1.0) -> Tensor:
        """Samples from the prior distribution. Note that sample dimensions are defined based on mean parameter.

        Args:
            shape: The size of tensor to be sampled.
            temperature: The temperature parameter.

        Returns:
            Sampled tensor of given shape.
        """
        h = torch.zeros(size=(shape[0], 2*shape[1], *shape[2:]), device=self.device)
        h = self.__conv(h) if self.__conv else h
        mean, logs = torch.chunk(h, 2, dim=1)
        prior = IsotropicGaussian(mean, logs)
        return prior.sample(temperature=temperature)


def save_model(logger_obj, model, p_dist, optim, current_epoch, current_iteration, checkpoint_directory, text=""):
    """Helper function to save model.
    """
    logger_obj.info(f"Saving the model. {text}")
    if isinstance(p_dist, GaussianPrior):
        keys, prefix = ("flow", "prior_dist"), "gaussian"
    else:
        keys, prefix = ("nf_backbone", "diffusion_prior"), "diffusion"
    torch.save({
        keys[0]: model.state_dict(),
        keys[1]: p_dist.state_dict(),
        "optimizer": optim.state_dict(),
        "current_iter": current_iteration,
    }, os.path.join(checkpoint_directory, f"model_{prefix}_{str(current_epoch).zfill(3)}.pt"))
