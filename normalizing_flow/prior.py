import torch
from torch.distributions import Normal

from .base import Prior


class GaussianPrior(Prior):
    """Prior distribution of Glow model.
    """

    def __init__(self, scale: float = 1.0):
        """Initializes the prior with standard normal distribution.

        Args:
            scale: Standard deviation of Normal distribution.
        """
        super(GaussianPrior, self).__init__()
        self.__prior = Normal(loc=0.0, scale=scale)

    def sample(self, shape: torch.Size) -> torch.tensor:
        """Samples from the prior distribution.

        Args:
            shape: The size of tensor to be sampled.

        Returns:
            Sampled tensor of given shape.
        """
        # TODO: Check if the returned tensor is on GPU, if not, add loc and scale to device.
        return self.__prior.sample(sample_shape=shape)

    def compute_log_prob(self, x: torch.tensor):
        """Computes the log density of the given tensor of shape [B, D] where B is the batch size.

        Args:
             x: Input tensor.

        Returns:
            Log density value.
        """
        return self.__prior.log_prob(x).view(x.size(0), -1).sum(dim=1)
