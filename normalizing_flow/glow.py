from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .base import Transform
from .transforms import ActNorm, InvConv2d, AffineCoupling, Squeeze, Split
from .utils import get_item


class StepFlow(Transform):
    """One step of the flow from Glow.

    Attributes:
        actnorm: Activation normalization layer.
        invconv2d: Invertible 1x1 convolution.
        affcoupling: Affine coupling layer.
    """

    def __init__(self, in_channels: int = 3, coupling_net_n_features: int = 512):
        """Initializes ActNorm, InvConv2d and AffineCoupling layers.

        Args:
            in_channels: The number of input channels to the flow.
            coupling_net_n_features: The number of hidden feature maps of the coupling network.
        """
        super(StepFlow, self).__init__()
        self.actnorm = ActNorm(in_channels=in_channels)
        self.invconv2d = InvConv2d(in_channels=in_channels)
        self.affcoupling = AffineCoupling(in_channels=in_channels, n_features=coupling_net_n_features)

    def transform(self, x: Tensor, log_det_jac: Tensor, logp: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Computes forward transformation and log abs determinant of jacobian matrix.

        Args:
            x: Input tensor of shape [B, C, H, W].
            log_det_jac: Ongoing log abs determinant of jacobian.
            logp: Ongoing latent variable log probability calculated using prior distribution.

        Returns:
            y: Forward transformed input.
            log_det_jac: log abs determinant of jacobian matrix of the transformation.
            logp: Latent variable log probability calculated using prior distribution.
        """
        y, log_det_jac, logp = self.actnorm.transform(x, log_det_jac, logp)
        y, log_det_jac, logp = self.invconv2d.transform(y, log_det_jac, logp)
        y, log_det_jac, logp = self.affcoupling.transform(y, log_det_jac, logp)
        return y, log_det_jac, logp

    def invert(self, y: Tensor) -> Tensor:
        """Computes inverse transformation.

        Args:
            y: Input tensor of shape [B, C, H, W].

        Returns:
            inv_y: Inverse transformed input.
        """
        inv_y = self.affcoupling.invert(y)
        inv_y = self.invconv2d.invert(inv_y)
        inv_y = self.actnorm.invert(inv_y)
        return inv_y


class GlowBlock(Transform):
    """A block of consisting of Squeeze, StepFlow and Split operations.

    Attributes:
        squeeze: Squeeze operation.
        flows: A list of step-flows.
        split: Split operation.
    """

    def __init__(self, in_channels: int = 3, K: int = 32, learn_prior_mean_logs: bool = True):
        """Initializes operations of the block.

        Args:
            in_channels: The number of input channels to the block.
            K: The number of flows in the block.
            learn_prior_mean_logs: Whether to learn mean and covariance of Gaussian prior.
        """
        super(GlowBlock, self).__init__()
        flow_channels = 4 * in_channels

        self.squeeze = Squeeze()
        self.flows = nn.ModuleList([StepFlow(in_channels=flow_channels) for _ in range(K)])
        self.split = Split(flow_channels, learn_prior_mean_logs=learn_prior_mean_logs)

    def transform(self, x: Tensor, log_det_jac: Tensor, logp: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Computes forward transformation and log abs determinant of jacobian matrix.

        Args:
            x: Input tensor of shape [B, C, H, W].
            log_det_jac: Ongoing log abs determinant of jacobian matrix.
            logp: Ongoing latent variable log probability calculated using prior distribution.

        Returns:
            y: Forward transformed input.
            log_det_jac: log abs determinant of jacobian matrix of the transformation.
            z: Latent variable.
            logp: Latent variable log probability calculated using prior distribution.
        """
        # Squeeze.
        y, log_det_jac, logp = self.squeeze.transform(x, log_det_jac, logp)

        # Step of flow.
        for flow in self.flows:
            y, log_det_jac, logp = flow.transform(y, log_det_jac, logp)

        # Split.
        y, log_det_jac, z, logp = self.split.transform(y, log_det_jac, logp)

        return y, log_det_jac, z, logp

    def invert(self, y: Tensor, latent: Tensor = None, temperature: float = 1.0) -> Tensor:
        """Computes inverse transformation.

        Args:
            y: Input tensor of shape [B, C, H, W].
            latent: Latent variable.
            temperature: The temperature parameter.

        Returns:
            inv_y: Inverse transformed input.
        """
        # Split.
        inv_y = self.split.invert(y, latent, temperature=temperature)

        # Step of flow.
        for flow in reversed(self.flows):
            inv_y = flow.invert(inv_y)

        # Squeeze.
        inv_y = self.squeeze.invert(inv_y)

        return inv_y


class Glow(Transform):
    """The Glow model.

    Attributes:
        blocks: A list of Glow blocks.
        final_squeeze: Squeeze operation applied in the end of the architecture.
        final_flows: A list of step-flows applied in the end of the architecture.
    """

    def __init__(self, in_channel: int = 3, L: int = 3, K: int = 32, learn_prior_mean_logs: bool = True):
        """Initializes operations of the model.

        Args:
            in_channel: The number of input channels.
            L: The number of Glow blocks.
            K: The number of flows in the Glow block.
            learn_prior_mean_logs: Whether to learn mean and covariance of Gaussian prior.
        """
        super(Glow, self).__init__()
        self.L = L
        self.K = K
        self.in_channel = in_channel

        self.blocks = nn.ModuleList(GlowBlock(in_channels=(2**i * self.in_channel), K=self.K,
                                              learn_prior_mean_logs=learn_prior_mean_logs)
                                    for i in range(self.L-1))

        self.final_squeeze = Squeeze()
        self.final_flows = nn.ModuleList(
            StepFlow(in_channels=(2**(self.L+1) * self.in_channel)) for _ in range(self.K)
        )

    def transform(self, x: Tensor, log_det_jac: Tensor, logp: Tensor) -> Tuple[list, Tensor, Tensor]:
        """Computes forward transformation and log abs determinant of jacobian matrix.

        Args:
            x: Input tensor of shape [B, C, H, W].
            log_det_jac: Ongoing log abs determinant of jacobian matrix.
            logp: Ongoing latent variable log probability calculated using prior distribution.

        Returns:
            parts_of_latent_variable: A list of parts of latent corresponding to each GlowBlock and final step.
            log_det_jac: log abs determinant of jacobian matrix of the transformation.
            logp: Latent variable log probability calculated using prior distribution.
        """
        parts_of_latent_variable = []
        y = x

        # Glow block.
        for block in self.blocks:
            y, log_det_jac, z, logp = block.transform(y, log_det_jac, logp)
            parts_of_latent_variable.append(z)

        # Squeeze.
        y, log_det_jac, logp = self.final_squeeze.transform(y, log_det_jac, logp)

        # Step of flow.
        for flow in self.final_flows:
            y, log_det_jac, logp = flow.transform(y, log_det_jac, logp)

        parts_of_latent_variable.append(y)
        return parts_of_latent_variable, log_det_jac, logp

    def invert(self, latents: list, temperature: float = 1.0) -> Tensor:
        """Computes inverse transformation.

        Args:
            latents: A list of Latent variables. Can also have a length of 1 indicating that it contains only the final
                part of latent variable.
            temperature: The temperature parameter.

        Returns:
            inv_y: Inverse transformed input.
        """
        # Step of flow.
        inv_y = latents[-1]  # Getting the final part.
        for flow in reversed(self.final_flows):
            inv_y = flow.invert(inv_y)

        # Squeeze.
        inv_y = self.final_squeeze.invert(inv_y)

        # Glow block.
        for i, block in enumerate(reversed(self.blocks)):
            latent = get_item(latents, -(i+2))  # Returns None if latents contains only the last part.
            # If None is passed to GlowBlock as latent, then it samples from Gaussian prior.
            inv_y = block.invert(inv_y, latent=latent, temperature=temperature)

        return inv_y

    @torch.no_grad()
    def sample(self, latents: list, postprocess_func=None, temperature: float = 1.0) -> Tensor:
        """Samples from Glow model.

        Args:
            latents: A list of parts of latent variable.
            postprocess_func: Postprocessor function to generate final output.
            temperature: The temperature parameter.

        Returns:
            A Tensor representing sampled data points.
        """
        self.eval()
        new_samples = self.invert(latents, temperature)
        final_output = postprocess_func(new_samples.float()) if postprocess_func else new_samples.float()
        self.train()
        return final_output
