from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import softplus

from .base import Transform
from .transforms import IdentityTransform, ActNorm, InvConv2d, AffineCoupling, Squeeze, Split
from .prior import GaussianPrior
from .utils import calc_chunk_sizes


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
        coupling_net_in_channels, _ = calc_chunk_sizes(in_channels)
        self.affcoupling = AffineCoupling(in_channels=coupling_net_in_channels, n_features=coupling_net_n_features)

    def transform(self, x, log_det_jac: torch.tensor):
        """Computes forward transformation and log abs determinant of jacobian matrix.

        Args:
            x: Input tensor of shape [B, C, H, W].
            log_det_jac: Ongoing log abs determinant of jacobian.

        Returns:
            y: Forward transformed input.
            log_det_jac: log abs determinant of jacobian matrix of the transformation.
        """
        y, log_det_jac = self.actnorm.transform(x, log_det_jac)
        y, log_det_jac = self.invconv2d.transform(y, log_det_jac)
        y, log_det_jac = self.affcoupling.transform(y, log_det_jac)
        return y, log_det_jac

    def invert(self, y, inv_log_det_jac: torch.tensor):
        """Computes inverse transformation and log abs determinant of jacobian matrix.

        Args:
            y: Input tensor of shape [B, C, H, W].
            inv_log_det_jac: Ongoing log abs determinant of jacobian.

        Returns:
            inv_y: Inverse transformed input.
            inv_log_det_jac: log abs determinant of jacobian matrix of the inverse transformation.
        """
        inv_y, inv_log_det_jac = self.affcoupling.invert(y, inv_log_det_jac)
        inv_y, inv_log_det_jac = self.invconv2d.invert(inv_y, inv_log_det_jac)
        inv_y, inv_log_det_jac = self.actnorm.invert(inv_y, inv_log_det_jac)
        return inv_y, inv_log_det_jac


class GlowBlock(Transform):
    """A block of consisting of Squeeze, StepFlow and Split operations.

    Attributes:
        squeeze: Squeeze operation.
        flows: A list of step-flows.
        split: Split operation.
    """

    def __init__(self, prior, in_channels: int = 3, K: int = 32):
        """Initializes operations of the block.

        Args:
            prior: Initialized prior distribution.
            in_channels: The number of input channels to the block.
            K: The number of flows in the block.
        """
        super(GlowBlock, self).__init__()
        flow_channels = 4 * in_channels
        self.squeeze = Squeeze()
        self.flows = nn.ModuleList([StepFlow(in_channels=flow_channels) for _ in range(K)])
        self.split = Split(prior=prior)

    def transform(self, x: torch.tensor, log_det_jac: torch.tensor):
        """Computes forward transformation and log abs determinant of jacobian matrix.

        Args:
            x: Input tensor of shape [B, C, H, W].
            log_det_jac: Ongoing log abs determinant of jacobian matrix.

        Returns:
            y: Forward transformed input.
            log_det_jac: log abs determinant of jacobian matrix of the transformation.
        """
        # Squeeze.
        y, log_det_jac = self.squeeze.transform(x, log_det_jac)

        # Step of flow.
        for flow in self.flows:
            y, log_det_jac = flow.transform(y, log_det_jac)

        # Split.
        y, log_det_jac = self.split.transform(y, log_det_jac)

        return y, log_det_jac

    def invert(self, y: torch.tensor, inv_log_det_jac: torch.tensor):
        """Computes inverse transformation and log abs determinant of jacobian matrix.

        Args:
            y: Input tensor of shape [B, C, H, W].
            inv_log_det_jac: Ongoing log abs determinant of jacobian.

        Returns:
            inv_y: Inverse transformed input.
            inv_log_det_jac: log abs determinant of jacobian matrix of the inverse transformation.
        """
        # Split.
        inv_y, inv_log_det_jac = self.split.invert(y, inv_log_det_jac)

        # Step of flow.
        for flow in reversed(self.flows):
            inv_y, inv_log_det_jac = flow.invert(inv_y, inv_log_det_jac)

        # Squeeze.
        inv_y, inv_log_det_jac = self.squeeze.invert(inv_y, inv_log_det_jac)

        return inv_y, inv_log_det_jac


class Glow(Transform):
    """The Glow model.

    Attributes:
        blocks: A list of Glow blocks.
        final_squeeze: Squeeze operation applied in the end of the architecture.
        final_flows: A list of step-flows applied in the end of the architecture.
    """

    def __init__(self,
                 in_channel: int = 3,
                 L: int = 3,
                 K: int = 32,
                 Prior=GaussianPrior,
                 temperature: float = 1.0,
                 apply_dequantization: bool = False):
        """Initializes operations of the model.

        Args:
            in_channel: The number of input channels.
            L: The number of Glow blocks.
            K: The number of flows in the Glow block.
            Prior: The prior distribution.
            temperature: Standard deviation of prior distribution often referred to as temperature of sampling.
            apply_dequantization: Boolean value indicating whether to apply Dequantization on data or not.
        """
        super(Glow, self).__init__()
        self.L = L
        self.K = K
        self.in_channel = in_channel

        self.__prior = Prior(scale=temperature)
        self.dequant = Dequantization() if apply_dequantization else IdentityTransform()   # VariationalDequantization()
        self.blocks = nn.ModuleList(GlowBlock(prior=self.__prior, in_channels=(2**i * self.in_channel),
                                              K=self.K)
                                    for i in range(self.L-1))
        self.final_squeeze = Squeeze()
        self.final_flows = nn.ModuleList(StepFlow(in_channels=(2**(self.L+1) * self.in_channel))
                                         for _ in range(self.K))

    def transform(self, x: torch.tensor, log_det_jac: torch.tensor):
        """Computes forward transformation and log abs determinant of jacobian matrix.

        Args:
            x: Input tensor of shape [B, C, H, W].
            log_det_jac: Ongoing log abs determinant of jacobian matrix.

        Returns:
            y: Forward transformed input.
            log_det_jac: log abs determinant of jacobian matrix of the transformation.
        """
        # Dequantization.
        y, log_det_jac = self.dequant.transform(x, log_det_jac)

        # Glow block.
        for block in self.blocks:
            y, log_det_jac = block.transform(y, log_det_jac)

        # Squeeze.
        y, log_det_jac = self.final_squeeze.transform(y, log_det_jac)

        # Step of flow.
        for flow in self.final_flows:
            y, log_det_jac = flow.transform(y, log_det_jac)

        log_det_jac += self.__prior.compute_log_prob(y)

        return y, log_det_jac

    def invert(self, y: torch.tensor, inv_log_det_jac: torch.tensor):
        """Computes inverse transformation and

        Args:
            y: Input tensor of shape [B, C, H, W].
            inv_log_det_jac: Ongoing log abs determinant of jacobian.

        Returns:
            inv_y: Inverse transformed input.
            inv_log_det_jac: log abs determinant of jacobian matrix of the inverse transformation.
        """
        inv_log_det_jac += self.__prior.compute_log_prob(y)

        # Step of flow.
        inv_y = y
        for flow in reversed(self.final_flows):
            inv_y, inv_log_det_jac = flow.invert(inv_y, inv_log_det_jac)

        # Squeeze.
        inv_y, inv_log_det_jac = self.final_squeeze.invert(inv_y, inv_log_det_jac)

        # Glow block.
        for block in reversed(self.blocks):
            inv_y, inv_log_det_jac = block.invert(inv_y, inv_log_det_jac)

        # Dequantization.
        inv_y, inv_log_det_jac = self.dequant.invert(inv_y, inv_log_det_jac)

        return inv_y, inv_log_det_jac

    def get_output_shape(self, height: int, width: int):
        """Calculates the output shape.

        Args:
            height: The input's height.
            width: The input's width.

        Returns:
            A tuple of 3 values representing the number of channels, the height and width of output.
        """
        const = 2 ** self.L
        n_channels = int(self.in_channel * const * 2)
        height, width = int(height / const), int(width / const)
        return n_channels, height, width

    @torch.no_grad()
    def sample(self, n_samples: int, img_shape: Tuple[int, int], postprocess_func=None):
        """Samples from Glow model.

        Args:
            n_samples: The number of samples to generate.
            img_shape: The image shape - a tuple containing height and width of input.
            postprocess_func: Postprocessor function to generate final output.

        Returns:
            A tuple of 2 values, the first is a torch.tensor representing sampled data points, the second
            torch.tensor stores the log likelihoods of those generated samples.
        """
        self.eval()
        in_height, in_width = img_shape
        n_channels, out_height, out_width = self.get_output_shape(in_height, in_width)
        z = self.__prior.sample((n_samples, n_channels, out_height, out_width)).to(self.device)
        log_likelihood = torch.zeros(n_samples, device=self.device)
        new_samples, log_likelihood = self.invert(z, log_likelihood)
        self.train()
        final_output = postprocess_func(new_samples.float()) if postprocess_func else new_samples.float()
        return final_output, log_likelihood


class Dequantization(Transform):
    """Image dequantization layer. Take a look at UvA DL Notebooks tutorial - Tutorial 11: Normalizing Flows for
    image modeling where the below implementation is taken from.


    Attributes:
        alpha: Small constant that is used to scale the original input. Prevents dealing with values very close to 0
            and 1 when inverting the sigmoid.
        quants: Number of possible discrete values (usually 256 for 8-bit image).
    """

    def __init__(self, alpha=1e-5, quants=256):
        """Initializes attributes."""
        super().__init__()
        self.alpha = alpha
        self.quants = quants

    def transform(self, x: torch.tensor, log_det_jac: torch.tensor):
        """Computes forward transformation and log abs determinant of jacobian matrix.

        Args:
            x: Input tensor of shape [B, C, H, W].
            log_det_jac: Ongoing log abs determinant of jacobian matrix.

        Returns:
            y: Forward transformed input.
            log_det_jac: log abs determinant of jacobian matrix of the transformation.
        """
        y, log_det_jac = self.dequant(x, log_det_jac)
        y, log_det_jac = self.sigmoid(y, log_det_jac, reverse=True)
        return y, log_det_jac

    def invert(self, y: torch.tensor, inv_log_det_jac: torch.tensor):
        """Computes inverse transformation and

        Args:
            y: Input tensor of shape [B, C, H, W].
            inv_log_det_jac: Ongoing log abs determinant of jacobian.

        Returns:
            inv_y: Inverse transformed input.
            inv_log_det_jac: log abs determinant of jacobian matrix of the inverse transformation.
        """
        inv_y, inv_log_det_jac = self.sigmoid(y, inv_log_det_jac)
        inv_y *= self.quants
        inv_log_det_jac += np.log(self.quants) * np.prod(inv_y.shape[1:])
        inv_y = torch.floor(inv_y).clamp(min=0, max=self.quants - 1).to(torch.int32)
        return inv_y, inv_log_det_jac

    def sigmoid(self, z: torch.tensor, log_det_jac: torch.tensor, reverse=False):
        """Invertible sigmoid transformation.

        Args:
            z: Input tensor.
            log_det_jac: Ongoing log abs determinant of jacobian.
            reverse: Whetehr to apply inverse transformation or not.

        Returns:
            y: Forward transformed input (or inverse transformed dependent on reverse parameter).
            log_det_jac: log abs determinant of jacobian matrix of the transformation (or inv_log_det_jac dependent
                on reverse parameter).
        """
        if not reverse:
            log_det_jac += (-z-2*softplus(-z)).sum(dim=[1, 2, 3])
            z = torch.sigmoid(z)
        else:
            z = z * (1 - self.alpha) + 0.5 * self.alpha  # Scales to prevent boundaries 0 and 1.
            log_det_jac += np.log(1 - self.alpha) * np.prod(z.shape[1:])
            log_det_jac += (-torch.log(z) - torch.log(1-z)).sum(dim=[1, 2, 3])
            z = torch.log(z) - torch.log(1-z)
        return z, log_det_jac

    def dequant(self, z: torch.tensor, log_det_jac: torch.tensor):
        """Dequantization operation - transforms discrete values to continuous ones.

        Args:
            z: Input tensor.
            log_det_jac: Ongoing log abs determinant of jacobian.

        Returns:
            y: Forward transformed input (or inverse transformed dependent on reverse parameter).
            log_det_jac: log abs determinant of jacobian matrix of the transformation (or inv_log_det_jac dependent
                on reverse parameter).
        """
        z = z.to(torch.float32)
        z += torch.rand_like(z).detach()
        z /= self.quants
        log_det_jac += -np.log(self.quants) * np.prod(z.shape[1:])
        return z, log_det_jac


# class VariationalDequantization(Dequantization):
#
#     def __init__(self, alpha=1e-5):
#         """
#         Inputs:
#             var_flows - A list of flow transformations to use for modeling q(u|x)
#             alpha - Small constant, see Dequantization for details
#         """
#         super().__init__(alpha=alpha)
#         # TODO: work on self.flows.
#         self.flows = nn.ModuleList([GlowBlock(3, 2)])
#
#     def dequant(self, z):
#         z = z.to(torch.float32)
#
#         # Prior of u is a uniform distribution as before
#         # As most flow transformations are defined on [-infinity,+infinity], we apply an inverse sigmoid first.
#
#         deq_noise = torch.rand_like(z).detach()
#         deq_noise, ldj = self.sigmoid(deq_noise, reverse=True)
#
#         print(deq_noise.shape)
#         for flow in self.flows:
#             deq_noise, logdet = flow.transform(deq_noise)
#             ldj += logdet
#         print(deq_noise.shape)
#
#         deq_noise, logdet = self.sigmoid(deq_noise)
#         ldj += logdet
#
#         # After the flows, apply u as in standard dequantization
#         z = (z + deq_noise) / 256.0
#         ldj -= np.log(256.0) * np.prod(z.shape[1:])
#         return z, ldj
