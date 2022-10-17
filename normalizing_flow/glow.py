from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import softplus

from .base import Transform
from .transforms import IdentityTransform, ActNorm, InvConv2d, AffineCoupling, Squeeze, Split
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

    def transform(self, x: Tensor, log_det_jac: Tensor) -> Tuple[Tensor, Tensor]:
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

    def __init__(self, in_channels: int = 3, K: int = 32):
        """Initializes operations of the block.

        Args:
            in_channels: The number of input channels to the block.
            K: The number of flows in the block.
        """
        super(GlowBlock, self).__init__()
        flow_channels = 4 * in_channels

        self.squeeze = Squeeze()
        self.flows = nn.ModuleList([StepFlow(in_channels=flow_channels) for _ in range(K)])
        self.split = Split()

    def transform(self, x: Tensor, log_det_jac: Tensor) -> Tuple[Tensor, Tensor]:
        """Computes forward transformation and log abs determinant of jacobian matrix.

        Args:
            x: Input tensor of shape [B, C, H, W].
            log_det_jac: Ongoing log abs determinant of jacobian matrix.

        Returns:
            y: Forward transformed input.
            log_det_jac: log abs determinant of jacobian matrix of the transformation.
            z: Latent variable.
        """
        # Squeeze.
        y, log_det_jac = self.squeeze.transform(x, log_det_jac)

        # Step of flow.
        for flow in self.flows:
            y, log_det_jac = flow.transform(y, log_det_jac)

        # Split.
        y, log_det_jac, z = self.split.transform(y, log_det_jac)

        return y, log_det_jac, z

    def invert(self, y: Tensor, z: Tensor = None) -> Tensor:
        """Computes inverse transformation.

        Args:
            y: Input tensor of shape [B, C, H, W].
            z: Latent variable.

        Returns:
            inv_y: Inverse transformed input.
        """
        # Split.
        inv_y = self.split.invert(y, z)

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

    def __init__(self,
                 in_channel: int = 3,
                 L: int = 3,
                 K: int = 32,
                 apply_dequantization: bool = False):
        """Initializes operations of the model.

        Args:
            in_channel: The number of input channels.
            L: The number of Glow blocks.
            K: The number of flows in the Glow block.
            apply_dequantization: Boolean value indicating whether to apply Dequantization on data or not.
        """
        super(Glow, self).__init__()
        self.L = L
        self.K = K
        self.in_channel = in_channel

        self.dequant = Dequantization() if apply_dequantization else IdentityTransform()   # VariationalDequantization()
        self.blocks = nn.ModuleList(GlowBlock(in_channels=(2**i * self.in_channel), K=self.K)
                                    for i in range(self.L-1))
        self.final_squeeze = Squeeze()
        self.final_flows = nn.ModuleList(StepFlow(in_channels=(2**(self.L+1) * self.in_channel))
                                         for _ in range(self.K))

    def transform(self, x: Tensor, log_det_jac: Tensor) -> Tuple[Tensor, Tensor]:
        """Computes forward transformation and log abs determinant of jacobian matrix.

        Args:
            x: Input tensor of shape [B, C, H, W].
            log_det_jac: Ongoing log abs determinant of jacobian matrix.

        Returns:
            latent_variables: A list of latent variables corresponding to each GlowBlock and final step.
            log_det_jac: log abs determinant of jacobian matrix of the transformation.
        """
        latent_variables = []

        # Dequantization.
        y, log_det_jac = self.dequant.transform(x, log_det_jac)

        # Glow block.
        for i, block in enumerate(self.blocks):
            y, log_det_jac, z = block.transform(y, log_det_jac)
            latent_variables.append(z)

        # Squeeze.
        y, log_det_jac = self.final_squeeze.transform(y, log_det_jac)

        # Step of flow.
        for flow in self.final_flows:
            y, log_det_jac = flow.transform(y, log_det_jac)

        latent_variables.append(y)
        return latent_variables, log_det_jac

    def invert(self, latents: list) -> Tensor:
        """Computes inverse transformation.

        Args:
            latents: A list of latent variables.

        Returns:
            inv_y: Inverse transformed input.
        """
        # Step of flow.
        inv_y = latents[-1]
        for flow in reversed(self.final_flows):
            inv_y = flow.invert(inv_y)

        # Squeeze.
        inv_y = self.final_squeeze.invert(inv_y)

        # Glow block.
        for i, block in enumerate(reversed(self.blocks)):
            inv_y = block.invert(inv_y, latents[-(2+i)])

        # Dequantization.
        inv_y = self.dequant.invert(inv_y)

        return inv_y

    @torch.no_grad()
    def sample(self, latents: list, postprocess_func=None) -> Tensor:
        """Samples from Glow model.

        Args:
            latents: A list of latent variables.
            postprocess_func: Postprocessor function to generate final output.

        Returns:
            A Tensor representing sampled data points.
        """
        self.eval()
        new_samples = self.invert(latents)
        self.train()
        final_output = postprocess_func(new_samples.float()) if postprocess_func else new_samples.float()
        return final_output


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

    def transform(self, x: Tensor, log_det_jac: Tensor) -> Tuple[Tensor, Tensor]:
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

    def invert(self, y: Tensor) -> Tensor:
        """Computes inverse transformation.

        Args:
            y: Input tensor of shape [B, C, H, W].

        Returns:
            inv_y: Inverse transformed input.
        """
        inv_y, _ = self.sigmoid(y, ignore_logdet=True)
        inv_y *= self.quants
        inv_y = torch.floor(inv_y).clamp(min=0, max=self.quants - 1).to(torch.int32)
        return inv_y

    def sigmoid(self, z: Tensor, log_det_jac: Tensor = None, reverse=False, ignore_logdet=False) \
            -> Tuple[Tensor, Tensor]:
        """Invertible sigmoid transformation.

        Args:
            z: Input tensor.
            log_det_jac: Ongoing log abs determinant of jacobian.
            reverse: Whetehr to apply inverse transformation or not.
            ignore_logdet: Either to ignore computation of logdet or not.

        Returns:
            y: Forward transformed input (or inverse transformed dependent on reverse parameter).
            log_det_jac: log abs determinant of jacobian matrix of the transformation (or inv_log_det_jac dependent
                on reverse parameter).
        """
        if not reverse:
            if not ignore_logdet:
                log_det_jac += (-z-2*softplus(-z)).sum(dim=[1, 2, 3])
            z = torch.sigmoid(z)
        else:
            z = z * (1 - self.alpha) + 0.5 * self.alpha  # Scales to prevent boundaries 0 and 1.
            if not ignore_logdet:
                log_det_jac += np.log(1 - self.alpha) * np.prod(z.shape[1:])
                log_det_jac += (-torch.log(z) - torch.log(1-z)).sum(dim=[1, 2, 3])
            z = torch.log(z) - torch.log(1-z)
        return z, log_det_jac

    def dequant(self, z: Tensor, log_det_jac: Tensor) -> Tuple[Tensor, Tensor]:
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
