import torch
import torch.nn as nn

from .base import Transform
from .transforms import ActNorm, InvConv2d, AffineCoupling, Squeeze, Split, Prior
from .utils import calc_chunk_sizes

import numpy as np
from torch.nn.functional import softplus


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

    def transform(self, x):
        """Computes forward transformation and log abs determinant of jacobian matrix.

        Args:
            x: Input tensor of shape [B, C, H, W].

        Returns:
            y: Forward transformed input.
            log_det_jac: log abs determinant of jacobian matrix of the transformation.
        """
        log_det_jac = torch.zeros(x.size(0), device=self.device)

        y, log_det_jac1 = self.actnorm.transform(x)
        y, log_det_jac2 = self.invconv2d.transform(y)
        y, log_det_jac3 = self.affcoupling.transform(y)

        log_det_jac += log_det_jac1 + log_det_jac2 + log_det_jac3
        return y, log_det_jac

    def invert(self, y):
        """Computes inverse transformation and log abs determinant of jacobian matrix.

        Args:
            y: Input tensor of shape [B, C, H, W].

        Returns:
            inv_y: Inverse transformed input.
            inv_log_det_jac: log abs determinant of jacobian matrix of the inverse transformation.
        """
        inv_log_det_jac = torch.zeros(y.size(0), device=self.device)

        inv_y, inv_log_det_jac1 = self.affcoupling.invert(y)
        inv_y, inv_log_det_jac2 = self.invconv2d.invert(inv_y)
        inv_y, inv_log_det_jac3 = self.actnorm.invert(inv_y)

        inv_log_det_jac += inv_log_det_jac1 + inv_log_det_jac2 + inv_log_det_jac3
        return inv_y, inv_log_det_jac


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

    def transform(self, x: torch.tensor):
        """Computes forward transformation and log abs determinant of jacobian matrix.

        Args:
            x: Input tensor of shape [B, C, H, W].

        Returns:
            y: Forward transformed input.
            log_det_jac: log abs determinant of jacobian matrix of the transformation.
        """
        log_det_jac = torch.zeros(x.size(0), device=self.device)

        # Squeeze.
        y, log_det_jac_squeeze = self.squeeze.transform(x)
        log_det_jac += log_det_jac_squeeze

        # Step of flow.
        for flow in self.flows:
            y, log_det_jac_flow = flow.transform(y)
            log_det_jac += log_det_jac_flow

        # Split.
        y, log_det_jac_split = self.split.transform(y)
        log_det_jac += log_det_jac_split

        return y, log_det_jac

    def invert(self, y: torch.tensor):
        """Computes inverse transformation and log abs determinant of jacobian matrix.

        Args:
            y: Input tensor of shape [B, C, H, W].

        Returns:
            inv_y: Inverse transformed input.
            inv_log_det_jac: log abs determinant of jacobian matrix of the inverse transformation.
        """
        inv_log_det_jac = torch.zeros(y.size(0), device=self.device)

        # Split.
        inv_y, inv_log_det_jac_split = self.split.invert(y)
        inv_log_det_jac += inv_log_det_jac_split

        # Step of flow.
        for flow in reversed(self.flows):
            inv_y, inv_log_det_jac_flow = flow.invert(inv_y)
            inv_log_det_jac += inv_log_det_jac_flow

        # Squeeze.
        inv_y, inv_log_det_squeeze = self.squeeze.invert(inv_y)
        inv_log_det_jac += inv_log_det_squeeze

        return inv_y, inv_log_det_jac


class Glow(Transform):
    """The Glow model.

    Attributes:
        blocks: A list of Glow blocks.
        final_squeeze: Squeeze operation applied in the end of the architecture.
        final_flows: A list of step-flows applied in the end of the architecture.
    """

    def __init__(self, in_channel: int = 3, L: int = 3, K: int = 32):
        """Initializes operations of the model.

        Args:
            in_channel: The number of input channels.
            L: The number of Glow blocks.
            K: The number of flows in the Glow block.
        """
        super(Glow, self).__init__()
        self.L = L
        self.K = K
        self.in_channel = in_channel

        # self.dequant = Dequantization()
        self.dequant = Dequantization()  # VariationalDequantization()
        self.blocks = nn.ModuleList(GlowBlock(in_channels=(2**i * self.in_channel), K=self.K)
                                    for i in range(self.L-1))
        self.final_squeeze = Squeeze()
        self.final_flows = nn.ModuleList(StepFlow(in_channels=(2**(self.L+1) * self.in_channel))
                                         for _ in range(self.K))
        self.__prior = Prior()

    def transform(self, x: torch.tensor):
        """Computes forward transformation and log abs determinant of jacobian matrix.

        Args:
            x: Input tensor of shape [B, C, H, W].

        Returns:
            y: Forward transformed input.
            log_det_jac: log abs determinant of jacobian matrix of the transformation.
        """
        log_det_jac = torch.zeros(x.size(0), device=self.device)

        # Dequantization.
        y, log_det_jac_dequant = self.dequant.transform(x)
        log_det_jac += log_det_jac_dequant

        # Glow block.
        for block in self.blocks:
            y, log_det_jac_block = block.transform(y)
            log_det_jac += log_det_jac_block

        # Squeeze.
        y, log_det_jac_squeeze = self.final_squeeze.transform(y)
        log_det_jac += log_det_jac_squeeze

        # Step of flow.
        for flow in self.final_flows:
            y, log_det_jac_flow = flow.transform(y)
            log_det_jac += log_det_jac_flow

        log_p = self.__prior.compute_log_prob(y)
        log_det_jac += log_p

        return y, log_det_jac

    def invert(self, y: torch.tensor):
        """Computes inverse transformation and

        Args:
            y: Input tensor of shape [B, C, H, W].

        Returns:
            inv_y: Inverse transformed input.
            log_likelihood:
        """
        log_likelihood = torch.zeros(y.size(0), device=self.device)

        log_p = self.__prior.compute_log_prob(y)
        log_likelihood += log_p

        # Step of flow.
        inv_y = y
        for flow in reversed(self.final_flows):
            inv_y, inv_log_det_jac_flow = flow.invert(inv_y)
            log_likelihood += inv_log_det_jac_flow

        # Squeeze.
        inv_y, inv_log_det_jac_squeeze = self.final_squeeze.invert(inv_y)
        log_likelihood += inv_log_det_jac_squeeze

        # Glow block.
        for block in reversed(self.blocks):
            inv_y, log_det_jac_block = block.invert(inv_y)
            log_likelihood += log_det_jac_block

        # Dequantization.
        inv_y, inv_log_det_jac_dequant = self.dequant.invert(inv_y)
        log_likelihood += inv_log_det_jac_dequant

        return inv_y, log_likelihood

    def get_output_shape(self, height, width):
        const = 2 ** self.L
        n_channels = int(self.in_channel * const * 2)
        height = int(height / const)
        width = int(width / const)
        return n_channels, height, width

    @torch.no_grad()
    def sample(self, n_samples, img_shape):
        self.eval()
        in_height, in_width = img_shape
        n_channels, out_height, out_width = self.get_output_shape(in_height, in_width)
        z = self.__prior.sample((n_samples, n_channels, out_height, out_width)).to(self.device)
        new_samples, log_likelihood = self.invert(z)
        self.train()
        return new_samples.float(), log_likelihood


# TODO: Check if @torch.jit applied on transform, invert, sigmoid and dequant introduce speedup or not.
class Dequantization(Transform):

    def __init__(self, alpha=1e-5, quants=256):
        """
        Inputs:
            alpha - small constant that is used to scale the original input.
                    Prevents dealing with values very close to 0 and 1 when inverting the sigmoid
            quants - Number of possible discrete values (usually 256 for 8-bit image)
        """
        super().__init__()
        self.alpha = alpha
        self.quants = quants

    def transform(self, x: torch.tensor):
        log_det_jac = torch.zeros(x.size(0))
        y, log_det_jac_1 = self.dequant(x)
        y, log_det_jac_2 = self.sigmoid(y, reverse=True)
        log_det_jac += log_det_jac_1 + log_det_jac_2
        return y, log_det_jac

    def invert(self, y: torch.tensor):
        inv_log_det_jac = torch.zeros(y.size(0))

        inv_y, inv_log_det_jac_1 = self.sigmoid(y, reverse=False)
        inv_y = inv_y * self.quants
        inv_log_det_jac += inv_log_det_jac_1 + np.log(self.quants) * np.prod(inv_y.shape[1:])
        inv_y = torch.floor(inv_y).clamp(min=0, max=self.quants - 1).to(torch.int32)
        return inv_y, inv_log_det_jac

    def sigmoid(self, z, reverse=False):
        # Applies an invertible sigmoid transformation
        if not reverse:
            ldj = (-z-2*softplus(-z)).sum(dim=[1, 2, 3])
            z = torch.sigmoid(z)
        else:
            z = z * (1 - self.alpha) + 0.5 * self.alpha  # Scale to prevent boundaries 0 and 1
            ldj = np.log(1 - self.alpha) * np.prod(z.shape[1:])
            ldj += (-torch.log(z) - torch.log(1-z)).sum(dim=[1, 2, 3])
            z = torch.log(z) - torch.log(1-z)
        return z, ldj

    def dequant(self, z):
        # Transform discrete values to continuous volumes
        z = z.to(torch.float32)
        z = z + torch.rand_like(z).detach()
        z = z / self.quants
        ldj = -np.log(self.quants) * np.prod(z.shape[1:])
        return z, ldj


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
