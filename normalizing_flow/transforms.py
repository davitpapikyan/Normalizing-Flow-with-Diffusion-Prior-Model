from typing import Tuple

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor
from torch.nn.functional import conv2d

from .base import Transform, Prior
from .prior import IsotropicGaussian
from .utils import coupling_network, ZeroConv2d


class IdentityTransform(Transform):
    """Identity transformation.
    """

    def __init__(self):
        super(IdentityTransform, self).__init__()

    def transform(self, x: Tensor, log_det_jac: Tensor, logp: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        return x, log_det_jac, logp

    def invert(self, y: Tensor) -> Tensor:
        return y


class ActNorm(Transform):
    """Activation normalization layer.

    Performs affine transformation using a scale and bias parameter
    per channel. The parameters are initialized such that transformed
    activations per channel have zero mean and unit variance for the
    initial minibatch of data. After this step, both parameters are
    updated as regular learnable parameters.

    Attributes:
        scale: The scale parameter of the affine transformation.
        bias: The bias parameter of the affine transformation.
    """

    def __init__(self, in_channels: int = 3):
        """Initializes weights. Note that this is pseudo-initialization, as the
        actual one happens once we first call forward transform.

        Args:
            in_channels: The number of channels of an input image.
        """
        super(ActNorm, self).__init__()
        self.scale = nn.Parameter(torch.zeros(size=(in_channels, 1, 1), device=self.device))
        self.bias = nn.Parameter(torch.zeros(size=(in_channels, 1, 1), device=self.device))

        # Controls the initialization.
        self.register_buffer("is_initialized", torch.tensor(0, dtype=torch.uint8, device=self.device))

    def transform(self, x: Tensor, log_det_jac: Tensor, logp: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Computes forward transformation and log abs determinant of jacobian matrix.
        The latter is computed via the formula: height * width * \sum \log(|scale|).

        Args:
            x: Input tensor of shape [B, C, H, W].
            log_det_jac: Ongoing log abs determinant of jacobian.
            logp: Ongoing latent variable log probability calculated using prior distribution.

        Returns:
            y: Forward transformed input.
            log_det_jac: log abs determinant of jacobian matrix of the transformation.
            logp: Latent variable log probability calculated using prior distribution.

        """
        _, _, height, width = x.shape

        # Initialization of weights is inserted into transform method as it depends on the input data.
        if self.is_initialized.item() == 0:
            with torch.no_grad():
                self.scale.data.copy_(rearrange(-torch.log(x.std(dim=(0, 2, 3)) + 1e-6), "c -> c () ()"))
                self.bias.data.copy_(rearrange(-x.mean(dim=(0, 2, 3)), "c -> c () ()"))
                self.is_initialized.fill_(1)  # Set is_initialized to True.

        y = torch.exp(self.scale) * (x + self.bias)
        log_det_jac += height * width * self.scale.sum()  # self.scale is already in logarithm.
        return y, log_det_jac, logp

    def invert(self, y: Tensor) -> Tensor:
        """Computes inverse transformation.

        Args:
            y: Input tensor of shape [B, C, H, W].

        Returns:
            inv_y: Inverse transformed input.
        """
        inv_y = y * torch.exp(-self.scale) - self.bias
        return inv_y


class InvConv2d(Transform):
    """Invertible 1x1 convolution.

    Attributes:
        weight: Weight matrix if the 2D convolution.
    """

    def __init__(self, in_channels: int = 3):
        """Initializes weights.

        Args:
            in_channels: The number of channels of an input image.
        """

        super(InvConv2d, self).__init__()
        random_matrix = torch.randn(size=(in_channels, in_channels), dtype=torch.float32, device=self.device)
        w_init, _ = torch.linalg.qr(random_matrix)
        self.weight = nn.Parameter(rearrange(w_init, "h w -> h w () ()"))

    def transform(self, x: Tensor, log_det_jac: Tensor, logp: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Computes forward transformation and log abs determinant of jacobian matrix.
        The latter is computed via the formula: height * width * \log |det(W)|.

        Args:
            x: Input tensor of shape [B, C, H, W].
            log_det_jac: Ongoing log abs determinant of jacobian.
            logp: Ongoing latent variable log probability calculated using prior distribution.

        Returns:
            y: Forward transformed input.
            log_det_jac: log abs determinant of jacobian matrix of the transformation.
            logp: Latent variable log probability calculated using prior distribution.
        """
        _, _, height, width = x.shape
        log_det_jac += height * width * torch.slogdet(self.weight.squeeze().double())[1].float()
        y = conv2d(x, self.weight)
        return y, log_det_jac, logp

    def invert(self, y: Tensor) -> Tensor:
        """Computes inverse transformation.

        Args:
            y: Input tensor of shape [B, C, H, W].

        Returns:
            inv_y: Inverse transformed input.
        """
        inv_y = conv2d(y, rearrange(self.weight.squeeze().inverse(), "h w -> h w () ()"))
        return inv_y


class AffineCoupling(Transform):
    """Affine coupling layer.

    Attributes:
        net: Neural network used to predict the parameters of affine transformation.
    """

    def __init__(self, in_channels: int = 2, n_features: int = 512):
        """Initializes network.

        Args:
            in_channels: The number of input channels of coupling network.
            n_features: The number of hidden feature maps of the network.
        """
        super(AffineCoupling, self).__init__()
        self.net = coupling_network(in_channels=in_channels//2, n_features=n_features,
                                    out_channels=in_channels).to(self.device)

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
        x_a, x_b = x.chunk(2, dim=1)
        log_scale, bias = self.net(x_a).chunk(2, dim=1)
        scale = torch.sigmoid(log_scale+2.0)
        y_a, y_b = x_a, (x_b + bias) * scale
        y = torch.concat([y_a, y_b], dim=1)
        log_det_jac += torch.sum(torch.log(scale+1e-6).view(x.shape[0], -1), 1)
        return y, log_det_jac, logp

    def invert(self, y: Tensor) -> Tensor:
        """Computes inverse transformation.

        Args:
            y: Input tensor of shape [B, C, H, W].

        Returns:
            inv_y: Inverse transformed input.
        """
        y_a, y_b = y.chunk(2, dim=1)
        log_scale, bias = self.net(y_a).chunk(2, dim=1)
        scale = torch.sigmoid(log_scale + 2.0)
        inv_y_a, inv_y_b = y_a, y_b / (scale+1e-6) - bias
        inv_y = torch.concat([inv_y_a, inv_y_b], 1)
        return inv_y


class Squeeze(Transform):
    """Implements squeeze operation on images.
    """

    def __init__(self):
        """Initializes squeeze flow."""
        super(Squeeze, self).__init__()

    def transform(self, x: Tensor, log_det_jac: Tensor, logp: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Transforms x tensor of shape [B, C, H, W] into a tensor of shape [B, 4C, H//2, W//2].
        Not that the log abs determinant is 0.

        Args:
            x: Input tensor of shape [B, C, H, W].
            log_det_jac: Ongoing log abs determinant of jacobian.
            logp: Ongoing latent variable log probability calculated using prior distribution.

        Returns:
            y: Forward transformed input.
            log_det_jac: log abs determinant of jacobian matrix of the transformation.
            logp: Latent variable log probability calculated using prior distribution.
        """
        y = rearrange(x, 'b c (h h1) (w w1) -> b (c h1 w1) h w', h1=2, w1=2)
        return y, log_det_jac, logp

    def invert(self, y: Tensor) -> Tensor:
        """Transforms y tensor of shape [B, C, H, W] into a tensor of shape [B, C//4, H*2, W*2].

        Args:
            y: Input tensor of shape [B, C, H, W].

        Returns:
            inv_y: Inverse transformed input.
        """
        inv_y = rearrange(y, 'b (c c1 c2) h w -> b c (h c1) (w c2)', c1=2, c2=2)
        return inv_y


class Split(Transform):
    """Implements split operation on images.
    """

    def __init__(self, in_channels, learn_prior_mean_logs: bool = True):
        """Initialization.

        Args:
            in_channels: The number of channels of an input image.
            learn_prior_mean_logs: Whether to learn mean and covariance of Gaussian prior.
        """
        super(Split, self).__init__()
        self.conv = ZeroConv2d(in_channels // 2, in_channels, padding=(3-1)//2) if learn_prior_mean_logs else None

    def __init_prior(self, x: Tensor) -> Prior:
        """Initializes and returns IsotropicGaussian object to serve as prior.

        Args:
            x: Input tensor based on which to learn prior distribution's parameters if learn_prior_mean_logs is set to
                True.

        Returns:
            Prior distribution object.
        """
        h = self.conv(x) if self.conv else torch.zeros(size=(x.shape[0], 2*x.shape[1], *x.shape[2:]))
        mean, logs = torch.chunk(h, 2, dim=1)
        return IsotropicGaussian(mean, logs)

    def transform(self, x: Tensor, log_det_jac: Tensor, logp: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Transforms x tensor of shape [B, C, H, W] into a tensor of shape [B, C//2, H, W]
        by splitting it channel-wise. Computes log probability of splited latent variable.

        Args:
            x: Input tensor of shape [B, C, H, W].
            log_det_jac: Ongoing log abs determinant of jacobian.
            logp: Ongoing latent variable log probability calculated using prior distribution. If logp is None then
                prior computation is being omitted.

        Returns:
            y: Forward transformed input.
            log_det_jac: log abs determinant of jacobian matrix of the transformation.
            z: Splited latent variable.
            logp: Latent variable log probability calculated using prior distribution.
        """
        y, y_split = x.chunk(2, dim=1)
        if logp is not None:
            prior_dist = self.__init_prior(y)
            logp += prior_dist.compute_log_prob(y_split)
        return y, log_det_jac, y_split, logp

    def invert(self, y: Tensor, inv_y_split: Tensor = None, temperature: float = 1.0) -> Tensor:
        """Transforms y tensor of shape [B, C, H, W] into a tensor of shape [B, 2*C, H, W]
        by concatenating with sampled latent variable inv_y_split if exists, otherwise,
        samples using prior distribution.

        Args:
            y: Input tensor of shape [B, C, H, W].
            inv_y_split: Latent variable.
            temperature: The temperature parameter.

        Returns:
            inv_y: Inverse transformed input.
        """
        if inv_y_split is None:
            prior_dist = self.__init_prior(y)
            inv_y_split = prior_dist.sample(None, temperature=temperature)
        inv_y = torch.concat([y, inv_y_split], dim=1)
        return inv_y
