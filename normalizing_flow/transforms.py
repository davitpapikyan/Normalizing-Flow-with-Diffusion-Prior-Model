import torch
import torch.nn as nn
from einops import rearrange
from torch.nn.functional import conv2d

from .base import Transform
from .utils import coupling_network, abs_log_sum


class IdentityTransform(Transform):
    """Identity transformation.
    """

    def __init__(self):
        super(IdentityTransform, self).__init__()

    def transform(self, x: torch.tensor, log_det_jac: torch.tensor):
        return x, log_det_jac

    def invert(self, y: torch.tensor, inv_log_det_jac: torch.tensor):
        return y, inv_log_det_jac


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

    def transform(self, x: torch.tensor, log_det_jac: torch.tensor):
        """Computes forward transformation and log abs determinant of jacobian matrix.
        The latter is computed via the formula: height * width * \sum \log(|scale|).

        Args:
            x: Input tensor of shape [B, C, H, W].
            log_det_jac: Ongoing log abs determinant of jacobian.

        Returns:
            y: Forward transformed input.
            log_det_jac: log abs determinant of jacobian matrix of the transformation.
        """
        _, _, height, width = x.shape
        log_det_jac += height * width * self.scale.sum()  # self.scale is already in logarithm.

        if self.is_initialized.item() == 0:  # If not initialized.
            with torch.no_grad():
                self.scale.data.copy_(rearrange(-torch.log(torch.clamp(x.std(dim=(0, 2, 3)), 1e-20)), "c -> c () ()"))
                self.bias.data.copy_(rearrange(-x.mean(dim=(0, 2, 3)), "c -> c () ()"))
                self.is_initialized.fill_(1)  # Set is_initialized to True.

        y = torch.exp(self.scale) * x + self.bias
        return y, log_det_jac

    def invert(self, y: torch.tensor, inv_log_det_jac: torch.tensor):
        """Computes inverse transformation and log abs determinant of jacobian matrix.
        The latter is computed via the formula: -height * width * \sum \log(|scale|).

        Args:
            y: Input tensor of shape [B, C, H, W].
            inv_log_det_jac: Ongoing log abs determinant of jacobian.

        Returns:
            inv_y: Inverse transformed input.
            inv_log_det_jac: log abs determinant of jacobian matrix of the inverse transformation.
        """
        _, _, height, width = y.shape
        inv_log_det_jac += -height * width * self.scale.sum()
        inv_y = (y - self.bias) * torch.exp(-self.scale)
        return inv_y, inv_log_det_jac


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

    def transform(self, x: torch.tensor, log_det_jac: torch.tensor):
        """Computes forward transformation and log abs determinant of jacobian matrix.
        The latter is computed via the formula: height * width * \log |det(W)|.

        Args:
            x: Input tensor of shape [B, C, H, W].
            log_det_jac: Ongoing log abs determinant of jacobian.

        Returns:
            y: Forward transformed input.
            log_det_jac: log abs determinant of jacobian matrix of the transformation.
        """
        _, _, height, width = x.shape
        log_det_jac += height * width * self.weight.squeeze().det().abs().log()
        y = conv2d(x, self.weight)
        return y, log_det_jac

    def invert(self, y: torch.tensor, inv_log_det_jac: torch.tensor):
        """Computes inverse transformation and log abs determinant of jacobian matrix.
        The latter is computed via the formula: -height * width * \log |det(W)|.
        Args:
            y: Input tensor of shape [B, C, H, W].
            inv_log_det_jac: Ongoing log abs determinant of jacobian.

        Returns:
            inv_y: Inverse transformed input.
            inv_log_det_jac: log abs determinant of jacobian matrix of the inverse transformation.
        """
        _, _, height, width = y.shape
        inv_log_det_jac += -height * width * self.weight.squeeze().det().abs().log()
        inv_y = conv2d(y, rearrange(self.weight.squeeze().inverse(), "h w -> h w () ()"))
        return inv_y, inv_log_det_jac


class InvConv2dLU(Transform):
    """Invertible 1x1 convolution.

    For efficient computation of the determinant of weight matrix, it is computed by
    applying LU decomposition to the weight matrix: W = P @ L @ (U + diag(s)).
    Then \log |det(W)| = \sum \log(|s|).

    Attributes:
        w_P: Permutation matrix P.
        w_L: Lower triangular matrix L with ones on the main diagonal.
        w_U: Upper triangular matrix U with zeros on the main diagonal.
        w_s: A vector of main diagonal of U.
    """

    def __init__(self, in_channels: int = 3):
        """Initializes weights.

        Args:
            in_channels: The number of channels of an input image.
        """
        super(InvConv2dLU, self).__init__()

        # Using Q matrix from the QR decomposition of a random matrix as the initial
        # value of weight to make sure it is invertible.

        random_matrix = torch.randn(size=(in_channels, in_channels), dtype=torch.float32, device=self.device)
        w_init, _ = torch.linalg.qr(random_matrix)

        # Constructing P, L, U matrices and s vector for efficient
        # computation of log abs det jacobian.

        q_lu, pivots = torch.lu(w_init)
        w_p, w_l, w_u = torch.lu_unpack(q_lu, pivots)
        w_s, w_u = w_u.diag(), w_u.fill_diagonal_(0)

        self.w_s = nn.Parameter(w_s)
        self.w_L = nn.Parameter(w_l)
        self.w_U = nn.Parameter(w_u)

        # The value of permutation matrix P remains fixed. See 3.2 section
        # of (arXiv:1807.03039) for reference.

        self.register_buffer("w_P", w_p)

    def transform(self, x: torch.tensor, log_det_jac: torch.tensor):
        """Computes forward transformation and log abs determinant of jacobian matrix.
        The latter is computed via the formula: height * width * \log |det(W)| where
        \log |det(W)| = \sum \log(|s|).

        Args:
            x: Input tensor of shape [B, C, H, W].
            log_det_jac: Ongoing log abs determinant of jacobian.

        Returns:
            y: Forward transformed input.
            log_det_jac: log abs determinant of jacobian matrix of the transformation.
        """
        _, _, height, width = x.shape
        weight = self.__compose_weight()
        y = conv2d(x, weight)
        log_det_jac += height * width * abs_log_sum(self.w_s)
        return y, log_det_jac

    def invert(self, y: torch.tensor, inv_log_det_jac: torch.tensor):
        """Computes inverse transformation and log abs determinant of jacobian matrix.
        The latter is computed via the formula: -height * width * \log |det(W)| where
        \log |det(W)| = \sum \log(|s|).

        Args:
            y: Input tensor of shape [B, C, H, W].
            inv_log_det_jac: Ongoing log abs determinant of jacobian.

        Returns:
            inv_y: Inverse transformed input.
            inv_log_det_jac: log abs determinant of jacobian matrix of the inverse transformation.
        """
        _, _, height, width = y.shape
        weight = self.__compose_weight()
        inv_y = conv2d(y, rearrange(weight.squeeze().inverse(), "h w -> h w () ()"))
        inv_log_det_jac += -height * width * abs_log_sum(self.w_s)
        return inv_y, inv_log_det_jac

    def __compose_weight(self):
        """Creates weight matrix via the formula W = P @ L @ (U + diag(s)).

        Returns:
            Weight matrix for invertible 1x1 convolution.
        """
        weight_matrix = self.w_P @ self.w_L @ (self.w_U + torch.diag(self.w_s))
        return rearrange(weight_matrix, "h w -> h w () ()")


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
        self.net = coupling_network(in_channels=in_channels, n_features=n_features).to(self.device)
        self.scaling_factor = nn.Parameter(torch.zeros(in_channels))

    def transform(self, x: torch.tensor, log_det_jac: torch.tensor):
        """Computes forward transformation and log abs determinant of jacobian matrix.

        Args:
            x: Input tensor of shape [B, C, H, W].
            log_det_jac: Ongoing log abs determinant of jacobian.

        Returns:
            y: Forward transformed input.
            log_det_jac: log abs determinant of jacobian matrix of the transformation.
        """
        x_a, x_b = x.chunk(2, dim=1)
        log_scale, bias = self.net(x_a).chunk(2, dim=1)

        s_fac = rearrange(self.scaling_factor.exp(), 'c -> () c () ()')
        scale = torch.tanh(log_scale/s_fac) * s_fac

        y_a, y_b = x_a, (x_b+bias) * scale.exp()
        inv_y = torch.concat([y_a, y_b], dim=1)
        log_det_jac += scale.view(x.size(0), -1).sum(dim=1)
        return inv_y, log_det_jac

    def invert(self, y: torch.tensor, inv_log_det_jac: torch.tensor):
        """Computes inverse transformation and log abs determinant of jacobian matrix.

        Args:
            y: Input tensor of shape [B, C, H, W].
            inv_log_det_jac: Ongoing log abs determinant of jacobian.

        Returns:
            inv_y: Inverse transformed input.
            inv_log_det_jac: log abs determinant of jacobian matrix of the inverse transformation.
        """
        y_a, y_b = y.chunk(2, dim=1)
        log_scale, bias = self.net(y_a).chunk(2, dim=1)

        s_fac = rearrange(self.scaling_factor.exp(), 'c -> () c () ()')
        scale = torch.tanh(log_scale / s_fac) * s_fac

        inv_y_a, inv_y_b = y_a, y_b * torch.exp(-scale) - bias
        inv_y = torch.concat([inv_y_a, inv_y_b], dim=1)
        inv_log_det_jac += -scale.view(y.size(0), -1).sum(dim=1)
        return inv_y, inv_log_det_jac


class Squeeze(Transform):
    """Implements squeeze operation on images.
    """

    def __init__(self):
        """Initializes squeeze flow."""
        super(Squeeze, self).__init__()

    def transform(self, x: torch.tensor, log_det_jac: torch.tensor):
        """Transforms x tensor of shape [B, C, H, W] into a tensor of shape [B, 4C, H//2, W//2].
        Not that the log abs determinant is 0.

        Args:
            x: Input tensor of shape [B, C, H, W].
            log_det_jac: Ongoing log abs determinant of jacobian.

        Returns:
            y: Forward transformed input.
            log_det_jac: log abs determinant of jacobian matrix of the transformation.
        """
        y = rearrange(x, 'b c (h h1) (w w1) -> b (c h1 w1) h w', h1=2, w1=2)
        return y, log_det_jac

    def invert(self, y: torch.tensor, inv_log_det_jac: torch.tensor):
        """Transforms y tensor of shape [B, C, H, W] into a tensor of shape [B, C//4, H*2, W*2].
        Not that the log abs determinant is 0.

        Args:
            y: Input tensor of shape [B, C, H, W].
            inv_log_det_jac: Ongoing log abs determinant of jacobian.

        Returns:
            inv_y: Inverse transformed input.
            inv_log_det_jac: log abs determinant of jacobian matrix of the inverse transformation.
        """
        inv_y = rearrange(y, 'b (c c1 c2) h w -> b c (h c1) (w c2)', c1=2, c2=2)
        return inv_y, inv_log_det_jac


class Split(Transform):
    """Implements split operation on images."""

    def __init__(self, prior):
        """Initializes prior distribution.

        Args:
            prior: Initialized prior distribution.
        """
        super(Split, self).__init__()
        self.__prior = prior  # TODO: Delete prior

    def transform(self, x: torch.tensor, log_det_jac: torch.tensor, return_latent: bool = False):
        """Transforms x tensor of shape [B, C, H, W] into a tensor of shape [B, C//2, H, W]
        by splitting it channel-wise. Log determinant is computed for the second part of split.

        Args:
            x: Input tensor of shape [B, C, H, W].
            log_det_jac: Ongoing log abs determinant of jacobian.
            return_latent: Either to return latent variable or not.

        Returns:
            y: Forward transformed input.
            log_det_jac: log abs determinant of jacobian matrix of the transformation.
        """
        y, y_split = x.chunk(2, dim=1)
        # log_det_jac += self.__prior.compute_log_prob(y_split)
        return (y, log_det_jac, y_split) if return_latent else (y, log_det_jac, None)

    def invert(self, y: torch.tensor, inv_log_det_jac: torch.tensor):
        """Transforms y tensor of shape [B, C, H, W] into a tensor of shape [B, 2*C, H, W]
        by sampling from the prior distribution.

        Args:
            y: Input tensor of shape [B, C, H, W].
            inv_log_det_jac: Ongoing log abs determinant of jacobian.

        Returns:
            inv_y: Inverse transformed input.
            inv_log_det_jac: log abs determinant of jacobian matrix of the inverse transformation.
        """
        inv_y_split = self.__prior.sample(y.size())
        inv_y = torch.concat([y, inv_y_split], dim=1)
        inv_log_det_jac += -self.__prior.compute_log_prob(inv_y_split)
        return inv_y, inv_log_det_jac
