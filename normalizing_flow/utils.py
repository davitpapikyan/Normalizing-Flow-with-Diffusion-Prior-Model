import os
from typing import Tuple

import aim
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as vutils
from PIL import Image


def abs_log_sum(x: torch.tensor) -> torch.tensor:
    """Computes sum(log|x+eps|).

    Args:
        x: The input tensor.

    Returns:
        A scalar result of the formula.
    """
    return torch.clamp(x.abs(), 1e-20).log().sum()


class ZeroConv2d(nn.Module):
    """nn.Conv2d module initialized with zeros. Output is scaled channel-wise."""

    def __init__(self, in_channels: int, out_channels: int):
        """Initializes weights.

        Args:
            in_channels: Input channels.
            out_channels: Output channels.
        """
        super(ZeroConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

        self.scale = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

    def forward(self, x):
        return self.conv(x) * torch.exp(self.scale * 3)


def coupling_network(in_channels: int, n_features: int = 512):
    """Defines a network to be used in affine coupling layer.
    The network's input and output dimensions are the same.

    Args:
        in_channels: Input channels.
        n_features: The number of hidden feature maps.

    Returns:
        Sequential neural network.
    """
    network = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=n_features, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=n_features, out_channels=n_features, kernel_size=1),
        nn.ReLU(inplace=True),
        ZeroConv2d(in_channels=n_features, out_channels=2*in_channels)
    )
    return network


def calc_chunk_sizes(n_dim: int) -> Tuple[int, int]:
    """Calculates the output shapes of torch.chunk(chunks=2, ...) operation.

    Args:
        n_dim: The size of input dimension along which to split the tensor.

    Returns:
        A tuple of integers representing the dimensions of output tensors.
    """
    if n_dim % 2 == 0:
        out = n_dim // 2
        return out, out
    else:
        return (n_dim+1) // 2, n_dim // 2


def init_optimizer(name: str, params, lr: float) -> torch.optim.Optimizer:
    """Initializes optimizer.

    Args:
        name: The name of optimizer.
        params: The list of learnable parameters.
        lr: The learning rate.

    Returns:
        A torch optimizer.
    """
    if name == "adam":
        from torch.optim import Adam as Optimizer
    elif name == "adamw":
        from torch.optim import AdamW as Optimizer
    else:
        raise ValueError("Unknown optimizer")
    return Optimizer(params, lr=lr)


@torch.no_grad()
def track_images(aim_logger, images: torch.tensor, step: int = None, epoch: int = None, context: dict = None) -> None:
    """Adds images to aim to be tracked.

    Args:
        aim_logger: Aim logger.
        images: Torch tensor of shape [N, C, H, W] where N is the number of samples.
        step: Sequence tracking iteration.
        epoch: Training epoch.
        context: Sequence tracking context.
    """
    aim_images = []
    for idx, image in enumerate(images):
        ndarr = image.permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        im = Image.fromarray(ndarr)
        aim_images.append(aim.Image(im, caption=f"#{idx}"))

    aim_logger.track(value=aim_images, name="generated", step=step, epoch=epoch, context=context)


@torch.no_grad()
def save_images(images: torch.tensor, path: str, name: str, dpi: int = 200) -> None:
    """Plots and saves 64 images.

    Args:
        images: Torch tensor of shape [N, C, H, W] where N is the number of samples.
        path: The directory wto save.
        name: The name of the image to save.
        dpi: Resolution of the figure.
    """
    assert images.size(0) >= 64, "There must be at least 64 images."
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Generated images")
    plt.imshow(np.transpose(vutils.make_grid(images[:64], normalize=True).cpu(), (1, 2, 0)))
    plt.savefig(os.path.join(path, f"{name}.eps"), dpi=dpi)
    plt.close()
