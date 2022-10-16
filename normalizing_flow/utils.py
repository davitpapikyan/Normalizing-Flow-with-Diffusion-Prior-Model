import os
from typing import Tuple

import aim
import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as vutils
from PIL import Image
from torchvision import transforms as vision_tranforms
from data import discretize


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

    # network[0].weight.data.normal_(0, 0.05)
    # network[0].bias.data.zero_()
    # network[2].weight.data.normal_(0, 0.05)
    # network[2].bias.data.zero_()
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


def calculate_output_shapes(L, in_channel, size):
    """Calculates output shapes of Glow. Model input must be of dimension (in_channel, size, size).

    Args:
        L:
        in_channel: The number of channels of input.
        size: The input dimensions.

    Returns:
        A list of tuples where each tuple is a tripple of values (channel, dim 1, dim 2).
    """
    z_shapes = []  # Stores shapes after each block and final output shape.

    for _ in range(L-1):
        if size % 2 != 0:
            raise ValueError("The input dimension is not divisible by 2!")

        in_channel *= 2
        size //= 2
        z_shapes.append((in_channel, size, size))

    z_shapes.append((in_channel * 4, size // 2, size // 2))
    return z_shapes


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


def get_data_transforms(data_name, img_size, apply_dequantization=False):
    """Creates train and test data transformations.

    Args:
        data_name: The dataset name.
        img_size: The image size for resizing.
        apply_dequantization: If dequatization is applied, then no preprocessing is needed.

    Returns:
        Train and test transformations.
    """
    train_transform, test_transform = [], []

    # Train data transformations.
    if data_name == "MNIST" and img_size > 28:
        train_transform.append(vision_tranforms.Pad((img_size-28)//2))
    else:
        train_transform.append(vision_tranforms.Resize((img_size, img_size)))

    if data_name in ("CelebA", "CIFAR10"):
        train_transform.append(vision_tranforms.RandomHorizontalFlip())

    # Test data transformations.
    test_transform.append(vision_tranforms.Resize((img_size, img_size)))

    # Final transformations.
    train_transform.append(vision_tranforms.ToTensor())
    test_transform.append(vision_tranforms.ToTensor())

    if apply_dequantization:
        train_transform.append(discretize)
        test_transform.append(discretize)

    train_transform = vision_tranforms.Compose(train_transform)
    test_transform = vision_tranforms.Compose(test_transform)
    return train_transform, test_transform


@torch.no_grad()
def preprocess_batch(batch, n_bits, n_bins, apply_dequantization=False):
    """Preprocesses a batch of images before feeding into NF.

    Args:
        batch: A batch of images of shape [B, C, H, W].
        n_bits: The number of bits to encode.
        n_bins: The number of bins.
        apply_dequantization: If dequatization is applied, then no preprocessing is needed.

    Returns:
        Preprocessed batch.
    """
    if apply_dequantization:
        return batch

    # Make sure that data transformations include ToTensor (i.e. [0, 1] mapping) as final transformation.
    # In general, a batch of images passed as input to this function must be within the range [0, 1].
    processed_batch = batch * 255

    # Encodes each entry in each color-channel in n_bits number of bits.
    if n_bits < 8:
        processed_batch = torch.floor(processed_batch / 2 ** (8 - n_bits))

    processed_batch = processed_batch / n_bins - 0.5
    return processed_batch


@torch.no_grad()
def postprocess_batch(batch, n_bins, apply_dequantization=False):
    """Postprocesses a batch of images before feeding into NF.

    Args:
        batch: A batch of images of shape [B, C, H, W].
        n_bins: The number of bins.
        apply_dequantization: If dequatization is applied, then no preprocessing is needed.

    Returns:
        Postprocessed batch.
    """

    return torch.clip(torch.floor((batch+0.5)*n_bins) * (256.0/n_bins), 0, 255).to("cpu", torch.uint8) if not \
        apply_dequantization else batch


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
    grid = vutils.make_grid(images.cpu().data, normalize=True, value_range=(-0.5, 0.5), padding=1)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    aim_im = aim.Image(im)
    aim_logger.track(value=aim_im, name="generated", step=step, epoch=epoch, context=context)


@torch.no_grad()
def save_images(images: torch.tensor, path: str, name: str) -> None:
    """Plots and saves 64 images.

    Args:
        images: Torch tensor of shape [N, C, H, W] where N is the number of samples.
        path: The directory wto save.
        name: The name of the image to save.
    """
    vutils.save_image(
        images.cpu().data,
        os.path.join(path, f"{name}.eps"),
        normalize=True,
        nrow=8,
        value_range=(-0.5, 0.5),
    )


def calculate_loss(log_likelihood: torch.Tensor, n_bins: float, n_pixel: float):
    """Calculates bits per dimension (BPD) loss.

    Args:
        log_likelihood: Torch tensor of shape [B, ].
        n_bins: The number of bins.
        n_pixel: The number of pixels.

    Returns:
        Bits per dimension.
    """
    bpd_const = np.log2(np.e) / n_pixel
    return ((np.log(n_bins) * n_pixel - log_likelihood) * bpd_const).mean(dim=0)
