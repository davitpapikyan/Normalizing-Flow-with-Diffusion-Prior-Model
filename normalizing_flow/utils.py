import os
from typing import Tuple, List, Sequence

import aim
import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as vutils
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms as vision_tranforms

LatentDim = Tuple[int, int, int]  # Channels, height, width.


class ZeroConv2d(nn.Conv2d):
    """nn.Conv2d module initialized with zeros. Output is scaled channel-wise.

    Attributes:
        logscale_factor: Scaling factor applied inside exponent.
    """

    def __init__(self, in_channels: int, out_channels: int, filter_size: int = 3, stride: int = 1, padding: int = 0,
                 logscale: float = 3.):
        """Initializes weights.

        Args:
            in_channels: Input channels.
            out_channels: Output channels.
            filter_size: Kernel size.
            stride: Stride.
            padding: Padding.
            logscale: Scaling factor.
        """
        super(ZeroConv2d, self).__init__(in_channels, out_channels, filter_size, stride=stride, padding=padding)
        self.weight.data.zero_()
        self.bias.data.zero_()

        self.register_parameter("logs", nn.Parameter(torch.zeros(1, out_channels, 1, 1)))
        self.logscale_factor = logscale

    def forward(self, x):
        return super().forward(x) * torch.exp(self.logs * self.logscale_factor)


class Conv2dActNorm(nn.Module):
    """Convolutional operation followed by actnorm.
    """

    def __init__(self, in_channels: int, out_channels: int, filter_size: int, stride: int = 1, padding: int = None):
        """Initializes operations.

        Args:
            in_channels: Input channels.
            out_channels: Output channels.
            filter_size: Kernel size.
            stride: Stride.
            padding: Padding.
        """
        super(Conv2dActNorm, self).__init__()
        from .transforms import ActNorm
        padding = (filter_size - 1) // 2 or padding
        self.__conv = nn.Conv2d(in_channels, out_channels, filter_size, stride=stride, padding=padding, bias=False)
        self.__actnorm = ActNorm(out_channels)

    def forward(self, x):
        temp = torch.zeros(x.size(0), device=self.__actnorm.device)
        return self.__actnorm.transform(self.__conv(x), temp, temp)[0]


def coupling_network(in_channels: int, n_features: int = 512, out_channels: int = None):
    """Defines a network to be used in affine coupling layer.

    Args:
        in_channels: Input channels.
        n_features: The number of hidden feature maps.
        out_channels: Output channels.

    Returns:
        Sequential neural network.
    """
    network = nn.Sequential(
        Conv2dActNorm(in_channels, n_features, 3, padding=1),
        nn.ReLU(inplace=True),
        Conv2dActNorm(n_features, n_features, 1, padding=0),
        nn.ReLU(inplace=True),
        ZeroConv2d(n_features, out_channels or in_channels, padding=1)
    )
    return network


def calculate_output_shapes(L: int, in_channels: int, size: int) -> List[LatentDim]:
    """Calculates output shapes of Glow. Model input must be of dimension (in_channels, H, W).

    Args:
        L: The number of blocks in Glow.
        in_channels: The number of channels of input.
        size: The input dimensions.

    Returns:
        A list of tuples where each tuple is a tripple of values (channel, dim 1, dim 2).
        Example ourput for L = 3, in_channels = 3, size = 32:
            [(6, 16, 16), (12, 8, 8), (48, 4, 4)]
    """
    z_shapes = []  # Stores shapes after each block and final output shape.

    for _ in range(L-1):
        if size % 2 != 0:
            raise ValueError("The input dimension is not divisible by 2!")

        in_channels *= 2
        size //= 2
        z_shapes.append((in_channels, size, size))

    z_shapes.append((in_channels * 4, size // 2, size // 2))
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
    return Optimizer(params, lr=lr if lr else params[0]["lr"])


def get_data_transforms(data_name: str, img_size: int, transformations: list = None):
    """Creates train and test data transformations.

    Args:
        data_name: The dataset name.
        img_size: The image size for resizing.
        transformations: A list of transformations.

    Returns:
        Train and test transformations.
    """
    train_transform, test_transform = [], []
    transformations = transformations or []

    # Train data transformations.
    if data_name == "MNIST" and img_size > 28:
        train_transform.append(vision_tranforms.Pad((img_size-28)//2))
    else:
        train_transform.append(vision_tranforms.Resize((img_size, img_size)))

    if "RandomHorizontalFlip" in transformations:
        train_transform.append(vision_tranforms.RandomHorizontalFlip())

    # Test data transformations.
    test_transform.append(vision_tranforms.Resize((img_size, img_size)))

    # Final transformations.
    train_transform.append(vision_tranforms.ToTensor())
    test_transform.append(vision_tranforms.ToTensor())

    train_transform = vision_tranforms.Compose(train_transform)
    test_transform = vision_tranforms.Compose(test_transform)
    return train_transform, test_transform


@torch.no_grad()
def preprocess_batch(batch: Tensor, n_bits: int, n_bins: int) -> Tensor:
    """Preprocesses a batch of images before feeding into NF.
    Make sure that data transformations include ToTensor (i.e. [0, 1] mapping) as final transformation.
    In general, a batch of images passed as input to this function must be within the range [0, 1].

    Args:
        batch: A batch of images of shape [B, C, H, W].
        n_bits: The number of bits to encode.
        n_bins: The number of bins.

    Returns:
        Preprocessed batch.
    """
    processed_batch = batch * 255

    # Encodes each entry in each color-channel in n_bits number of bits.
    if n_bits < 8:
        processed_batch = torch.floor(processed_batch / 2 ** (8 - n_bits))

    processed_batch = processed_batch / n_bins - 0.5
    return processed_batch


@torch.no_grad()
def postprocess_batch(batch: Tensor, n_bins: int) -> Tensor:
    """Postprocesses a batch of images before feeding into NF.

    Args:
        batch: A batch of images of shape [B, C, H, W].
        n_bins: The number of bins.

    Returns:
        Postprocessed batch.
    """
    return torch.clip(torch.floor((batch+0.5)*n_bins) * (256.0/n_bins), 0, 255).to("cpu", torch.uint8)


@torch.no_grad()
def track_images(aim_logger, images: Tensor, step: int = None, epoch: int = None, context: dict = None) -> None:
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
def save_images(images: Tensor, path: str, name: str) -> None:
    """Plots and saves 64 images.

    Args:
        images: Torch tensor of shape [N, C, H, W] where N is the number of samples.
        path: The directory wto save.
        name: The name of the image to save.
    """
    vutils.save_image(images.cpu().data, os.path.join(path, f"{name}.pdf"), normalize=True, nrow=8,
                      value_range=(-0.5, 0.5))


def calculate_loss(log_likelihood: Tensor, n_bins: float, n_pixel: float):
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


def initialize_with_zeros(n: int, batch_size: int, device: torch.device) -> Tuple[Tensor]:
    """Creates n tensors initialized with zeros.

    Args:
        n: The number of tensors.
        batch_size: The dimension of tensors.
        device: Device.

    Returns:
        n tensors filled with zeros of size batch_size.
    """
    if n == 1:
        return torch.zeros(batch_size, device=device, dtype=torch.float64)
    return (torch.zeros(batch_size, device=device, dtype=torch.float64) for _ in range(n))


@torch.no_grad()
def data_dependent_nf_initialization(flow, dataloader: DataLoader, device: torch.device, n_bits: int, n_bins: int) \
        -> None:
    """Initializes Normalizing Flow certain trainsforms (e.g. ActNorm) based on data statistics.

    Args:
        flow: Normalizing Flow model.
        dataloader: The data loader.
        device: Device.
        n_bits: The number of bits.
        n_bins: The number of bins.
    """
    flow.eval()
    sample = next(iter(dataloader))
    batch = sample[0].to(device) if isinstance(sample, list) else sample.to(device)
    batch = preprocess_batch(batch, n_bits, n_bins)
    log_likelihood, logp = initialize_with_zeros(2, batch.size(0), device)
    _ = flow.transform(batch + torch.rand_like(batch) / n_bins, log_likelihood, logp)


def get_item(sequence: Sequence, index: int):
    """Helper function to return sequence[index] if index is valid."""
    try:
        return sequence[index]
    except IndexError:
        return None
