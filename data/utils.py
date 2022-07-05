import os
from sys import platform

import pandas as pd
import torch
from skimage import io
from torch.utils.data import Dataset
from torch_fidelity import calculate_metrics
from torchvision.transforms import ToPILImage


class CelebA(Dataset):
    """Creates CelebA dataset."""

    def __init__(self, data_dir, partition_file_path, split, transform):
        self.partition_file = pd.read_csv(partition_file_path)
        self.data_dir, self.transform = data_dir, transform
        self.convert_to_PIL = ToPILImage()
        self.partition_file_sub = self.partition_file[self.partition_file["partition"].isin(split)]

    def __len__(self):
        return len(self.partition_file_sub)

    def __getitem__(self, idx: int):
        img_name = os.path.join(self.data_dir, self.partition_file_sub.iloc[idx, 0])
        pil_img = self.convert_to_PIL(io.imread(img_name))
        return self.transform(pil_img) if self.transform else pil_img


class MapDataset(Dataset):
    """
    Given a dataset, creates a dataset which applies a mapping function
    to its items (lazily, only when an item is called).

    Note that data is not cloned/copied from the initial dataset.
    """

    def __init__(self, dataset, map_fn):
        self.dataset = dataset
        self.map = map_fn

    def __getitem__(self, index):
        if self.map:
            image, label = self.dataset[index]
            return self.map(image), label
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


class FIDDataset(Dataset):
    """Given a dataset or torch.Tensor, creates a dataset to feed torch_fidelity to calculate FID.
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        return self.dataset[index][0] if isinstance(self.dataset[index], tuple) else self.dataset[index]

    def __len__(self):
        return len(self.dataset)


def calculate_fid(input1, input2, device) -> float:
    """Calculates FID score.

    Args:
        input1: Either torch.Tensor or dataset.
        input2: Either torch.Tensor or dataset.
        device: Device.

    Returns:
        FID between input1 and input2.
    """
    cuda = device.type == 'gpu'
    return calculate_metrics(input1=FIDDataset(input1), input2=FIDDataset(input2), batch_size=32, cuda=cuda, isc=False,
                             fid=True, kid=False, verbose=False)["frechet_inception_distance"]


# TODO: Check if @torch.jit.script applied introduces speedup or not.
def discretize(img):
    """The opposite transformation of torchvision.transforms.ToTensor().

    Args:
        img: Input image.

    Returns:
        Discretized tesnor image.
    """
    return (img * 255).to(torch.uint8)  # int32


def set_start_method_for_darwin():
    if platform == "darwin":
        import multiprocessing
        multiprocessing.set_start_method("fork")
