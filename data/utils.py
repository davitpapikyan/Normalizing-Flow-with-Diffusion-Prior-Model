import os

import pandas as pd
import torch
from skimage import io
from torch.utils.data import Dataset
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


@torch.jit.script  # TODO: Check if this makes iterations faster on server with GPU.
def discretize(img):
    """The opposite transformation of torchvision.transforms.ToTensor().

    Args:
        img: Input image.

    Returns:
        Discretized tesnor image.
    """
    return (img * 255).to(torch.int32)


def set_start_method_for_darwin():
    import multiprocessing
    multiprocessing.set_start_method("fork")
