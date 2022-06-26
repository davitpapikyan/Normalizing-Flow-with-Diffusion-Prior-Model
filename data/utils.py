import os

import pandas as pd
from skimage import io
from torch.utils.data import Dataset
import torch


class CelebA(Dataset):
    """Creates CelebA dataset."""

    def __init__(self, data_dir, partition_file_path, split, transform):
        self.partition_file = pd.read_csv(partition_file_path)
        self.data_dir, self.transform = data_dir, transform
        self.partition_file_sub = self.partition_file[self.partition_file["partition"].isin(split)]

    def __len__(self):
        return len(self.partition_file_sub)

    def __getitem__(self, idx: int):
        img_name = os.path.join(self.data_dir, self.partition_file_sub.iloc[idx, 0])
        return self.transform(io.imread(img_name)) if self.transform else io.imread(img_name)


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


def discretize(img):
    return (img * 255).to(torch.int32)


__all__ = [CelebA, MapDataset]
