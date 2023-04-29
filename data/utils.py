import os
import pickle
from itertools import chain
from sys import platform

import numpy as np
import pandas as pd
from PIL import Image
from skimage import io
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
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


class FilteredMNIST(Dataset):
    """Extends MNIST class to filter digits and convert to RGB."""

    def __init__(self, data_dir, split, digits: list, transform=None):
        if digits is None:
            digits = set(range(10))
        train = split == "train"
        mnist_dataset = MNIST(root=data_dir, train=train, download=True)
        self.data = [tup for tup in mnist_dataset if tup[1] in digits]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        img, label = self.data[idx][0], self.data[idx][1]
        return (self.transform(img), label) if self.transform else (img, label)


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


def set_start_method_for_darwin():
    if platform == "darwin":
        import multiprocessing
        multiprocessing.set_start_method("fork")


def filter_non_nones(*arguments):
    """Filters out Nones from arguments."""
    return [arg for arg in arguments if arg is not None]


def merge_generators(*generators):
    """Merges generators to act as a single unit."""
    return chain(*filter_non_nones(*generators))


def unpickle(file):
    with open(file, "rb") as fo:
        data_dict = pickle.load(fo)
    return data_dict


class ImageNet(Dataset):

    def __init__(self, data_dir, split, transform, res=32):
        # res is either 32 or 64, pick it from the name of the dataset.
        # split is either 'train' or 'val'.

        assert res in (32, 64), "Only resolutions 32 and 64 are supported for ImageNet."
        assert split in ("train", "val"), "Only resolutions 32 and 64 are supported for ImageNet."

        if split == "train" and res == 32:
            files = [os.path.join(data_dir, f"{split}/{split}_data_batch_{idx}") for idx in range(1, 11)]
            data = np.vstack([unpickle(file)["data"] for file in files])
            self.labels = np.hstack([unpickle(file)["labels"] for file in files])
        else:
            dataset = unpickle(os.path.join(data_dir, f"{split}/{split}_data"))
            data, self.labels = dataset["data"], np.array(dataset["labels"])

        data = np.dstack((data[:, :res ** 2], data[:, res ** 2:2 * res ** 2], data[:, 2 * res ** 2:]))
        self.data = data.reshape((data.shape[0], res, res, 3))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        pil_img = Image.fromarray(self.data[idx], "RGB")
        return (self.transform(pil_img), self.labels[idx]) if self.transform else (pil_img, self.labels[idx])
