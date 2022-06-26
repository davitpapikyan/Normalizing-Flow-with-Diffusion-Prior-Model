import logging
import os
from typing import Tuple
from sys import platform

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose
from .utils import discretize

from .utils import CelebA, MapDataset, set_start_method_for_darwin


def get_cifar10_dataloaders(*, root: str, batch_size: int = 4, num_workers: int = 1,
                            train_transform: Compose = None, test_transform: Compose = None,
                            pin_memory: bool = False, verbose: bool = False) -> \
        Tuple[DataLoader, DataLoader, DataLoader]:
    """Data loader. Combines the CIFAR10 dataset and a sampler, and provides an iterable over
    the given dataset.

    Args:
        root: The path to the data.
        batch_size: How many samples per batch to load.
        num_workers: How many subprocesses to use for data loading.
        train_transform: Transformations to be applied on train dataset.
        test_transform: Transformations to be applied on validation and test datasets.
        pin_memory: If True, the data loader will copy Tensors into CUDA pinned memory before returning them.
        verbose: If True provides additional details about the dataset.

    Returns:
        Training, validation and test dataloaders.
    """
    path_to_data = os.path.join(root, "cifar10")
    dataset = CIFAR10(root=path_to_data, train=True, download=False, transform=None)
    indices, labels = np.arange(stop=len(dataset)), [tup[1] for tup in dataset]
    train_indies, val_indices = train_test_split(indices, test_size=0.20, stratify=labels)

    trainset = Subset(dataset, train_indies)
    trainset = MapDataset(trainset, train_transform)
    train_loader = DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

    valset = Subset(dataset, val_indices)
    valset = MapDataset(valset, test_transform)
    val_loader = DataLoader(valset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

    testset = CIFAR10(root=path_to_data, train=False, download=False, transform=test_transform)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)

    if verbose:
        logger = logging.getLogger("base")
        logger.info("Dataset: CIFAR10")
        logger.info(f"Train size: {len(train_indies):,}, val size: {len(val_indices):,}, test size: {len(testset):,}.")

    return train_loader, val_loader, test_loader


def get_celeba_dataloaders(*, root: str, batch_size: int = 4, num_workers: int = 1,
                           train_transform: Compose = None, test_transform: Compose = None,
                           pin_memory: bool = False, verbose: bool = False) -> \
        Tuple[DataLoader, DataLoader, DataLoader]:
    """Data loader. Combines the CelebA dataset and a sampler, and provides an iterable over
    the given dataset.

    Args:
        root: The path to the data.
        batch_size: How many samples per batch to load.
        num_workers: How many subprocesses to use for data loading.
        train_transform: Transformations to be applied on train data.
        test_transform: Transformations to be applied on validation and test data.
        pin_memory: If True, the data loader will copy Tensors into CUDA pinned memory before returning them.
        verbose: If True provides additional details about the dataset.

    Returns:
        Training, validation and test dataloaders.
    """
    if platform == "darwin":
        set_start_method_for_darwin()

    path_to_data = os.path.join(root, "celeba/img_align_celeba/img_align_celeba")
    path_to_partitions = os.path.join(root, "celeba/list_eval_partition.csv")
    trainset = CelebA(data_dir=path_to_data,
                      partition_file_path=path_to_partitions,
                      split=[0], transform=train_transform)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)

    valset = CelebA(data_dir=path_to_data,
                    partition_file_path=path_to_partitions,
                    split=[1], transform=test_transform)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)

    testset = CelebA(data_dir=path_to_data,
                     partition_file_path=path_to_partitions,
                     split=[2], transform=test_transform)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)

    if verbose:
        logger = logging.getLogger("base")
        logger.info("Dataset: CelebA")
        logger.info(f"Train size: {len(trainset):,}, val size: {len(valset):,}, test size: {len(testset):,}.")

    return train_loader, val_loader, test_loader


def read_dataset(*, root: str, name: str, batch_size=4, num_workers=1, train_transform: Compose = None,
                 test_transform: Compose = None, pin_memory=False, verbose: bool = False):
    """Prepares the dataloaders of the specified dataset.

    Args:
        root: The path to the data.
        name: The dataset name.
        batch_size: How many samples per batch to load.
        num_workers: How many subprocesses to use for data loading.
        train_transform: Transformations to be applied on train data.
        test_transform: Transformations to be applied on validation and test data.
        pin_memory: If True, the data loader will copy Tensors into CUDA pinned memory before returning them.
        verbose: If True provides additional details about the dataset.

    Returns:
        Training, validation and test dataloaders.
    """
    if name == "CelebA":
        get_dataloaders = get_celeba_dataloaders
    elif name == "CIFAR10":
        get_dataloaders = get_cifar10_dataloaders
    else:
        raise ValueError("Dataset name is invalid, currently supported datasets are 'CelebA' and 'CIFAR10'.")
    return get_dataloaders(root=root, batch_size=batch_size, num_workers=num_workers, train_transform=train_transform,
                           test_transform=test_transform, pin_memory=pin_memory, verbose=verbose)


__all__ = [read_dataset, discretize]
