import logging
import os
from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose

from .utils import CelebA, FilteredMNIST, ImageNet, MapDataset, set_start_method_for_darwin


def get_cifar10_mnist_dataloaders(*, data_name: str, root: str, validate: bool = True, digits: list,
                                  batch_size: int = 4, num_workers: int = 1, train_transform: Compose = None,
                                  test_transform: Compose = None, pin_memory: bool = False, verbose: bool = False) -> \
        Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """Data loader. Combines the CIFAR10 dataset and a sampler, and provides an iterable over
    the given dataset.

    Args:
        data_name: The dataset name.
        root: The path to the data.
        validate: Whether to create validation set or not.
        digits: A list of digits to select. If None, all digits will be selected.
        batch_size: How many samples per batch to load.
        num_workers: How many subprocesses to use for data loading.
        train_transform: Transformations to be applied on train dataset.
        test_transform: Transformations to be applied on validation and test datasets.
        pin_memory: If True, the data loader will copy Tensors into CUDA pinned memory before returning them.
        verbose: If True provides additional details about the dataset.

    Returns:
        Training, validation and test dataloaders.
    """
    if data_name == "cifar10":
        set_start_method_for_darwin()
        path_to_data = os.path.join(root, "cifar10")
        dataset = CIFAR10(root=path_to_data, train=True, download=True, transform=None)
    elif data_name == "MNIST":
        path_to_data = os.path.join(root, "MNIST")
        dataset = FilteredMNIST(path_to_data, "train", digits)
    else:
        raise ValueError("Unknown dataset name.")

    if validate:
        indices, labels = np.arange(stop=len(dataset)), [tup[1] for tup in dataset]
        train_indies, val_indices = train_test_split(indices, test_size=0.20, stratify=labels)

        trainset = Subset(dataset, train_indies)
        trainset = MapDataset(trainset, train_transform)
        train_loader = DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
                                  drop_last=False)

        valset = Subset(dataset, val_indices)
        valset = MapDataset(valset, test_transform)
        val_loader = DataLoader(valset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
                                drop_last=False)

        train_size, val_size = len(trainset), len(valset)
    else:
        trainset = MapDataset(dataset, train_transform)
        train_loader = DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
                                  drop_last=False)
        val_loader = None
        train_size, val_size = len(dataset), 0

    if data_name == "cifar10":
        testset = CIFAR10(root=path_to_data, train=False, download=True, transform=test_transform)
        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                 pin_memory=pin_memory, drop_last=False)
    elif data_name == "MNIST":
        testset = FilteredMNIST(path_to_data, "test", digits, test_transform)
        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                 pin_memory=pin_memory, drop_last=False)
    else:
        raise ValueError("Unknown data_name")

    if verbose:
        logger = logging.getLogger("base")
        logger.info(f"Dataset: {data_name}")
        logger.info(f"Train size: {train_size:,}, val size: {val_size:,}, test size: {len(testset):,}.")

    # Note that test_transform must not contain non-deterministic transformation.
    pure_trainset = MapDataset(dataset, test_transform)
    eval_dataloader = DataLoader(pure_trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                 pin_memory=pin_memory, drop_last=False)
    return train_loader, val_loader, test_loader, eval_dataloader


def get_celeba_dataloaders(*, root: str, validate: bool = True, batch_size: int = 4, num_workers: int = 1,
                           train_transform: Compose = None, test_transform: Compose = None,
                           pin_memory: bool = False, verbose: bool = False) -> \
        Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """Data loader. Combines the CelebA dataset and a sampler, and provides an iterable over
    the given dataset.

    Args:
        root: The path to the data.
        validate: Whether to create validation set or not.
        batch_size: How many samples per batch to load.
        num_workers: How many subprocesses to use for data loading.
        train_transform: Transformations to be applied on train data.
        test_transform: Transformations to be applied on validation and test data.
        pin_memory: If True, the data loader will copy Tensors into CUDA pinned memory before returning them.
        verbose: If True provides additional details about the dataset.

    Returns:
        Training, validation and test dataloaders.
    """
    set_start_method_for_darwin()

    path_to_data = os.path.join(root, "celeba/img_align_celeba/img_align_celeba")
    path_to_partitions = os.path.join(root, "celeba/list_eval_partition.csv")

    if validate:
        train_split = [0]
        trainset = CelebA(data_dir=path_to_data,
                          partition_file_path=path_to_partitions,
                          split=train_split, transform=train_transform)
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  pin_memory=pin_memory, drop_last=False)

        valset = CelebA(data_dir=path_to_data,
                        partition_file_path=path_to_partitions,
                        split=[1], transform=test_transform)
        val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                pin_memory=pin_memory, drop_last=False)

        train_size, val_size = len(trainset), len(valset)
    else:
        train_split = [0, 1]
        trainset = CelebA(data_dir=path_to_data,
                          partition_file_path=path_to_partitions,
                          split=train_split, transform=train_transform)
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  pin_memory=pin_memory, drop_last=False)
        val_loader = None
        train_size, val_size = len(trainset), 0

    testset = CelebA(data_dir=path_to_data,
                     partition_file_path=path_to_partitions,
                     split=[2], transform=test_transform)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                             pin_memory=pin_memory, drop_last=False)

    if verbose:
        logger = logging.getLogger("base")
        logger.info("Dataset: CelebA")
        logger.info(f"Train size: {train_size:,}, val size: {val_size:,}, test size: {len(testset):,}.")

    # Note that test_transform must not contain non-deterministic transformation.
    pure_trainset = CelebA(data_dir=path_to_data, partition_file_path=path_to_partitions, split=train_split,
                           transform=test_transform)
    eval_dataloader = DataLoader(pure_trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                 pin_memory=pin_memory, drop_last=False)

    return train_loader, val_loader, test_loader, eval_dataloader


def get_imagenet_dataloaders(*, data_name: str, root: str, validate: bool = True, batch_size: int = 4,
                             num_workers: int = 1, train_transform: Compose = None, test_transform: Compose = None,
                             pin_memory: bool = False, verbose: bool = False) -> \
        Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """Data loader. Combines the ImageNet32 and ImageNet64 datasets and a sampler, and provides an iterable over
    the given dataset.

    Args:
        data_name: The dataset name.
        root: The path to the data.
        validate: Whether to create validation set or not.
        batch_size: How many samples per batch to load.
        num_workers: How many subprocesses to use for data loading.
        train_transform: Transformations to be applied on train dataset.
        test_transform: Transformations to be applied on validation and test datasets.
        pin_memory: If True, the data loader will copy Tensors into CUDA pinned memory before returning them.
        verbose: If True provides additional details about the dataset.

    Returns:
        Training, validation and test dataloaders.
    """
    if data_name == "imagenet32":
        set_start_method_for_darwin()
        path_to_data = os.path.join(root, "Imagenet32")
        dataset = ImageNet(path_to_data, split="train", transform=None, res=32)
    elif data_name == "imagenet64":
        set_start_method_for_darwin()
        path_to_data = os.path.join(root, "Imagenet64")
        dataset = ImageNet(path_to_data, split="train", transform=None, res=64)
    else:
        raise ValueError("Unknown dataset name.")

    if validate:
        indices, labels = np.arange(stop=len(dataset)), [tup[1] for tup in dataset]
        train_indies, val_indices = train_test_split(indices, test_size=0.20, stratify=labels)

        trainset = Subset(dataset, train_indies)
        trainset = MapDataset(trainset, train_transform)
        train_loader = DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
                                  drop_last=False)

        valset = Subset(dataset, val_indices)
        valset = MapDataset(valset, test_transform)
        val_loader = DataLoader(valset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
                                drop_last=False)

        train_size, val_size = len(trainset), len(valset)
    else:
        trainset = MapDataset(dataset, train_transform)
        train_loader = DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
                                  drop_last=False)
        val_loader = None
        train_size, val_size = len(dataset), 0

    if data_name == "imagenet32":
        testset = ImageNet(path_to_data, split="val", transform=test_transform, res=32)
        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                 pin_memory=pin_memory, drop_last=False)
    elif data_name == "imagenet64":
        testset = ImageNet(path_to_data, split="val", transform=test_transform, res=64)
        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                 pin_memory=pin_memory, drop_last=False)
    else:
        raise ValueError("Unknown data_name")

    if verbose:
        logger = logging.getLogger("base")
        logger.info(f"Dataset: {data_name}")
        logger.info(f"Train size: {train_size:,}, val size: {val_size:,}, test size: {len(testset):,}.")

    # Note that test_transform must not contain non-deterministic transformation.
    pure_trainset = MapDataset(dataset, test_transform)
    eval_dataloader = DataLoader(pure_trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                 pin_memory=pin_memory, drop_last=False)
    return train_loader, val_loader, test_loader, eval_dataloader


def read_dataset(*, root: str, name: str, validate: bool = True, batch_size: int = 4, num_workers=2,
                 train_transform: Compose = None, test_transform: Compose = None, digits: list = None,
                 pin_memory=True, verbose: bool = False):
    """Prepares the dataloaders of the specified dataset.

    Args:
        root: The path to the data.
        name: The dataset name.
        validate: Whether to create validation set or not.
        batch_size: How many samples per batch to load.
        num_workers: How many subprocesses to use for data loading.
        train_transform: Transformations to be applied on train data.
        test_transform: Transformations to be applied on validation and test data.
        digits: A list of digits to select. If None, all digits will be selected.
        pin_memory: If True, the data loader will copy Tensors into CUDA pinned memory before returning them.
        verbose: If True provides additional details about the dataset.

    Returns:
        Training, validation and test dataloaders.
    """
    if name == "celeba":
        return get_celeba_dataloaders(root=root, validate=validate, batch_size=batch_size, num_workers=num_workers,
                                      train_transform=train_transform, test_transform=test_transform,
                                      pin_memory=pin_memory, verbose=verbose)
    elif name in ("cifar10", "MNIST"):
        return get_cifar10_mnist_dataloaders(data_name=name, root=root, validate=validate, digits=digits,
                                             batch_size=batch_size, num_workers=num_workers,
                                             train_transform=train_transform, test_transform=test_transform,
                                             pin_memory=pin_memory, verbose=verbose)
    elif name in ("imagenet32", "imagenet64"):
        return get_imagenet_dataloaders(data_name=name, root=root, validate=validate, batch_size=batch_size,
                                        num_workers=num_workers, train_transform=train_transform,
                                        test_transform=test_transform, pin_memory=pin_memory, verbose=verbose)
    else:
        raise ValueError("Unknown dataset name.")
