from .dataset import read_dataset
from .utils import merge_generators, unpickle


# DATASET_SIZE stores partition sizes for each dataset supported which is used for FID/KID calculation.
# In Particular, the values determine the number of samples to generate from a model to calculate the metric against
# the corrresponding partition.
DATASET_SIZE = {
    "cifar10": {
        "train": 50000,
        "test": 10000,
    },
    "celeba": {
        "train": 20000,
        "test": 5000,
    },
    "imagenet32": {
        "train": 50000,
        "val": 10000
    }
}

__all__ = [read_dataset, merge_generators, unpickle, DATASET_SIZE]
