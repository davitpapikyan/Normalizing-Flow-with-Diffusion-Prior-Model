import zipfile

import numpy as np
import torch
import torchvision
from PIL import Image
from cleanfid.resize import build_resizer
from torch import Tensor


def discretize(img):
    """The opposite transformation of torchvision.transforms.ToTensor().
    Usually appled before Dequantization as torchvision transformation.

    Args:
        img: Input image.

    Returns:
        Discretized tensor image.
    """
    return (img * 255).to(torch.uint8)


class Storage:
    """Storage for sampled images.
    Useful for FID/KID calculation to reuse the sampled images for other metrics too."""

    def __init__(self, data: Tensor = None, ready: bool = False, index: int = 0):
        """Initializes the storage.

        Args:
            data: The actual storage variable.
            ready: Is the storage ready to be used.
            index: The current index from which to start the usage of stored data.
        """
        self.data, self.ready, self.index = data, ready, index

    def reset(self):
        """Removes storage.
        """
        self.data, self.ready, self.index = None, False, 0

    def set_ready_for_usage(self):
        """Marks storage as ready to be used.
        """
        self.ready, self.index = True, 0

    def append_gen_images(self, new_samples: Tensor):
        """Appends new sampled images to storage.

        Args:
            new_samples: Newly sampled images.
        """
        self.data = torch.cat([self.data, new_samples], dim=0) if self.data is not None else new_samples.clone()

    def iterative_reuse_of_data(self, batch_size: int) -> Tensor:
        """Calculates the next chunk of data from storage of size batch_size.

        Args:
            batch_size: The number of samples to return.

        Returns:
            The next chunk of data from storage of size batch_size.
        """
        end = self.index + batch_size
        sampled_data = self.data[self.index: end]
        self.index = end
        return sampled_data


class ResizeDatasetNumPy(torch.utils.data.Dataset):
    """
    A placeholder Dataset that enables parallelizing the resize operation
    using multiple CPU cores. Assumes that dataset is provided as a NumPy array.

    files: list of all files in the folder
    fn_resize: function that takes a np_array as input [0,255]
    """

    def __init__(self, files: np.ndarray, mode, size=(299, 299), fdir=None):
        self.files = files
        self.fdir = fdir
        self.transforms = torchvision.transforms.ToTensor()
        self.size = size
        self.fn_resize = build_resizer(mode)
        self.custom_image_tranform = lambda x: x
        self._zipfile = None

    def _get_zipfile(self):
        assert self.fdir is not None and '.zip' in self.fdir
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self.fdir)
        return self._zipfile

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img_pil = Image.fromarray(self.files[i], "RGB")
        img_np = np.array(img_pil)

        # apply a custom image transform before resizing the image to 299x299
        img_np = self.custom_image_tranform(img_np)
        # fn_resize expects a np array and returns a np array
        img_resized = self.fn_resize(img_np)

        # ToTensor() converts to [0,1] only if input in uint8
        if img_resized.dtype == "uint8":
            img_t = self.transforms(np.array(img_resized))*255
        elif img_resized.dtype == "float32":
            img_t = self.transforms(img_resized)

        return img_t
