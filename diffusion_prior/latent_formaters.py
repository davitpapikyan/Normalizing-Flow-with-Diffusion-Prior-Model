import copy
from abc import ABC, abstractmethod
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from normalizing_flow import calculate_output_shapes


class BaseFormater(nn.Module, ABC):
    """Base class for processing normalizing flow's latent variable.
    """
    def __init__(self, L: int, in_channels: int, size: int):
        """Initialization of the formater.

        Args:
            L: The number of blocks in Glow.
            in_channels: The number of channels of input.
            size: The input dimensions.
        """
        super(BaseFormater, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.latent_dims = np.array(calculate_output_shapes(L=L, in_channels=in_channels, size=size))
        self.mins, self.maxs = None, None

    @abstractmethod
    def process_latents(self, latents: list) -> list:
        """Processes parts of latent variable of normalizing flow into a suitable
        input to Diffusion prior.

        Args:
            latents: A list of parts of latent variable.

        Returns:
            A list of processed latent parts.
        """
        ...

    @abstractmethod
    def postprocess(self, latents: list) -> list:
        """Postprocesses parts of latent variable sampled from Diffusion model.
        It is the inverse transformation of process_latents method.

        Args:
            latents: A list of parts of latent variable sampled from Diffusion model.

        Returns:
            A list of postprocessed latent parts.
        """
        ...

    def get_num_latent_parts(self) -> int:
        """Calculates and return the number of latent parts outputed by the
        corresponding Normalizing Flow backbone.
        """
        return len(self.latent_dims)

    def standardize_latents(self, latents: list) -> list:
        """Standardizes all latent parts.
        Args:
            latents: A list of latent parts.
        Returns:
            Standardized latent parts in the same list.
        """
        # self.mins, self.maxs = [], []
        # for idx, latent in enumerate(latents):
        #     min_value, max_value = latent.min(), latent.max()
        #     latents[idx] = (latents[idx] - min_value) / (max_value - min_value + 1e-6)
        #     self.mins.append(min_value)
        #     self.maxs.append(max_value)
        return latents

    def inv_standardize_latents(self, latents: list) -> list:
        """Back-standardizes all latent parts.
        Args:
            latents: A list of latent parts.
        Returns:
            Back-standardized latent parts in the same list.
        """
        # for idx, latent in enumerate(latents):
        #     min_value, max_value = self.mins[idx], self.maxs[idx]
        #     latents[idx] = (latents[idx] * (max_value - min_value + 1e-6)) + min_value
        return latents


class IdentityFormater(BaseFormater):
    """Identity formater.
    """

    def __init__(self, L: int, in_channels: int, size: int):
        """Initialization of the formater.

        Args:
            L: The number of blocks in Glow.
            in_channels: The number of channels of input.
            size: The input dimensions.
        """
        super(IdentityFormater, self).__init__(L, in_channels, size)
        self.postprocessed_latent_shapes = self.latent_dims

    def process_latents(self, latents: list) -> list:
        """Identity processor.

        Args:
            latents: A list of parts of latent variable.

        Returns:
            The standardized input latents.
        """
        assert len(latents) == len(self.latent_dims), "IdentityFormater expects L latent tensors from Diffusion prior."
        # return latents
        return self.standardize_latents(latents)

    def postprocess(self, latents: list) -> list:
        """Identity postprocessor.

        Args:
            latents: A list of parts of latent variable sampled from Diffusion model.

        Returns:
            The back-standardized input latents.
        """
        # return latents
        return self.inv_standardize_latents(latents)

    def get_input_shapes(self) -> list:
        """Returns input shapes of latent parts outputed by Normalizing Flow backbone.
        """
        return self.postprocessed_latent_shapes


class CatFormater(BaseFormater):
    """Concat formater.
    """

    def __init__(self, L: int, in_channels: int, size: int):
        """Initialization of the formater.

        Args:
            L: The number of blocks in Glow.
            in_channels: The number of input channels.
            size: The input dimensions.
        """
        super(CatFormater, self).__init__(L, in_channels, size)
        self.unsqueeze = partial(rearrange, pattern="b (c c1 c2) h w -> b c (h c1) (w c2)", c1=2, c2=2)
        self.squeeze = partial(rearrange, pattern="b c (h h1) (w w1) -> b (c h1 w1) h w", h1=2, w1=2)

        dim = copy.deepcopy(self.latent_dims[(len(self.latent_dims) - 1) // 2])
        dim[0] *= 2
        self.postprocessed_latent_shapes = [dim]

        self.__data = {}

    def process_latents(self, latents: list) -> list:
        """Reshapes and concatenates the latent parts into a single tensor.

        Args:
            latents: A list of parts of latent variable.

        Returns:
            The input latents.
        """
        target_idx = (len(latents) - 1) // 2
        transformed_latents = []
        degrees = [target_idx - idx for idx in range(len(latents))]

        for pos, degree in enumerate(degrees):
            transformation = self.squeeze if degree > 0 else self.unsqueeze
            latent_tensor = latents[pos]
            for _ in range(abs(degree)):
                latent_tensor = transformation(latent_tensor)
            transformed_latents.append(latent_tensor)

        del latent_tensor
        self.__data["processed_shapes"] = torch.tensor([latent.shape[1:] for latent in transformed_latents])
        # return torch.cat(transformed_latents, dim=1)
        return self.standardize_latents([torch.cat(transformed_latents, dim=1)])

    def postprocess(self, latents: list) -> list:
        """Splits the latent (sampled from diffusion) into desirable shapes for Glow.
        Note that latents list must be of length 1.

        Args:
            latents: A list of parts of latent variable sampled from Diffusion model.

        Returns:
            The input latents.
        """
        assert len(latents) == 1, "CatFormater expects a single latent tensor from Diffusion prior."

        inverse_transformed_latents = []
        # transformed_latents = latents[0]
        transformed_latents = self.inv_standardize_latents(latents)[0]
        target_idx = (len(self.__data["processed_shapes"]) - 1) // 2
        target_channels = self.latent_dims[target_idx][0]

        self.latent_dims = np.array(self.latent_dims)
        factor = int(self.__data["processed_shapes"][:target_idx][:, 0].sum() /
                     self.__data["processed_shapes"][target_idx + 1:][:, 0].sum())
        n_chunks = (self.__data["processed_shapes"][:, 0].sum().item() - target_channels) // (factor + 1)

        to_unsqueeze = transformed_latents[:, :int(factor * n_chunks)]
        to_squeeze = transformed_latents[:, int(factor * n_chunks) + target_channels:]
        target_tensor = transformed_latents[:, int(factor * n_chunks):int(factor * n_chunks) + target_channels]
        del transformed_latents

        if to_unsqueeze.shape[1] != 0:
            remaining_part = to_unsqueeze
            for i in range(1, target_idx):
                remaining_part = self.unsqueeze(remaining_part)
                latent_tensor = remaining_part[:, - self.latent_dims[target_idx - i][0]:]
                remaining_part = remaining_part[:, :-self.latent_dims[target_idx - i][0]]
                inverse_transformed_latents.append(latent_tensor)

            inverse_transformed_latents.append(self.unsqueeze(remaining_part))

        del remaining_part
        inverse_transformed_latents = list(reversed(inverse_transformed_latents))
        inverse_transformed_latents.append(target_tensor)
        del target_tensor

        remaining_part = to_squeeze
        for i in range(1, len(self.__data["processed_shapes"]) - target_idx - 1):
            remaining_part = self.squeeze(remaining_part)
            latent_tensor = remaining_part[:, :-self.latent_dims[target_idx + i][0]]
            remaining_part = remaining_part[:, -self.latent_dims[target_idx + i][0]:]
            inverse_transformed_latents.append(latent_tensor)

        inverse_transformed_latents.append(self.squeeze(remaining_part))
        return inverse_transformed_latents

    def get_num_latent_parts(self):
        """Calculates and return the number of latent parts outputed by the
        corresponding Normalizing Flow backbone.
        """
        return 1

    def get_input_shapes(self) -> list:
        """Returns input shapes of latent parts outputed by Normalizing Flow backbone.
        """
        return self.postprocessed_latent_shapes


def get_formater(name: str):
    """Returns a formater class corresponding to the name.

    Args:
        name: The name of a formater. Currently supported names are
            ['IdentityFormater', 'CatFormater'].

    Returns:
        A latent formater class.
    """
    if name == "IdentityFormater":
        return IdentityFormater
    elif name == "CatFormater":
        return CatFormater
    else:
        raise ValueError("Invalid formater name")


__all__ = [get_formater]
