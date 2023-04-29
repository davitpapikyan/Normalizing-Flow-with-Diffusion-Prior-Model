import unittest

import torch

from diffusion_prior.latent_formaters import CatFormater


def check_equality_of_lists_of_tensors(list_a: list, list_b: list) -> bool:
    """Checks the equality of two lists of tensors.

    Args:
        list_a: The first list of tensors.
        list_b: The second list of tensors.

    Returns:
        True if the corresponding items of given lists are equal to each other.
    """
    if len(list_a) != len(list_b):
        return False
    for idx in range(len(list_a)):
        tensor_a, tensor_b = list_a[idx], list_b[idx]
        if (tensor_a.shape != tensor_b.shape) or (not torch.all(tensor_a == tensor_b).item()):
            return False
    return True


class TestValidityOfGlowFormaters(unittest.TestCase):
    """Testing validity of latent formaters of Glow for feeding to Diffusion prior.
    """
    def setUp(self):
        """Setting up testing objects.
        """
        self.glow_formaters = [CatFormater]
        self.in_channels, self.size = 3, 256

    def test_validity(self):
        """Postprocess transformation must be the inverse of forard processing.
        """
        for Formater in self.glow_formaters:
            for L in (2, 3, 4, 5, 6, 7):
                formater = Formater(L=L, in_channels=self.in_channels, size=self.size)

                latents = [torch.randn(size=(4, *dim)) for dim in formater.latent_dims]
                transformed_latents = formater.process_latents(latents)
                inverse_transformed_latents = formater.postprocess([transformed_latents])

                self.assertTrue(check_equality_of_lists_of_tensors(latents, inverse_transformed_latents),
                                msg=f"{formater.__class__.__name__} is not valid for L={L}, in_channels=\
                                {self.in_channels},size={self.size}.")


if __name__ == '__main__':
    unittest.main()
