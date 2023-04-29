from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from utils import set_mode
from .glow import StepFlow, GlowBlock, Glow
from .prior import IsotropicGaussian, GaussianPrior, save_model
from .trainer import train, calculate_bpd
from .transforms import InvConv2d, ActNorm, AffineCoupling, Squeeze, Split
from .utils import get_data_transforms, init_optimizer, preprocess_batch, postprocess_batch, initialize_with_zeros, \
    calculate_output_shapes, track_images, save_images, data_dependent_nf_initialization


class NFBackbone(nn.Module):
    """Wrapper class of Normalizing Flow model designed to be used with Diffusion prior.

    Attributes:
        model: Normalizing Flow model.
        device: Device.
        freeze_flow: Whether model is frozen or not.
    """
    def __init__(self, model_dir: str, in_channel: int, L: int, K: int, learn_prior_mean_logs: bool, freeze_flow: bool):
        """Initialiation.

        Args:
            in_channel: The number of input channels.
            L: The number of Glow blocks.
            K: The number of flows in the Glow block.
            learn_prior_mean_logs: Whether to learn mean and covariance of Gaussian prior.
            model_dir: The path to .pt file containing model state dict. If the training is intended to start from
                scratch, pass None.
            freeze_flow: If True then freeze normalizing flow parameters.
        """
        super(NFBackbone, self).__init__()
        self.L, self.K, self.learn_prior_mean_logs = L, K, learn_prior_mean_logs
        self.model = Glow(in_channel=in_channel, L=L, K=K, learn_prior_mean_logs=learn_prior_mean_logs)
        self.device = self.model.device
        self.model.to(self.device)
        self.freeze_flow = freeze_flow

        if model_dir:
            checkpoint = torch.load(model_dir, map_location=torch.device('cpu'))  # TODO: Remove map_location
            self.model.load_state_dict(checkpoint["flow"])

        for param in self.model.parameters():
            param.requires_grad = not self.freeze_flow

    def is_frozen(self):
        """Checks if a model is frozen.

        Returns:
            True if model is frozen otherwise False.
        """
        return self.freeze_flow

    def set_train_mode(self):
        """Setting model to traning mode based on whether it is frozen or not.
        The model can be set to training mode only if it is not frozen.
        """
        if not self.is_frozen():
            self.train()
        else:
            self.eval()

    def set_eval_mode(self):
        self.eval()

    def transform(self, x: Tensor, log_det_jac: Tensor) -> Tuple[list, Tensor]:
        """Computes forward transformation.

        Args:
            x: Input tensor.
            log_det_jac: Ongoing log abs determinant of jacobian.

        Returns:
            parts_of_latent_variable: A list of parts of latent variable. The number of parts is equal to L in Glow.
            log_det_jac: Log abs determinant of jacobian matrix of the transformation.
        """
        parts_of_latent_variable, log_det_jac, _ = self.model.transform(x, log_det_jac, None)
        return parts_of_latent_variable, log_det_jac

    def invert(self, latents: list) -> Tensor:
        """Computes inverse transformation.

        Args:
            latents: A list of parsts of latent variable. The number of parts is equal to L in Glow.

        Returns:
            Inverse transformed input.
        """
        return self.model.invert(latents)

    @torch.no_grad()
    def sample(self, latents: list, postprocess_func=None) -> Tensor:
        """Samples from Normalizing Flow model.

        Args:
            latents: A list of parts of latent variable.
            postprocess_func: Postprocessor function to generate final output.

        Returns:
            A Tensor representing sampled data points.
        """
        return self.model.sample(latents, postprocess_func)


__all_ = [InvConv2d, ActNorm, AffineCoupling, StepFlow, Squeeze, Split, GlowBlock, Glow, get_data_transforms, train,
          calculate_bpd, GaussianPrior, init_optimizer, set_mode, preprocess_batch, postprocess_batch,
          calculate_output_shapes, initialize_with_zeros, NFBackbone, data_dependent_nf_initialization, save_model]
