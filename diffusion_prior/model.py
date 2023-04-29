import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from .gaussian_diffusion import Unet, GaussianDiffusion


class DiffusionModel(nn.Module):
    """Initializes a single Diffusion model.
    """

    def __init__(self, unet_kwargs, diffusion_kwargs, image_size, channels):
        """Initialization.

        Arguments:
            unet_kwargs: A dict with the following keys
                dim: The starting channels. The following channels multiples of dim defined by dim_mults.
                dim_mults: A tuple specifying the scaling factors of channels.
                resnet_block_groups: The number of residual blocks.
                learned_sinusoidal_cond: Either to use random sinusoidal positional embeddings or not.
                random_fourier_features: Either to learn positional embeddings or not.
                learned_sinusoidal_dim: The dimension of positional embeddings.
            diffusion_kwargs: A dict with the following keys
                timesteps: The number of diffusion timesteps.
                sampling_timesteps: The number of inference timesteps.
                loss_type: One of 'l1' and 'l2'.
                beta_schedule: One of 'cosine' and 'linear'.
                ddim_sampling_eta: Controls sampling stochasticity. When sampling_timesteps is less that training
                    timesteps, small eta results in better sample quality.
            image_size: Input image size.
            channels: Input image channels.
        """
        super(DiffusionModel, self).__init__()
        network = Unet(**unet_kwargs, channels=channels)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__diffusion = GaussianDiffusion(network, **diffusion_kwargs, image_size=image_size,
                                             auto_normalize=False).to(self.device)
        self.loss_type = self.__diffusion.loss_type

    def forward(self, x):
        loss = self.__diffusion(x)
        return loss

    def sample_latent(self, n_samples: int, return_all_timesteps: bool = False):
        """Samples a part of latent variable from the Diffusion model.

        Args:
            n_samples: The number of samples.
            return_all_timesteps:

        Returns:
            A sampled tensor.
        """
        return self.__diffusion.sample(batch_size=n_samples, return_all_timesteps=return_all_timesteps)

    def sample_latent_given_start(self, start):
        timesteps = torch.ones(start.size(0), dtype=torch.int64, device=start.device) \
                    * (self.__diffusion.num_timesteps-1)
        x_t = self.__diffusion.q_sample(start, timesteps, noise=None)
        # Starting sampling from x_t.
        x_start = None

        for t in tqdm(reversed(range(0, self.__diffusion.num_timesteps)), desc='sampling loop time step',
                      total=self.__diffusion.num_timesteps):
            self_cond = x_start if self.__diffusion.self_condition else None
            x_t, x_start = self.__diffusion.p_sample(x_t, t, self_cond)

        ret = self.__diffusion.unnormalize(x_t)
        return ret

    def interpolate_latent(self, latent_1, latent_2, lam):
        return self.__diffusion.interpolate(latent_1, latent_2, t=None, lam=lam)

    def calc_neg_log_likelihood_loop(self, x):
        return self.__diffusion.calc_neg_log_likelihood_loop(x)


class DiffusionPrior(nn.Module):
    """Models Normalizing Flow prior with Diffusion model.

    Attributes:
        device: Device.
        latent_formater: The latent formater.
    """

    def __init__(self, *, latent_formater, unet_kwargs, diffusion_kwargs, image_sizes, latent_channels):
        """Initialization.

        Args:
            unet_kwargs: A dict with keyword arguments of Unet.
            diffusion_kwargs: A dict with keyword arguments of diffusion process.
            image_sizes: Latent dimensions.
            latent_channels: A list of channels of latent parts outputed by Normalizing Flow backbone.
        """
        super(DiffusionPrior, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.latent_formater = latent_formater

        self.__priors = [DiffusionModel(unet_kwargs, diffusion_kwargs, image_size=image_sizes[i],
                                        channels=latent_channels[i])
                         for i in range(self.latent_formater.get_num_latent_parts())]
        self.loss_type = self.__priors[0].loss_type

    def forward(self, latents: list) -> list:
        """Computes loss of Diffusion model on latents.

        Args:
            latents: A list of parts of latent variable.

        Returns:
            A list of losses of the Diffusion model.
        """
        losses = []
        processed_latents = self.latent_formater.process_latents(latents)
        for i in range(0, len(processed_latents)):
            prior, latent = self.__priors[i], processed_latents[i]
            losses.append(prior(latent))
        return losses

    def sample_latents(self, n_samples: int, return_all_timesteps: bool = False) -> list:
        """Samples parts of latent variable of Normalizing Flow.

        Args:
            n_samples: The number of samples.
            return_all_timesteps:

        Returns:
            A list of parts of latent variable.
        """
        if not return_all_timesteps:
            return self.latent_formater.postprocess([prior.sample_latent(n_samples, False) for prior in self.__priors])
        else:
            return [prior.sample_latent(n_samples, True) for prior in self.__priors]

    @torch.no_grad()
    def sample_latents_given_start(self, latents: list) -> list:
        return [prior.sample_latent_given_start(latents[idx]) for idx, prior in enumerate(self.__priors)]

    def evaluate_neg_log_likelihood(self, latents: list) -> list:
        neg_lls = []
        processed_latents = self.latent_formater.process_latents(latents)
        for i in range(0, len(processed_latents)):
            prior, latent = self.__priors[i], processed_latents[i]
            neg_ll = prior.calc_neg_log_likelihood_loop(latent) / np.prod(latent.shape[1:])
            neg_lls.append(neg_ll)
        return neg_lls

    def interpolate_latents(self, latents1, latents2, lam):
        return [prior.interpolate_latent(latents1[idx], latents2[idx], lam) for idx, prior in enumerate(self.__priors)]

    def parameters(self, only_trainable=False):
        """A generator that yields paramters of the model.
        """
        for prior in self.__priors:
            for param in prior.parameters():
                if only_trainable and not param.requires_grad:
                    continue
                yield param
