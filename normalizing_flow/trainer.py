import logging
import os
from datetime import datetime
from typing import List

import numpy as np
import torch
from aim import Run, Distribution, Text
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import read_dataset, DATASET_SIZE
from metrics.compute import evaluate_model
from utils import set_mode
from .prior import save_model
from .utils import init_optimizer, track_images, save_images, preprocess_batch, calculate_loss, \
    get_data_transforms, calculate_output_shapes, initialize_with_zeros, data_dependent_nf_initialization, \
    postprocess_batch


@torch.no_grad()
def calculate_bpd(flow, prior_dist, *, data_loader: DataLoader, n_bits: int, n_bins: int, n_pixel: int,
                  bpd_const: float, device: torch.device):
    """Calculating BPD of normalizing flow with importance sampling.

    Args:
        flow: Normalizing flow model.
        prior_dist: The prior distribution.
        data_loader: The data loader.
        n_bits: The number of bits to encode.
        n_bins: The number of bins.
        n_pixel: The number of pixels.
        bpd_const: Constant to compute BPD.
        device: Device.

    Returns:
        Bits per dimension.
    """
    set_mode(flow, mode="eval")
    bpds = []

    for _, data in enumerate(tqdm(data_loader, desc="Calculating bpd")):
        batch = data[0].to(device) if isinstance(data, list) else data.to(device)
        batch = preprocess_batch(batch, n_bits, n_bins)

        log_likelihood, logp = initialize_with_zeros(2, batch.size(0), device)
        latents, log_likelihood, logp = flow.transform(batch + torch.rand_like(batch) / n_bins, log_likelihood, logp)
        logp += prior_dist.compute_log_prob(latents[-1])
        log_likelihood = log_likelihood + logp

        bpd_val = ((np.log(n_bins) * n_pixel - log_likelihood) * bpd_const).mean(dim=0)
        bpds.append(bpd_val)

    set_mode(flow, mode="train")
    return torch.stack(bpds).mean()


def train(flow, prior_dist, *, logger: logging.Logger, experiment_name: str, exp_output_dir: str, data_root: str,
          data_name: str, transformations: List[str], batch_size: int, num_workers: int, optim_name: str, lr: float,
          n_epochs: int, print_freq: int, save_checkpoint_freq: int, log_param_distribution: bool,
          log_gen_images_per_iter: int, device: torch.device, checkpoint_dir: str, result_dir: str,
          resume_info: dict, img_size: int = 32, n_bits: int = 5, temperature: float = 1.0, digits: list = None,
          fid_kwargs=None, kid_kwargs=None, ssim_psnr_kwargs=None):
    """Trains the normalizing flow model with Gaussian prior, runs validation and test steps.

    Args:
        flow: Normalizing flow model.
        prior_dist: The prior distribution.
        logger: The logger.
        experiment_name: The name of the experiment to run.
        exp_output_dir: The output directory of the experiment.
        data_root: The directory of datasets stored.
        data_name: Which data to load.
        transformations: A list of transformations.
        batch_size: How many samples per batch to load.
        num_workers: How many subprocesses to use for data loading.
        optim_name: The optimizer name.
        lr: The learning rate.
        n_epochs: The number of epochs to train.
        print_freq: The number of iterations per to log the training results.
        save_checkpoint_freq: The number of epochs per to save the normalizing flow model.
        log_param_distribution: Either to log model parameters' densities or not.
        log_gen_images_per_iter: Per how many logging iterations to log generated images.
        device: Device.
        checkpoint_dir: The directory for saving the normalizing flow model.
        result_dir: The directroy for storing generated samples.
        resume_info: A dict containing info about from which directory to load the saved model and from which epoch.
        img_size: The image size to resize.
        n_bits: Number of bits for BPD computation.
        temperature: The temperature parameter.
        digits: A list of digits to select. If None, all digits will be selected.
        fid_kwargs: A list of kwargs used to evaluate FID.
        kid_kwargs: A list of kwargs used to evaluate KID.
        ssim_psnr_kwargs: A dict of kwargs used to evaluate SSIM and PSNR.
    """
    # Preparing data.
    train_transform, test_transform = get_data_transforms(data_name, img_size, transformations)
    train_loader, _, test_loader, dataloader = read_dataset(root=data_root, name=data_name, validate=False,
                                                            batch_size=batch_size, num_workers=num_workers,
                                                            train_transform=train_transform,
                                                            test_transform=test_transform, digits=digits,
                                                            pin_memory=False, verbose=True)
    logger.info("Training, validation and test dataloaders are successfully loaded.")

    # Initializing SSIM dataloader to compute the metric against.
    if ssim_psnr_kwargs:
        ssim_psnr_kwargs["dataloader"] = dataloader if ssim_psnr_kwargs else None

    optimizer = init_optimizer(optim_name, flow.parameters(), lr=lr)

    if resume_info:  # If resume training, then load model and optimizer states.
        checkpoint = torch.load(os.path.join(checkpoint_dir, f"model_{str(resume_info['epoch']).zfill(3)}.pt"))
        flow.load_state_dict(checkpoint["flow"])
        prior_dist.load_state_dict(checkpoint["prior_dist"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch, current_iter = resume_info["epoch"], checkpoint["current_iter"]

        # Making sure that learning rate is up-to-date.
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        logger.info("Glow model is loaded.")
    else:
        current_iter, start_epoch = 0, 0

    # Setting Aim logger (make sure run_hash doesn't contain double underscores).
    aim_logger = Run(repo='../aim/', experiment=experiment_name)
    aim_logger.name = exp_output_dir
    if not resume_info:
        aim_logger["hparams"] = {"dataset": data_name, "batch_size": batch_size, "lr": lr, "L": flow.L, "K": flow.K}

    # Precomputing constants used for BPD computation from log-likelihood.
    n_bins = 2.0 ** n_bits
    n_pixel = img_size * img_size * 3.0
    bpd_const = np.log2(np.e) / n_pixel

    # Initialize flow based on data.
    data_dependent_nf_initialization(flow, train_loader, device, n_bits, n_bins)
    logger.info("Data-driven initialization of NF is completed.")

    running_loss = 0.0
    latent_dimensions = calculate_output_shapes(L=flow.L, in_channels=flow.in_channel, size=img_size)
    logger.info("Starting the training.\n")
    set_mode(flow, mode="train")

    # [Training]
    for epoch in range(start_epoch + 1, start_epoch + n_epochs + 1):

        start_epoch_time = datetime.now()
        for iteration, data in enumerate(train_loader):
            batch = data[0].to(device) if isinstance(data, list) else data.to(device)
            batch = preprocess_batch(batch, n_bits, n_bins)

            log_likelihood, logp = initialize_with_zeros(2, batch.size(0), device)
            latents, log_likelihood, logp = flow.transform(batch + torch.rand_like(batch)/n_bins, log_likelihood, logp)
            logp += prior_dist.compute_log_prob(latents[-1])
            log_likelihood = log_likelihood + logp
            # See the discussion on why noise is being added in [Section 3.1, A note on the evaluation of generative
            # models] paper (https://arxiv.org/abs/1511.01844).

            loss = calculate_loss(log_likelihood, n_bins, n_pixel)  # Bits per dimension.

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(flow.parameters(), 1)
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 1)
            optimizer.step()

            running_loss += loss.item()

            if iteration % print_freq == print_freq - 1:  # Print every print_freq minibatches.
                running_loss /= print_freq

                logger.info(f"Epoch: {epoch:5}  |  Iteration: {(iteration + 1):8}  |  bpd: {running_loss:.3f}")
                aim_logger.track(running_loss, name="bpd", step=current_iter, epoch=epoch, context={"subset": "train"})
                current_iter += print_freq

                # After every log_gen_images_per_iter train logs generate and visualize images on Aim.
                if ((iteration + 1) // print_freq) % log_gen_images_per_iter == 0:

                    if log_param_distribution:
                        # Tracking param distributions.
                        for name, param in flow.named_parameters():
                            dist = Distribution(distribution=param.clone().cpu().data.numpy())
                            aim_logger.track(dist, name=name, step=current_iter // print_freq)

                    if epoch % 5 == 0:  # TODO: Remove me
                        set_mode(flow, mode="eval")
                        last_latent = prior_dist.sample(shape=(4, *latent_dimensions[-1]), temperature=temperature)
                        generated_images = flow.sample([last_latent], temperature=temperature)
                        track_images(aim_logger, generated_images, step=current_iter // print_freq)
                        set_mode(flow, mode="train")

                running_loss = 0.0

        # An epoch of training is completed.
        # Saving the model.
        if epoch % save_checkpoint_freq == 0:
            log_text = f"Epoch: {epoch:5}  |  Saving"

            if data_name != "MNIST":
                metrics = evaluate_model(base=flow, prior=prior_dist, prior_type="gaussian",
                                         postprocess_func=lambda img: postprocess_batch(img, n_bins),
                                         data_name=data_name, dataset_res=img_size, batch_size=batch_size, num_gen=15,
                                         dataset_split="train", device=device, fid_kwargs=fid_kwargs,
                                         kid_kwargs=kid_kwargs, temperature=temperature,
                                         last_latent_dim=latent_dimensions[-1])

                for metric, value in metrics.items():
                    aim_logger.track(value, name=metric, epoch=epoch, context={"subset": "train_checkpoints"})
                    log_text += f"  |  {metric}: {value:.3f}"

            logger.info(log_text)
            save_model(logger, flow, prior_dist, optimizer, epoch, current_iter, checkpoint_dir, text=log_text)

            # Generate and save images after each epoch.
            set_mode(flow, mode="eval")
            last_latent = prior_dist.sample(shape=(64, *latent_dimensions[-1]), temperature=temperature)
            generated_images = flow.sample([last_latent], temperature=temperature)
            save_images(generated_images.float(), path=result_dir, name=f"generated_{epoch}")
            set_mode(flow, mode="train")

        running_loss = 0.0

        # Logging datetime information duration.
        logger.info("-" * 70)
        if epoch != start_epoch + n_epochs:  # If not the last epoch.
            end_epoch_time = datetime.now()
            duration = end_epoch_time - start_epoch_time
            logger.info(f"Duration of epoch: {duration}")
            estimated_finish = datetime.now() + duration * (n_epochs - epoch)
            logger.info(f"Estimated end of training: {estimated_finish}")
            logger.info(f"Time remaining: {estimated_finish - datetime.now()}\n")

    # Making sure to save the model parameters after the last epoch.
    if epoch % save_checkpoint_freq != 0:
        save_model(logger, flow, prior_dist, optimizer, epoch, current_iter, checkpoint_dir)

    # [Testing]
    logger.info("Starting evaluation.")
    log_text = f"Testing  "

    # BPD calculation.
    test_bpd = calculate_bpd(flow, prior_dist, data_loader=test_loader, n_bits=n_bits, n_bins=n_bins, n_pixel=n_pixel,
                             bpd_const=bpd_const, device=device)
    train_bpd = calculate_bpd(flow, prior_dist, data_loader=dataloader, n_bits=n_bits, n_bins=n_bins, n_pixel=n_pixel,
                              bpd_const=bpd_const, device=device)
    aim_logger.track(train_bpd, name="bpd", context={"subset": "test"})
    aim_logger.track(test_bpd, name="bpd", context={"subset": "test"})
    log_text += f"  |  train_bpd: {train_bpd:.3f}  |  test_bpd: {test_bpd:.3f}"

    # FID, KID calculation.
    if data_name != "MNIST":
        final_metrics = evaluate_model(base=flow, prior=prior_dist, prior_type="gaussian",
                                       postprocess_func=lambda img: postprocess_batch(img, n_bins),
                                       data_name=data_name, dataset_res=img_size, batch_size=batch_size,
                                       num_gen=DATASET_SIZE[data_name]["train"], dataset_split="train", device=device,
                                       fid_kwargs=fid_kwargs, kid_kwargs=kid_kwargs, ssim_psnr_kwargs=ssim_psnr_kwargs,
                                       temperature=temperature, last_latent_dim=latent_dimensions[-1])

        for metric, value in final_metrics.items():
            aim_logger.track(value, name=metric, epoch=epoch, context={"subset": "final_metrics"})
            log_text += f"  |  {metric}: {value:.3f}"

    logger.info(log_text)
    aim_logger.track(Text(log_text), name="NF final stats")

    aim_logger.close()
    logger.info("Experiment is finished.")
