import logging
import math
from datetime import datetime
from typing import List

import torch
from aim import Run, Distribution, Text
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import read_dataset, DATASET_SIZE
from metrics.compute import evaluate_model
from normalizing_flow import init_optimizer, get_data_transforms, preprocess_batch, postprocess_batch, \
    track_images, save_images, save_model, initialize_with_zeros


@torch.no_grad()
def calculate_bpd_with_diff_prior(nf, diff_prior, *, data_loader: DataLoader, n_bits: int, n_bins: int,
                                  device: torch.device):
    """Calculating BPD of normalizing flow with importance sampling.

    Args:
        nf: Normalizing flow model.
        diff_prior: The prior distribution.
        data_loader: The data loader.
        n_bits: The number of bits to encode.
        n_bins: The number of bins.
        device: Device.

    Returns:
        Bits per dimension.
    """
    nf.set_eval_mode()
    diff_prior.eval()
    bpd = 0.0

    for _, data in enumerate(tqdm(data_loader, desc="Calculating bpd")):
        batch = data[0].to(device) if isinstance(data, list) else data.to(device)
        batch = preprocess_batch(batch, n_bits, n_bins)

        log_likelihood = initialize_with_zeros(1, batch.size(0), device)
        latents, log_likelihood = nf.transform(batch + torch.rand_like(batch) / n_bins, log_likelihood)
        diff_nll = sum(diff_prior.evaluate_neg_log_likelihood(latents)) - log_likelihood

        bpd = (diff_nll / math.log(2.0)).mean(dim=0)

    bpd /= len(data_loader)
    nf.set_train_mode()
    diff_prior.train()
    return bpd


def train(nf_backbone, diffusion_prior, *, logger: logging.Logger, experiment_name: str, exp_output_dir: str,
          device: torch.device, data_name: str, transformations: List[str], data_root: str, batch_size: int,
          num_workers: int, digits: list, n_bits: int, img_size: int, checkpoint_dir: str,
          is_frozen: bool, optim_name: str, result_dir: str, lr_nf_backbone: float, lr_diffusion: float, n_epochs: int,
          print_freq: int, save_checkpoint_freq: int, log_param_distribution: bool, log_gen_images_per_iter: int,
          fid_kwargs=None, kid_kwargs=None, ssim_psnr_kwargs=None):
    """Trains the normalizing flow model with Diffusion prior.

    Args:
        nf_backbone: Normalizing flow model.
        diffusion_prior: The diffusion prior.
        logger: The logger.
        experiment_name: The name of the experiment to run.
        exp_output_dir: The output directory of the experiment.
        device: Device.
        data_name: Which data to load.
        transformations: A list of transformations.
        data_root: The directory of datasets stored.
        batch_size: How many samples per batch to load.
        num_workers: How many subprocesses to use for data loading.
        digits: A list of digits to select. If None, all digits will be selected.
        n_bits: Number of bits for BPD computation.
        img_size: The image size to resize.
        checkpoint_dir: The directory for saving the normalizing flow model.
        is_frozen:
        result_dir: The directroy for storing generated samples.
        optim_name: The optimizer name.
        lr_nf_backbone: Normalizing flow learning rate which is used when the backbone is not frozen.
        lr_diffusion: Diffusion prior learning rate.
        n_epochs: The number of epochs to train.
        print_freq: The number of iterations per to log the training results.
        save_checkpoint_freq: The number of epochs per to save the normalizing flow model.
        log_param_distribution: Either to log model parameters' densities or not.
        log_gen_images_per_iter: Per how many logging iterations to log generated images.
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
    logger.info("Training and test dataloaders are successfully loaded.")

    # Initializing SSIM dataloader to compute the metric against.
    if ssim_psnr_kwargs:
        ssim_psnr_kwargs["dataloader"] = dataloader if ssim_psnr_kwargs else None

    # Defining optimizer with separate learning rates for diffusion prior and NF backbone.
    params = [{"params": diffusion_prior.parameters(), "lr": lr_diffusion}]
    if not is_frozen:
        params.append({"params": nf_backbone.parameters(), "lr": lr_nf_backbone})
    optimizer = init_optimizer(optim_name, params, lr=lr_diffusion)
    logger.info("Optimizer is initialized.")

    loss_type = diffusion_prior.loss_type
    if not is_frozen:
        loss_type = f"{loss_type}_plus_bpd"

    aim_logger = Run(repo='../aim/', experiment=experiment_name)
    aim_logger.name = exp_output_dir
    aim_logger["hparams"] = {"dataset": data_name, "batch_size": batch_size, "is_nf_frozen": is_frozen,
                             "lr_diffusion": lr_diffusion, "L": nf_backbone.L, "K": nf_backbone.K,
                             "lr_nf_backbone": lr_nf_backbone if not is_frozen else None}

    n_bins = 2.0 ** n_bits  # n_bins is used for image processing before feeding to NF.
    n_pixel = img_size * img_size * 3.0

    current_iter, start_epoch, running_loss = 0, 0, 0.0
    logger.info("Starting the training.\n")

    nf_backbone.set_train_mode()
    diffusion_prior.train()

    # [Training]
    for epoch in range(start_epoch+1, start_epoch+n_epochs+1):

        start_epoch_time = datetime.now()
        for iteration, data in enumerate(train_loader):
            batch = data[0].to(device) if isinstance(data, list) else data.to(device)
            batch = preprocess_batch(batch, n_bits, n_bins)

            log_likelihood = initialize_with_zeros(1, batch.size(0), device)
            latents, log_likelihood = nf_backbone.transform(batch + torch.rand_like(batch)/n_bins, log_likelihood)
            losses = diffusion_prior(latents)
            loss = sum(losses)

            if not is_frozen:
                # 0.5 is a weighting factor.
                nf_bpd_loss = 0.5 * (-log_likelihood / (math.log(2.0) * n_pixel)).mean(dim=0)
                loss += nf_bpd_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if iteration % print_freq == print_freq-1:  # Print every print_freq minibatches.
                running_loss /= print_freq

                logger.info(
                    f"Epoch: {epoch:5}  |  Iteration: {(iteration+1):8}  |  {loss_type}: {running_loss:.3f}")
                aim_logger.track(running_loss, name=f"{loss_type}", step=current_iter, epoch=epoch,
                                 context={"subset": "train"})
                current_iter += print_freq

                # After every log_gen_images_per_iter train logs generate and visualize images on Aim.
                if ((iteration+1) // print_freq) % log_gen_images_per_iter == 0:

                    if log_param_distribution:
                        # Tracking param distributions.
                        if not is_frozen:
                            for name, param in nf_backbone.named_parameters():
                                dist = Distribution(distribution=param.clone().cpu().data.numpy())
                                aim_logger.track(dist, name=name, step=current_iter // print_freq)

                        for name, param in diffusion_prior.named_parameters():
                            dist = Distribution(distribution=param.clone().cpu().data.numpy())
                            aim_logger.track(dist, name=name, step=current_iter // print_freq)

                    if epoch % 5 == 0:  # TODO: Remove me
                        nf_backbone.set_eval_mode()
                        diffusion_prior.eval()

                        latents = diffusion_prior.sample_latents(n_samples=4)
                        generated_images = nf_backbone.sample(latents)
                        track_images(aim_logger, generated_images, step=current_iter // print_freq)

                        nf_backbone.set_train_mode()
                        diffusion_prior.train()

                running_loss = 0.0

        # An epoch of training is completed.
        # Saving the model.
        if epoch % save_checkpoint_freq == 0:
            log_text = f"Epoch: {epoch:5}  |  Saving"

            if data_name != "MNIST":
                metrics = evaluate_model(base=nf_backbone, prior=diffusion_prior, prior_type="diffusion",
                                         postprocess_func=lambda img: postprocess_batch(img, n_bins),
                                         data_name=data_name, dataset_res=img_size, batch_size=batch_size, num_gen=2000,
                                         dataset_split="train", device=device, fid_kwargs=fid_kwargs,
                                         kid_kwargs=kid_kwargs)

                for metric, value in metrics.items():
                    aim_logger.track(value, name=metric, epoch=epoch, context={"subset": "train_checkpoints"})
                    log_text += f"  |  {metric}: {value:.3f}"

            # Generate and save images after each epoch.
            nf_backbone.set_eval_mode()
            diffusion_prior.eval()

            latents = diffusion_prior.sample_latents(n_samples=64)
            generated_images = nf_backbone.sample(latents)
            save_images(generated_images.float(), path=result_dir, name=f"generated_{epoch}")

            nf_backbone.set_train_mode()
            diffusion_prior.train()

            logger.info(log_text)
            save_model(logger, nf_backbone, diffusion_prior, optimizer, epoch, current_iter, checkpoint_dir,
                       text=log_text)

        # Logging datetime information duration.
        logger.info("-"*70)
        if epoch != start_epoch+n_epochs:  # If not the last epoch.
            end_epoch_time = datetime.now()
            duration = end_epoch_time - start_epoch_time
            logger.info(f"Duration of epoch: {duration}")
            estimated_finish = datetime.now() + duration * (n_epochs-epoch)
            logger.info(f"Estimated end of training: {estimated_finish}")
            logger.info(f"Time remaining: {estimated_finish-datetime.now()}\n")

        running_loss = 0.0

    # Making sure to save the model parameters after the last epoch.
    if epoch % save_checkpoint_freq != 0:
        save_model(logger, nf_backbone, diffusion_prior, optimizer, epoch, current_iter, checkpoint_dir)

    # [Testing]
    logger.info("Starting evaluation.")
    log_text = f"Final evaluation"

    # BPD calculation.
    # test_bpd = calculate_bpd_with_diff_prior(nf_backbone, diffusion_prior, data_loader=test_loader, n_bits=n_bits,
    #                                          n_bins=n_bins)
    # train_bpd = calculate_bpd_with_diff_prior(nf_backbone, diffusion_prior, data_loader=dataloader, n_bits=n_bits,
    #                                           n_bins=n_bins)
    # aim_logger.track(train_bpd, name="bpd", context={"subset": "test"})
    # aim_logger.track(test_bpd, name="bpd", context={"subset": "test"})
    # log_text += f"  |  train_bpd: {train_bpd:.3f}  |  test_bpd: {test_bpd:.3f}"

    # FID, KID calculation.
    if data_name != "MNIST":
        final_metrics = evaluate_model(base=nf_backbone, prior=diffusion_prior, prior_type="diffusion",
                                       postprocess_func=lambda img: postprocess_batch(img, n_bins),
                                       data_name=data_name, dataset_res=img_size, batch_size=batch_size,
                                       num_gen=DATASET_SIZE[data_name]["train"], dataset_split="train", device=device,
                                       fid_kwargs=fid_kwargs, kid_kwargs=kid_kwargs, ssim_psnr_kwargs=ssim_psnr_kwargs)

        for metric, value in final_metrics.items():
            aim_logger.track(value, name=metric, epoch=epoch, context={"subset": "final_metrics"})
            log_text += f"  |  {metric}: {value:.3f}"

    logger.info(log_text)
    aim_logger.track(Text(log_text), name="NF final stats")

    aim_logger.close()
    logger.info("Experiment is finished.")
