import os
import sys
sys.path.append("..")
from datetime import datetime
import numpy as np
import torch
from aim import Run

from data import read_dataset, calculate_fid
from .glow import StepFlow, GlowBlock, Glow
from .transforms import InvConv2d, InvConv2dLU, ActNorm, AffineCoupling, Squeeze, Split
from .prior import GaussianPrior
from .utils import init_optimizer, track_images, save_images, preprocess_batch, postprocess_batch, calculate_loss, \
    get_data_transforms, calculate_output_shapes
from aim import Distribution, Text

# TODO: (on hold) Add variational dequatization.

@torch.no_grad()
def evaluate(flow, prior_dist, apply_dequantization: bool, data_loader, device, num_imp_samples, img_size: int = 32,
             n_bits: int = 5, scores: tuple = ("BPD", "FID")):
    """Evaluates the metrics provided in scores parameter.

    Args:
        flow: Normalizing flow model.
        prior_dist: The prior distribution.
        apply_dequantization: Whether to assume dequantized input or not.
        data_loader: The data loader.
        device: Device.
        num_imp_samples: The number of samples to genreate.
        img_size: The image size.
        n_bits: The number of bits to encode.
        scores: A tuple of scores to evaluate.

    Returns:
        A dict mapping the provided scores to estimated values.
    """
    metrics = {}

    if "BPD" in scores:
        n_pixel = img_size * img_size * 3.0
        bpd = calculate_bpd(flow, prior_dist, apply_dequantization, data_loader, n_bits, 2.0 ** n_bits, n_pixel,
                            np.log2(np.e) / n_pixel, num_imp_samples, device)
        metrics["BPD"] = bpd

    if "FID" in scores:
        fid = 0.0

        # TODO: implement FID calculation.
        #  for test set(ask Christina on how to do correctly calculate FID)
        #  Ask Christina on how to evaluate FID score, (test set against generated images) * X times???
        #   https://github.com/pfnet-research/sngan_projection/blob/master/evaluation.py#L220

        metrics["FID"] = fid

    return metrics


@torch.no_grad()
def calculate_bpd(flow, prior_dist, apply_dequantization: bool, data_loader, n_bits, n_bins, n_pixel, bpd_const,
                  num_imp_samples, device):
    """Calculating BPD of normalizing flow with importance sampling.

    Args:
        flow: Normalizing flow model.
        prior_dist: The prior distribution.
        apply_dequantization: Whether to assume dequantized input or not.
        data_loader: The data loader.
        n_bits: The number of bits to encode.
        n_bins: The number of bins.
        n_pixel: The number of pixels.
        bpd_const: Constant to compute BPD.
        num_imp_samples: The number of samples to genreate.
        device: Device.

    Returns:
        Bits per dimension.
    """
    flow.eval()
    bpd = 0.0

    for _, data in enumerate(data_loader):
        batch = data[0].to(device) if isinstance(data, list) else data.to(device)
        batch = preprocess_batch(batch, n_bits, n_bins, apply_dequantization)

        sample_log_likelihoods = []
        for _ in range(num_imp_samples):  # Importance sampling.
            ll = torch.zeros(batch.size(0), device=device)

            latents, ll = flow.transform(batch if apply_dequantization else batch + torch.rand_like(batch) / n_bins, ll)
            ll += sum(prior_dist.compute_log_prob(z) for z in latents)

            sample_log_likelihoods.append(ll)

        log_likelihoods = torch.stack(sample_log_likelihoods)
        log_likelihood = torch.logsumexp(log_likelihoods, dim=0) - np.log(num_imp_samples)
        bpd += ((np.log(n_bins) * n_pixel - log_likelihood) * bpd_const).mean(dim=0).item()

    bpd /= len(data_loader)
    flow.train()
    return bpd


def train(flow, prior_dist, logger, experiment_name, exp_output_dir, data_root, data_name, validate, batch_size,
          apply_dequantization, num_workers, optim_name, lr, n_epochs, val_freq, print_freq, save_checkpoint_freq,
          log_param_distribution, log_gen_images_per_iter, device, checkpoint_dir, num_imp_samples, result_dir,
          resume_info: dict, img_size: int = 32, n_bits: int = 5, digits: list = None):
    """Trains the normalizing flow model, runs validation and test steps.

    Args:
        flow: Normalizing flow model.
        prior_dist: The prior distribution.
        logger: The logger.
        experiment_name: The name of the experiment to run.
        exp_output_dir: The output directory of the experiment.
        data_root: The directory of datasets stored.
        data_name: Which data to load.
        validate: Whether to create validation set or not.
        batch_size: How many samples per batch to load.
        apply_dequantization: Whether to assume dequantized input or not.
        num_workers: How many subprocesses to use for data loading.
        optim_name: The optimizer name.
        lr: The learning rate.
        n_epochs: The number of epochs to train.
        val_freq: The number of epoch per to run validation step.
        print_freq: The number of iterations per to log the training results.
        save_checkpoint_freq: The number of epochs per to save the normalizing flow model.
        log_param_distribution: Either to log model parameters' densities or not.
        log_gen_images_per_iter: Per how many logging iterations to log generated images.
        device: Device.
        checkpoint_dir: The directory for saving the normalizing flow model.
        num_imp_samples: The number of samples to generate in importance sampling.
        result_dir: The directroy for storing generated samples.
        resume_info: A dict containing info about from which directory to load the saved model and from which epoch.
        img_size: The image size to resize.
        n_bits: Number of bits for BPD computation.
        digits: A list of digits to select. If None, all digits will be selected.
    """

    def save_model(logger_obj, model, optim, current_epoch, current_iteration, checkpoint_directory):
        """Helper function to save model."""
        logger_obj.info("Saving the model.")
        torch.save({
            "flow": model.state_dict(),
            "optimizer": optim.state_dict(),
            "current_iter": current_iteration,
        }, os.path.join(checkpoint_directory, f"model_{str(current_epoch).zfill(3)}.pt"))

    # Preparing data.
    train_transform, test_transform = get_data_transforms(data_name, img_size, apply_dequantization)
    train_loader, val_loader, test_loader, testset = read_dataset(root=data_root, name=data_name, validate=validate,
                                                                  batch_size=batch_size, num_workers=num_workers,
                                                                  train_transform=train_transform,
                                                                  test_transform=test_transform, digits=digits,
                                                                  pin_memory=False, verbose=True)
    logger.info("Training, validation and test dataloaders are successfully loaded.")

    optimizer = init_optimizer(optim_name, flow.parameters(), lr=lr)

    if resume_info:  # If resume training, then load model and optimizer states.
        checkpoint = torch.load(os.path.join(checkpoint_dir, f"model_{str(resume_info['epoch']).zfill(3)}.pt"))
        flow.load_state_dict(checkpoint["flow"])
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

    running_loss = 0.0
    latent_dimensions = calculate_output_shapes(L=flow.L, in_channel=flow.in_channel, size=img_size)
    logger.info("Starting the training.\n")
    flow.train()

    # [Training]
    for epoch in range(start_epoch+1, start_epoch+n_epochs+1):

        start_epoch_time = datetime.now()
        for iteration, data in enumerate(train_loader):
            batch = data[0].to(device) if isinstance(data, list) else data.to(device)
            batch = preprocess_batch(batch, n_bits, n_bins, apply_dequantization)

            # Setting gradients to None which reduces the number of memory operations compared to model.zero_grad().
            for param in flow.parameters():
                param.grad = None

            log_likelihood = torch.zeros(batch.size(0), device=device)
            latents, log_likelihood = flow.transform(batch if apply_dequantization else
                                                     batch + torch.rand_like(batch) / n_bins, log_likelihood)
            log_likelihood += sum(prior_dist.compute_log_prob(z) for z in latents)  # Prior log probability.

            # See the discussion on why noise is being added in [Section 3.1,
            # A note on the evaluation of generative models] paper.

            loss = calculate_loss(log_likelihood, n_bins, n_pixel)  # Bits per dimension.
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if iteration % print_freq == print_freq-1:  # Print every print_freq minibatches.
                running_loss /= print_freq

                logger.info(
                    f"Epoch: {epoch:5}  |  Iteration: {(iteration+1):8}  |  bpd: {running_loss:.3f}")
                aim_logger.track(running_loss, name="bpd", step=current_iter, epoch=epoch, context={"subset": "train"})
                current_iter += print_freq

                # After every log_gen_images_per_iter train logs generate and visualize images on Aim.
                if ((iteration+1) // print_freq) % log_gen_images_per_iter == 0:

                    if log_param_distribution:
                        # Tracking param distributions.
                        for name, param in flow.named_parameters():
                            dist = Distribution(distribution=param.clone().cpu().data.numpy())
                            aim_logger.track(dist, name=name, step=current_iter // print_freq)

                    flow.eval()
                    latents = [prior_dist.sample(shape=(4, *dim)) for dim in latent_dimensions]
                    generated_images = flow.sample(latents)
                    track_images(aim_logger, generated_images, step=current_iter // print_freq)
                    flow.train()

                running_loss = 0.0

        # An epoch of training is completed.
        # Saving the model.
        if epoch % save_checkpoint_freq == 0:
            save_model(logger, flow, optimizer, epoch, current_iter, checkpoint_dir)

        # Generate and save images after each epoch.
        flow.eval()
        latents = [prior_dist.sample(shape=(64, *dim)) for dim in latent_dimensions]
        generated_images = flow.sample(latents)
        save_images(generated_images.float(), path=result_dir, name=f"generated_{epoch}")
        flow.train()

        # [Validation]
        if validate and epoch % val_freq == 0:
            logger.info("Starting validation.")
            val_bpd = calculate_bpd(flow, prior_dist, apply_dequantization, val_loader, n_bits, n_bins, n_pixel,
                                    bpd_const, 1, device)
            logger.info(f"Epoch: {epoch:5}  |  Validation  |  bpd: {val_bpd:.3f}")
            aim_logger.track(val_bpd, name="bpd", epoch=epoch, context={"subset": "val"})

        running_loss = 0.0

        # Logging datetime information duration.
        logger.info("-"*70)
        end_epoch_time = datetime.now()
        duration = end_epoch_time - start_epoch_time
        logger.info(f"Duration of epoch: {duration}")
        estimated_finish = datetime.now() + duration * (n_epochs-epoch)
        logger.info(f"Estimated end of training: {estimated_finish}")
        logger.info(f"Time remaining: {estimated_finish-datetime.now()}\n")

    # Making sure to save the model parameters after the last epoch.
    if epoch % save_checkpoint_freq != 0:
        save_model(logger, flow, optimizer, epoch, current_iter, checkpoint_dir)

    # [Testing]
    logger.info("Starting testing.")
    log_text = f"Testing  imp_samples={num_imp_samples}"
    # train_bpd = calculate_bpd(flow, train_loader, n_bits, n_bins, n_pixel, bpd_const, num_imp_samples, device)
    # test_bpd = calculate_bpd(flow, test_loader, n_bits, n_bins, n_pixel, bpd_const, num_imp_samples, device)
    # aim_logger.track(test_bpd, name="bpd", context={"subset": "test"})
    # log_text = f"Testing  imp_samples={num_imp_samples}  |  train_bpd: {train_bpd:.3f}  |  test_bpd: {test_bpd:.3f}"
    # aim_logger.track(Text(log_text), name="NF final stats")


    # This is workable solution.

    # fid = 0.0
    # Testing with importance sampling technique.
    # logger.info("Calculating FID.")
    # fid_values = []
    # for _ in range(10):
    #     generated_images, _ = flow.sample(n_samples=100, img_shape=(img_size, img_size))
    #     fid_values.append(calculate_fid(generated_images.to(torch.uint8), testset, device))
    # fid, std = np.mean(fid_values), np.std(fid_values)
    # logger.info(f"FID score mean: {fid}, std: {std}\n")


    metrics = evaluate(flow, prior_dist, apply_dequantization, test_loader, flow.device, num_imp_samples, img_size,
                       n_bits, scores=("BPD", "FID"))
    train_bpd = calculate_bpd(flow, prior_dist, apply_dequantization, train_loader, n_bits, n_bins, n_pixel,
                              bpd_const, num_imp_samples, device)
    aim_logger.track(metrics["BPD"], name="bpd", context={"subset": "test"})
    aim_logger.track(metrics["FID"], name="fid", context={"subset": "test"})
    log_text += f"  |  train_bpd: {train_bpd:.3f}  |  test_bpd: {metrics['BPD']:.3f}  |  fid: {metrics['FID']:.3f}"
    aim_logger.track(Text(log_text), name="NF final stats")
    logger.info(log_text)

    # logger.info(log_text + f"  |  fid: {fid:.3f}")
    # aim_logger.track(fid, name="fid", context={"subset": "test"})

    aim_logger.close()
    logger.info("Experiment is completed.")


__all__ = [InvConv2d, InvConv2dLU, ActNorm, AffineCoupling, StepFlow, Squeeze, GaussianPrior, Split, GlowBlock, Glow,
           get_data_transforms, train, evaluate]


# TODO: Refactor codes and change all pytorch tensor type annotations to Tensor!!!