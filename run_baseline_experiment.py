import os
import pickle
from datetime import datetime

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from data import read_dataset, DATASET_SIZE
from metrics.compute import evaluate_model
from normalizing_flow import Glow, train, calculate_bpd, get_data_transforms, GaussianPrior, calculate_output_shapes, \
    postprocess_batch
from utils import setup_logger, log_environment, set_seeds, parse_metric

logger = setup_logger(name="base")


@hydra.main(config_path="configs", config_name="nf_base", version_base="1.2")
def run_nf_base_experiment(configs: DictConfig):
    log_environment(logger)
    set_seeds(configs.seed)  # For reproducability.
    logger.info(f"Set seed value: {configs.seed}")

    workdir = os.getcwd()  # The experiment directory in hydra-outputs.

    logger.info(f"The working directory is {workdir}")
    logger.info(OmegaConf.to_yaml(configs))

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    if configs.data.name in ("cifar10", "celeba", "imagenet32", "imagenet32"):
        in_channel = 3
    elif configs.data.name == "MNIST":
        in_channel = 1
    else:
        raise ValueError("Unknown dataset name!")

    # L: The number of blocks (including the last blocks).
    # K: The number of StepFlows in each block.
    flow = Glow(in_channel=in_channel, L=configs.model.architecture.L, K=configs.model.architecture.K,
                learn_prior_mean_logs=configs.model.architecture.learn_prior_mean_logs)
    flow.to(flow.device)
    logger.info(f"Device: {flow.device}")

    # The prior distribution.
    gaussian_prior = GaussianPrior(in_channels=2**(configs.model.architecture.L+1) * in_channel,
                                   learn_prior_mean_logs=configs.model.architecture.learn_prior_mean_logs)

    # Metrics.
    fid_kwargs = parse_metric(configs.model.evaluation.metrics.FID) \
        if "FID" in configs.model.evaluation.metrics else []
    kid_kwargs = parse_metric(configs.model.evaluation.metrics.KID) \
        if "KID" in configs.model.evaluation.metrics else []
    ssim_psnr_kwargs = {"data_range": configs.model.evaluation.metrics.SSIM_and_PSNR.data_range,
                        "dataloader": None} if "SSIM_and_PSNR" in configs.model.evaluation.metrics else None

    if configs.phase == "train":
        if not configs.load.load_exp_dir:  # If the training startes from scratch.
            logger.info("Glow model is created.")
            load_info = None

            # Saving architecture. Will be useful for training diffusion prior.
            architecture = {"L": configs.model.architecture.L, "K": configs.model.architecture.K,
                            "learn_prior_mean_logs": configs.model.architecture.learn_prior_mean_logs}
            with open(os.path.join(workdir, "architecture.pkl"), "wb") as f:
                pickle.dump(architecture, f)
        else:
            load_info = {"dir": configs.load.load_exp_dir, "epoch": configs.load.load_epoch}

        if load_info:
            workdir = "/".join(workdir.split("/")[:-1]) + "/" + load_info["dir"]

        # Creating directory for model checkpoints.
        checkpoint_dir = os.path.join(workdir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Creating directory for results.
        result_dir = os.path.join(workdir, "results")
        os.makedirs(result_dir, exist_ok=True)

        num_params = sum(p.numel() for p in flow.parameters() if p.requires_grad)
        logger.info(f"The model has {num_params:,} parameters.")

        exp_output_dir = workdir.split("/")[-1] if not load_info else load_info["dir"]

        # Training the model.
        train(flow, gaussian_prior, logger=logger, experiment_name=configs.experiment_name,
              exp_output_dir=exp_output_dir, data_root=configs.data.root, data_name=configs.data.name,
              transformations=configs.data.transformations, batch_size=configs.data.batch_size,
              num_workers=configs.data.num_workers, optim_name=configs.model.optimizer.type,
              lr=configs.model.optimizer.lr, n_epochs=configs.model.training.epochs,
              print_freq=configs.model.training.print_freq,
              log_param_distribution=configs.model.logging.log_param_distribution,
              log_gen_images_per_iter=configs.model.logging.log_gen_images_per_iter,
              save_checkpoint_freq=configs.model.training.save_checkpoint_freq, device=flow.device,
              checkpoint_dir=checkpoint_dir, result_dir=result_dir, resume_info=load_info,
              img_size=configs.data.img_size, n_bits=configs.model.training.n_bits,
              temperature=configs.model.training.temperature, digits=configs.data.digits, fid_kwargs=fid_kwargs,
              kid_kwargs=kid_kwargs, ssim_psnr_kwargs=ssim_psnr_kwargs)

    elif configs.phase == "eval":
        load_info = {"dir": configs.load.load_exp_dir, "epoch": configs.load.load_epoch}
        workdir = "/".join(workdir.split("/")[:-1]) + "/" + load_info["dir"]

        # Creating directory for results.
        result_dir = os.path.join(workdir, "results")
        os.makedirs(result_dir, exist_ok=True)

        checkpoint_dir = os.path.join(workdir, "checkpoints")
        checkpoint = torch.load(os.path.join(checkpoint_dir, f"model_gaussian_{str(load_info['epoch']).zfill(3)}.pt"))
        flow.load_state_dict(checkpoint["flow"])
        gaussian_prior.load_state_dict(checkpoint["prior_dist"])

        train_transform, test_transform = get_data_transforms(configs.data.name, configs.data.img_size)
        _, _, test_loader, train_loader = read_dataset(root=configs.data.root, name=configs.data.name,
                                                       batch_size=configs.data.batch_size,
                                                       num_workers=configs.data.num_workers,
                                                       train_transform=train_transform, test_transform=test_transform,
                                                       digits=configs.data.digits, pin_memory=False, verbose=True)
        logger.info("Starting evaluation.")

        # Evaluating BPD.
        n_bins = 2.0 ** configs.model.training.n_bits
        n_pixel = configs.data.img_size * configs.data.img_size * 3.0
        bpd_const = np.log2(np.e) / n_pixel

        test_bpd = calculate_bpd(flow, gaussian_prior, data_loader=test_loader, n_bits=configs.model.training.n_bits,
                                 n_bins=n_bins, n_pixel=n_pixel, bpd_const=bpd_const, device=flow.device)
        train_bpd = calculate_bpd(flow, gaussian_prior, data_loader=train_loader, n_bits=configs.model.training.n_bits,
                                  n_bins=n_bins, n_pixel=n_pixel, bpd_const=bpd_const, device=flow.device)
        log_text = f"Evaluation results  |  train_bpd: {train_bpd:.3f}  |  test_bpd: {test_bpd:.3f}"

        # Calculating FID, KID, SSIM and PSNR.
        if configs.data.name != "MNIST":
            latent_dimensions = calculate_output_shapes(L=flow.L, in_channels=flow.in_channel,
                                                        size=configs.data.img_size)
            metrics = evaluate_model(base=flow, prior=gaussian_prior, prior_type="gaussian",
                                     postprocess_func=lambda img: postprocess_batch(img, n_bins),
                                     data_name=configs.data.name, dataset_res=configs.data.img_size,
                                     num_gen=DATASET_SIZE[configs.data.name]["train"], dataset_split="train",
                                     device=flow.device, fid_kwargs=fid_kwargs, kid_kwargs=kid_kwargs,
                                     ssim_psnr_kwargs=ssim_psnr_kwargs, temperature=configs.model.training.temperature,
                                     last_latent_dim=latent_dimensions[-1])

            for metric, value in metrics.items():
                log_text += f"  |  {metric}: {value:.3f}"

        logger.info(log_text)
        logger.info("Evaluation is completed.")


if __name__ == "__main__":
    experiment_start = datetime.now()
    run_nf_base_experiment()
    experiment_duration = datetime.now() - experiment_start
    logger.info(f"Experiment duration: {experiment_duration}")
