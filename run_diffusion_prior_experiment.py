import os
import pickle
from datetime import datetime

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import ConfigAttributeError

from data import read_dataset, DATASET_SIZE
from diffusion_prior import train
from diffusion_prior.latent_formaters import get_formater
from diffusion_prior.model import DiffusionPrior
from metrics.compute import evaluate_model
from normalizing_flow import get_data_transforms, postprocess_batch, NFBackbone
from utils import setup_logger, log_environment, set_seeds, parse_metric

logger = setup_logger(name="base")


@hydra.main(config_path="configs", config_name="nf_diffusion", version_base="1.2")
def run_nf_diffusion_experiment(configs: DictConfig):
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

    # Defining normalizing flow.
    if configs.model.normalizing_flow.init_nf.mode == "pretrain":

        nf_exp_dir = "/".join(workdir.split("/")[:-1]) + "/" + configs.model.normalizing_flow.init_nf.pretrain.dir
        nf_checkpoint_dir = os.path.join(nf_exp_dir, "checkpoints")
        checkpoint_folder = f"model_gaussian_{str(configs.model.normalizing_flow.init_nf.pretrain.epoch).zfill(3)}.pt"
        nf_checkpoint_name = os.path.join(nf_checkpoint_dir, checkpoint_folder)

        # Loading pretrained normalizing flow architecture.
        with open(os.path.join(nf_exp_dir, "architecture.pkl"), "rb") as f:
            architecture = pickle.load(f)

        nf = NFBackbone(model_dir=nf_checkpoint_name, in_channel=in_channel, L=architecture["L"], K=architecture["K"],
                        learn_prior_mean_logs=architecture["learn_prior_mean_logs"],
                        freeze_flow=configs.model.normalizing_flow.freeze)

        logger.info(f"Using pretrained normalizing flow from: {nf_checkpoint_name}")

    elif configs.model.normalizing_flow.init_nf.mode == "scratch":
        nf = NFBackbone(model_dir=None, in_channel=in_channel, L=configs.model.normalizing_flow.scratch.L,
                        K=configs.model.normalizing_flow.scratch.K,
                        learn_prior_mean_logs=configs.model.normalizing_flow.scratch.learn_prior_mean_logs,
                        freeze_flow=configs.model.normalizing_flow.freeze)
        logger.info(f"Training normalzing flow from scratch with diffusion prior.")

    nf.to(nf.device)
    for param in nf.parameters():
        param.requires_grad = not configs.model.normalizing_flow.freeze

    logger.info(f"Device: {nf.device}")

    formater_class = get_formater(configs.model.normalizing_flow.latent_formater)
    try:
        latent_formater = formater_class(L=configs.model.normalizing_flow.scratch.L, in_channels=in_channel,
                                         size=configs.data.img_size)
    except ConfigAttributeError:
        latent_formater = formater_class(L=architecture["L"], in_channels=in_channel, size=configs.data.img_size)

    # Getting shapes of diffent latent parts.
    latent_sizes = [dim[1] for dim in latent_formater.postprocessed_latent_shapes]
    latent_channels = [dim[0] for dim in latent_formater.postprocessed_latent_shapes]

    unet_kwargs = {"dim": configs.model.unet.dim, "dim_mults": configs.model.unet.dim_mults,
                   "resnet_block_groups": configs.model.unet.resnet_block_groups,
                   "learned_sinusoidal_cond": configs.model.unet.learned_sinusoidal_cond,
                   "random_fourier_features": configs.model.unet.random_fourier_features,
                   "learned_sinusoidal_dim": configs.model.unet.learned_sinusoidal_dim}
    diffusion_kwargs = {"timesteps": configs.model.diffusion.timesteps,
                        "sampling_timesteps": configs.model.diffusion.sampling_timesteps,
                        "loss_type": configs.model.diffusion.loss_type,
                        "beta_schedule": configs.model.diffusion.beta_schedule,
                        "ddim_sampling_eta": configs.model.diffusion.ddim_sampling_eta}
    diffusion_prior = DiffusionPrior(latent_formater=latent_formater, unet_kwargs=unet_kwargs,
                                     latent_channels=latent_channels, diffusion_kwargs=diffusion_kwargs,
                                     image_sizes=latent_sizes)
    logger.info("Diffusion Prior is ready.")

    # Metrics.
    fid_kwargs = parse_metric(configs.model.evaluation.metrics.FID) \
        if "FID" in configs.model.evaluation.metrics else []
    kid_kwargs = parse_metric(configs.model.evaluation.metrics.KID) \
        if "KID" in configs.model.evaluation.metrics else []
    ssim_psnr_kwargs = {"data_range": configs.model.evaluation.metrics.SSIM_and_PSNR.data_range,
                        "dataloader": None} if "SSIM_and_PSNR" in configs.model.evaluation.metrics else None

    if configs.phase == "train":
        logger.info("Latent Diffusion Glow is created.")

        # Creating directory for model checkpoints.
        checkpoint_dir = os.path.join(workdir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Creating directory for results.
        result_dir = os.path.join(workdir, "results")
        os.makedirs(result_dir, exist_ok=True)

        num_params_flow = sum(p.numel() for p in nf.parameters() if p.requires_grad)
        num_params_diff = sum(p.numel() for p in diffusion_prior.parameters() if p.requires_grad)
        logger.info(f"The model has {num_params_flow+num_params_diff:,} parameters.")

        exp_output_dir = workdir.split("/")[-1]

        # Training the model.
        train(nf, diffusion_prior, logger=logger, experiment_name=configs.experiment_name,
              exp_output_dir=exp_output_dir, device=nf.device, data_name=configs.data.name,
              transformations=configs.data.transformations, data_root=configs.data.root,
              batch_size=configs.data.batch_size, num_workers=configs.data.num_workers,
              digits=configs.data.digits, n_bits=configs.model.training.n_bits,
              temperature=configs.model.normalizing_flow.temperature, img_size=configs.data.img_size,
              checkpoint_dir=checkpoint_dir, is_frozen=configs.model.normalizing_flow.freeze,
              optim_name=configs.model.optimizer.type, result_dir=result_dir,
              lr_nf_backbone=configs.model.normalizing_flow.lr, lr_diffusion=configs.model.optimizer.lr,
              n_epochs=configs.model.training.epochs, print_freq=configs.model.training.print_freq,
              save_checkpoint_freq=configs.model.training.save_checkpoint_freq,
              log_param_distribution=configs.model.logging.log_param_distribution,
              log_gen_images_per_iter=configs.model.logging.log_gen_images_per_iter,
              fid_kwargs=fid_kwargs, kid_kwargs=kid_kwargs, ssim_psnr_kwargs=ssim_psnr_kwargs)

    elif configs.phase == "eval":
        eval_info = {"dir": configs.eval.dir, "epoch": configs.eval.epoch}
        workdir = "/".join(workdir.split("/")[:-1]) + "/" + eval_info["dir"]

        # Creating directory for results.
        result_dir = os.path.join(workdir, "results")
        os.makedirs(result_dir, exist_ok=True)

        checkpoint_dir = os.path.join(workdir, "checkpoints")
        checkpoint = torch.load(os.path.join(checkpoint_dir, f"model_{str(eval_info['epoch']).zfill(3)}.pt"))
        nf.load_state_dict(checkpoint["flow"])
        diffusion_prior.load_state_dict(checkpoint["prior_dist"])

        train_transform, test_transform = get_data_transforms(configs.data.name, configs.data.img_size)
        _, _, test_loader, train_loader = read_dataset(root=configs.data.root, name=configs.data.name,
                                                       batch_size=configs.data.batch_size,
                                                       num_workers=configs.data.num_workers,
                                                       train_transform=train_transform, test_transform=test_transform,
                                                       digits=configs.data.digits, pin_memory=False, verbose=True)
        logger.info("Starting evaluation.")

        n_bins = 2.0 ** configs.model.training.n_bits
        log_text = f"Evaluation results"

        # Calculating FID, KID, SSIM and PSNR.
        if configs.data.name != "MNIST":
            metrics = evaluate_model(base=nf, prior=diffusion_prior, prior_type="diffusion",
                                     postprocess_func=lambda img: postprocess_batch(img, n_bins),
                                     data_name=configs.data.name, dataset_res=configs.data.img_size,
                                     num_gen=DATASET_SIZE[configs.data.name]["train"], dataset_split="train",
                                     device=nf.device, fid_kwargs=fid_kwargs, kid_kwargs=kid_kwargs,
                                     ssim_psnr_kwargs=ssim_psnr_kwargs)

            for metric, value in metrics.items():
                log_text += f"  |  {metric}: {value:.3f}"

        logger.info(log_text)
        logger.info("Evaluation is completed.")


if __name__ == "__main__":
    experiment_start = datetime.now()
    run_nf_diffusion_experiment()
    experiment_duration = datetime.now() - experiment_start
    logger.info(f"Experiment duration: {experiment_duration}")
