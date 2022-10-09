import os
from datetime import datetime

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from normalizing_flow import Glow, train, evaluate, get_data_transforms
from utils import setup_logger, set_seeds
from data import read_dataset


# import argparse

SEED = 5
logger = setup_logger(name="base")


@hydra.main(config_path="configs", config_name="nf_base", version_base="1.2")
def run_nf_base_experiment(configs: DictConfig):
    logger.info(f"Set seed value: {SEED}")

    workdir = os.getcwd()  # The experiment directory in hydra-outputs.

    logger.info(f"The working directory is {workdir}")
    logger.info(OmegaConf.to_yaml(configs))

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    if configs.data.name in ("CIFAR10", "CelebA"):
        in_channel = 3
    elif configs.data.name == "MNIST":
        in_channel = 1
    else:
        raise ValueError("Unknown dataset name!")

    # L: The number of blocks (including the last blocks).
    # K: The number of StepFlows in each block.
    flow = Glow(in_channel=in_channel, L=configs.model.architecture.L, K=configs.model.architecture.K,
                temperature=configs.model.architecture.temperature,
                apply_dequantization=configs.model.architecture.apply_dequantization)
    flow.to(flow.device)

    if configs.phase == "train":

        if not configs.resume.resume_exp_dir:  # If the training startes from scratch.
            logger.info("Glow model is created.")
            resume_info = None
        else:
            resume_info = {"dir": configs.resume.resume_exp_dir, "epoch": configs.resume.resume_epoch}

        if resume_info:
            workdir = "/".join(workdir.split("/")[:-1]) + "/" + resume_info["dir"]

        # Creating directory for model checkpoints.
        checkpoint_dir = os.path.join(workdir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Creating directory for results.
        result_dir = os.path.join(workdir, "results")
        os.makedirs(result_dir, exist_ok=True)

        num_params = sum(p.numel() for p in flow.parameters() if p.requires_grad)
        logger.info(f"The model has {num_params:,} learnable parameters.")

        exp_output_dir = workdir.split("/")[-1] if not resume_info else resume_info["dir"]

        # Training the model.
        train(flow, logger=logger, experiment_name=configs.experiment_name, exp_output_dir=exp_output_dir,
              data_root=configs.data.root, data_name=configs.data.name, validate=configs.data.validate,
              batch_size=configs.data.batch_size, apply_dequantization=configs.model.architecture.apply_dequantization,
              num_workers=configs.data.num_workers,
              optim_name=configs.model.optimizer.type, lr=configs.model.optimizer.lr,
              n_epochs=configs.model.training.epochs, val_freq=configs.model.training.val_freq,
              print_freq=configs.model.training.print_freq,
              log_param_distribution=configs.model.logging.log_param_distribution,
              log_gen_images_per_iter=configs.model.logging.log_gen_images_per_iter,
              save_checkpoint_freq=configs.model.training.save_checkpoint_freq, device=flow.device,
              checkpoint_dir=checkpoint_dir, num_imp_samples=configs.model.testing.num_imp_samples,
              result_dir=result_dir, resume_info=resume_info, img_size=configs.data.img_size,
              n_bits=configs.model.training.n_bits, digits=configs.data.digits)

    # TODO: test
    #  provide resume_info to load the model
    elif configs.phase == "eval":
        resume_info = {"dir": configs.resume.resume_exp_dir, "epoch": configs.resume.resume_epoch}
        workdir = "/".join(workdir.split("/")[:-1]) + "/" + resume_info["dir"]

        # Creating directory for results.
        result_dir = os.path.join(workdir, "results")
        os.makedirs(result_dir, exist_ok=True)

        checkpoint_dir = os.path.join(workdir, "checkpoints")
        checkpoint = torch.load(os.path.join(checkpoint_dir, f"model_{str(resume_info['epoch']).zfill(3)}.pt"))
        flow.load_state_dict(checkpoint["flow"])

        train_transform, test_transform = get_data_transforms(configs.data.name, configs.data.img_size,
                                                              configs.model.architecture.apply_dequantization)
        _, _, test_loader, _ = read_dataset(root=configs.data.root, name=configs.data.name,
                                            validate=configs.data.validate, batch_size=configs.data.batch_size,
                                            num_workers=configs.data.num_workers, train_transform=train_transform,
                                            test_transform=test_transform, digits=configs.data.digits, pin_memory=False,
                                            verbose=True)
        logger.info("Evaluating on test set")
        metrics = evaluate(flow, configs.model.architecture.apply_dequantization, test_loader, flow.device,
                           configs.model.testing.num_imp_samples, configs.data.img_size,
                           configs.model.training.n_bits, scores=("BPD", "FID"))
        logger.info(f"Evaluation results  |  bpd: {metrics['BPD']:.3f}  |  fid:  {metrics['FID']:.3f}")
        logger.info("Evaluation is completed.")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--hydra.job.chdir", type=bool, help="Set True for Hydra to create a new working directory.",
    #                     nargs='?', default=True, const=True)
    # parser.add_argument("--config_file", type=str, nargs='?',
    #                     help="Hydra config file to read configuration parameters from.", const=True)
    # args = parser.parse_args()
    # print(args.config_file)

    set_seeds(SEED)  # For reproducability.

    experiment_start = datetime.now()
    run_nf_base_experiment()  # Trains/evals a Glow model based on configuration parameters from configs/nf_base.yaml.
    logger.info("="*70)
    experiment_duration = datetime.now() - experiment_start
    logger.info(f"Experiment duration: {experiment_duration}")
