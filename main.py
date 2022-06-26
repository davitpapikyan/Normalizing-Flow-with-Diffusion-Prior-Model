import os

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from normalizing_flow import Glow, train
from utils import setup_logger, set_seeds

# import argparse


@hydra.main(config_path="./configs", config_name="nf_base_configs", version_base="1.1")
def main(configs: DictConfig):
    workdir = os.getcwd()  # The experiment directory in hydra-outputs.

    logger = setup_logger(name="base")
    logger.info(f"The working directory is {workdir}")
    logger.info(OmegaConf.to_yaml(configs))

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    flow = Glow(L=configs.model.architecture.L, K=configs.model.architecture.K)
    flow.to(flow.device)

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

    logger.info(f"The model has {sum(p.numel() for p in flow.parameters() if p.requires_grad):,} learnable parameters.")

    exp_output_dir = workdir.split("/")[-1] if not resume_info else resume_info["dir"]

    train(flow, logger=logger, experiment_name=configs.experiment_name, exp_output_dir=exp_output_dir,
          data_root=configs.data.root, data_name=configs.data.name, batch_size=configs.data.batch_size,
          num_workers=configs.data.num_workers, optim_name=configs.model.optimizer.type, lr=configs.model.optimizer.lr,
          n_epochs=configs.model.training.epochs, val_freq=configs.model.training.val_freq,
          print_freq=configs.model.training.print_freq,
          save_checkpoint_freq=configs.model.training.save_checkpoint_freq, device=flow.device,
          checkpoint_dir=checkpoint_dir, num_imp_samples=configs.model.testing.num_imp_samples, result_dir=result_dir,
          resume_info=resume_info)


    # TODO: Write a function to set a model on device. go to you DDPM repo and see the set_device function in model.


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--hydra.job.chdir", type=bool, help="Set True for Hydra to create a new working directory.",
    #                     nargs='?', default=True, const=True)
    # args = parser.parse_args()

    set_seeds()  # For reproducability.
    main()
