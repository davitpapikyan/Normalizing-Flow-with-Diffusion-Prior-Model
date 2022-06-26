import os
import sys
sys.path.append("..")

import numpy as np
import torch
from aim import Run
from torchvision import transforms as vision_tranforms

from data import read_dataset, discretize
from .glow import StepFlow, GlowBlock, Glow
from .transforms import InvConv2d, InvConv2dLU, ActNorm, AffineCoupling, Squeeze, Prior, Split
from .utils import init_optimizer, track_images, save_images
from aim import Distribution


# TODO: Add variational dequatization.
# TODO: Test everything!
# TODO (on hold): Add different metrics like FID.


@torch.no_grad()
def evalueate(flow, data_loader, imp_samples, bpd_const, device):
    """Evaluating normalizing flow with importance sampling.

    Args:
        flow: Normalizing flow model.
        data_loader: The data loader.
        imp_samples: The number of samples to genreate.
        bpd_const: Bits per dimension constant.
        device: Device.

    Returns:
        Negative log-likelihood and bit per dim. values.
    """
    flow.eval()
    nll = 0.0

    for _, data in enumerate(data_loader):
        batch = data[0].to(device) if isinstance(data, list) else data.to(device)

        samples = []
        for _ in range(imp_samples):  # Importance sampling.
            _, ll = flow.transform(batch)
            samples.append(ll)

        log_likelihoods = torch.stack(samples)
        log_likelihood = torch.logsumexp(log_likelihoods, dim=0) - np.log(imp_samples)
        nll += -log_likelihood.mean(dim=0).item()

    nll /= len(data_loader)
    bpd = nll * bpd_const
    flow.train()
    return nll, bpd


def train(flow, logger, experiment_name, exp_output_dir, data_root, data_name, batch_size, num_workers, optim_name, lr,
          n_epochs, val_freq, print_freq, save_checkpoint_freq, device, checkpoint_dir, num_imp_samples, result_dir,
          resume_info: dict, img_size: int = 32):
    """Trains the normalizing flow model, runs validation steps and test step.

    Args:
        flow: Normalizing flow model.
        logger: The logger.
        experiment_name: The name of the experiment to run.
        exp_output_dir: The output directory of the experiment.
        data_root: The directory of datasets stored.
        data_name: Which data to load.
        batch_size: How many samples per batch to load.
        num_workers: How many subprocesses to use for data loading.
        optim_name: The optimizer name.
        lr: The learning rate.
        n_epochs: The number of epochs to train.
        val_freq: The number of epoch per to run validation step.
        print_freq: The number of iterations per to log the training results.
        save_checkpoint_freq: The number of epochs per to save the normalizing flow model.
        device: Device.
        checkpoint_dir: The directory for saving the normalizing flow model.
        num_imp_samples: The number of samples to generate in importance sampling.
        result_dir: The directroy for storing generated samples.
        resume_info: A dict containing info about from which directory to load the saved model and from which epoch.
        img_size: The image size to resize.
    """

    train_transform = vision_tranforms.Compose([
            vision_tranforms.Resize((img_size, img_size)),
            # vision_tranforms.RandomHorizontalFlip(),
            vision_tranforms.ToTensor(),
            discretize
    ])
    test_transform = vision_tranforms.Compose([
            vision_tranforms.Resize((img_size, img_size)),
            vision_tranforms.ToTensor(),
            discretize
    ])

    train_loader, val_loader, test_loader = read_dataset(root=data_root, name=data_name, batch_size=batch_size,
                                                         num_workers=num_workers, train_transform=train_transform,
                                                         test_transform=test_transform,
                                                         pin_memory=False, verbose=True)
    logger.info("Training, validation and test dataloaders are successfully loaded.")

    optimizer = init_optimizer(optim_name, flow.parameters(), lr=lr)

    if resume_info:  # If resume training, then load model and optimizer states.
        checkpoint = torch.load(os.path.join(checkpoint_dir, f"model_{str(resume_info['epoch']).zfill(3)}.pt"))
        flow.load_state_dict(checkpoint["flow"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = resume_info["epoch"]
        current_iter = checkpoint["current_iter"]

        # Making sure that learning rate is up-to-date.
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        logger.info("Glow model is loaded.")
    else:
        current_iter, start_epoch = 0, 0

    # Setting Aim logger (make sure run_hash doesn't contain double underscores).
    aim_logger = Run(run_hash=exp_output_dir, repo='../aim/', experiment=experiment_name)
    if not resume_info:
        aim_logger["hparams"] = {"batch_size": batch_size, "lr": lr, "L": flow.L, "K": flow.K}

    # BPD = NLL * bpd_const.
    bpd_const = np.log2(np.exp(1)) / (img_size * img_size * 3) if data_name in ("CelebA", "CIFAR10") else None

    running_loss = 0.0
    logger.info("Starting the training.\n")
    flow.train()

    for epoch in range(start_epoch+1, start_epoch+n_epochs+1):

        for iteration, data in enumerate(train_loader):

            batch = data[0].to(device) if isinstance(data, list) else data.to(device)

            # Setting gradients to None which reduced the number of memory operations than model.zero_grad().
            for param in flow.parameters():
                param.grad = None

            _, log_likelihood = flow.transform(batch)

            loss = -log_likelihood.mean(dim=0)  # Negative log-likelihood.
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if iteration % print_freq == print_freq-1:  # Print every print_freq minibatches.
                running_loss /= print_freq

                bpd = running_loss * bpd_const
                logger.info(
                    f"Epoch: {epoch:5}  |  Iteration: {(iteration+1):8}  |  nll: {running_loss:.3f}  |  bpd: {bpd:.3f}")
                aim_logger.track(bpd, name="bpd", step=current_iter, epoch=epoch, context={"subset": "train"})
                current_iter += print_freq

                generated_images, _ = flow.sample(n_samples=4, img_shape=(img_size, img_size))
                track_images(aim_logger, generated_images, step=current_iter//print_freq)

                # Tracking param distributions.
                for name, param in flow.named_parameters():
                    dist = Distribution(distribution=param.clone().cpu().data.numpy())
                    aim_logger.track(dist, name=name, step=current_iter//print_freq)

                running_loss = 0.0

        # Saving the model.
        if epoch % save_checkpoint_freq == save_checkpoint_freq-1:
            logger.info("Saving the model.")
            torch.save({
                "flow": flow.state_dict(),
                "optimizer": optimizer.state_dict(),
                "current_iter": current_iter,
            }, os.path.join(checkpoint_dir, f"model_{str(epoch).zfill(3)}.pt"))

        # Generate and track images.
        generated_images, _ = flow.sample(n_samples=64, img_shape=(img_size, img_size))
        save_images(generated_images.float(), path=result_dir, name=f"generated_{epoch}",)

        # Validation.
        if epoch % val_freq == 0:
            logger.info("Starting validation.")
            val_nll, val_bpd = evalueate(flow, val_loader, 1, bpd_const, device)
            logger.info(
                f"Epoch: {epoch:5}  |  Validation  |  nll: {val_nll:.3f}  |  bpd: {val_bpd:.3f}\n")
            aim_logger.track(val_nll, name="nll", epoch=epoch, context={"subset": "val"})
            aim_logger.track(val_bpd, name="bpd", epoch=epoch, context={"subset": "val"})

    # Testing with importance sampling technique.
    logger.info("Starting testing.")
    test_nll, test_bpd = evalueate(flow, test_loader, num_imp_samples, bpd_const, device)
    logger.info(
        f"Testing  imp_samples={num_imp_samples}  |  nll: {test_nll:.3f}  |  bpd: {test_bpd:.3f}")
    aim_logger.track(test_nll, name="nll", context={"subset": "test"})
    aim_logger.track(test_bpd, name="bpd", context={"subset": "test"})

    aim_logger.close()
    logger.info("Experiment is completed.")


__all__ = [InvConv2d, InvConv2dLU, ActNorm, AffineCoupling, StepFlow, Squeeze, Prior, Split, GlowBlock, Glow, train]
