# ---------------------------------------------------------------
# Copyright (c) 2023, Davit Papikyan. All rights reserved.
#
# This file has been modified from a file in the clean-fid library
# which was released under the MIT License.
#
# Source:
# https://github.com/GaParmar/clean-fid/blob/main/cleanfid/fid.py
# Reused functions are renamed by appending '_ex' suffix.
#
# The license for the original version of this file can be
# found in this directory (LICENSE_clean-fid). The modifications
# to this file are subject to the same MIT License.
# ---------------------------------------------------------------

import glob
import os
import random
import sys

import cleanfid
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from cleanfid import fid
from cleanfid.features import build_feature_extractor
from cleanfid.fid import get_batch_features
from cleanfid.utils import ResizeDataset
from ignite.metrics import SSIM, PSNR
from tqdm import tqdm

try:
    from .utils import discretize, Storage, ResizeDatasetNumPy
except ImportError:
    from utils import discretize, Storage, ResizeDatasetNumPy

try:
    sys.path.append(os.path.abspath(os.path.join('.', '')))
    from data import unpickle
except ImportError:
    print("Cannot import unpickle")


GENERATED_SAMPLES = Storage(data=None, ready=False, index=0)
celeba_split2id = {"train": 0, "val": 1, "test": 2}
resize = T.Resize(size=224)


# Source: https://www.cs.cmu.edu/~clean-fid/stats/
SUPPORTED_STATISTICS = ("cifar10_clean_test_32.npz",
                        "cifar10_clean_train_32.npz",
                        "cifar10_legacy_pytorch_test_32.npz",
                        "cifar10_legacy_pytorch_train_32.npz",
                        "cifar10_legacy_tensorflow_test_32.npz",
                        "cifar10_legacy_tensorflow_train_32.npz")


def construct_stat_filename(data_name, mode, model_name, split, res, metric):
    model_modifier = "" if model_name == "inception_v3" else "_" + model_name
    if metric == "KID":
        return f"{data_name}_{mode}{model_modifier}_{split}_{res}_kid.npz".lower()
    else:
        return f"{data_name}_{mode}{model_modifier}_{split}_{res}.npz".lower()


# When adding a new dataset, implement this function to return a list of paths of images.
def get_dataset_files(data_root: str, dataset_name: str, split: str):  # split in (train, val)
    if dataset_name == "celeba":
        files = get_celeba_files(data_root, tuple((celeba_split2id[split], )))
    elif dataset_name in ("imagenet32", ):
        sys.path.append(os.path.abspath(os.path.join('..', 'data')))
        # from data.utils import unpickle
        if dataset_name == "imagenet32":
            path_to_data, res = os.path.join(data_root, "Imagenet32"), 32

            if split == "train" and res == 32:
                files = [os.path.join(path_to_data, f"{split}/{split}_data_batch_{idx}") for idx in range(1, 11)]
                data = np.vstack([unpickle(file)["data"] for file in files])
            else:
                data = unpickle(os.path.join(data_root, f"Imagenet32/{split}/{split}_data"))["data"]

        data = np.dstack((data[:, :res ** 2], data[:, res ** 2:2 * res ** 2], data[:, 2 * res ** 2:]))
        files = data.reshape((data.shape[0], res, res, 3))
    else:
        raise ValueError(f"Unknow dataset name {dataset_name}.")
    return files


def get_celeba_files(data_root, split):
    path_to_data = os.path.join(data_root, "celeba/img_align_celeba/img_align_celeba")
    partition_file_path = os.path.join(data_root, "celeba/list_eval_partition.csv")

    partition_file = pd.read_csv(partition_file_path)
    partition_file_sub = partition_file[partition_file["partition"].isin(split)]

    files = [os.path.join(path_to_data, file) for file in partition_file_sub.image_id.values]
    return files


def make_custom_stats_ex(data_root, name, dataset_split, dataset_res, num=200000, mode="clean", model_name="inception_v3",
                         num_workers=0, batch_size=64, device=torch.device("cuda"), verbose=True):
    stats_folder = os.path.join(os.path.dirname(cleanfid.__file__), "stats")
    os.makedirs(stats_folder, exist_ok=True)
    split, res = dataset_split, dataset_res
    if model_name == "inception_v3":
        model_modifier = ""
    else:
        model_modifier = "_" + model_name
    outf = os.path.join(stats_folder, f"{name}_{mode}{model_modifier}_{split}_{res}.npz".lower())
    # if the custom stat file already exists
    if os.path.exists(outf):
        msg = f"The statistics file {name} already exists. "
        msg += "Use remove_custom_stats function to delete it first."
        return  # The statistics file already exists
    if model_name == "inception_v3":
        feat_model = build_feature_extractor(mode, device)
        custom_fn_resize = None
        custom_image_tranform = None
    elif model_name == "clip_vit_b_32":
        from cleanfid.clip_features import CLIP_fx, img_preprocess_clip
        clip_fx = CLIP_fx("ViT-B/32", device=device)
        feat_model = clip_fx
        custom_fn_resize = img_preprocess_clip
        custom_image_tranform = None
    else:
        raise ValueError(f"The entered model name - {model_name} was not recognized.")

    # get all inception features for folder images
    np_feats = get_folder_features_ex(data_root, name, dataset_split, feat_model, num_workers=num_workers, num=num,
                                      batch_size=batch_size, device=device, verbose=verbose,
                                      mode=mode, description=f"",
                                      custom_image_tranform=custom_image_tranform,
                                      custom_fn_resize=custom_fn_resize)

    mu = np.mean(np_feats, axis=0)
    sigma = np.cov(np_feats, rowvar=False)
    print(f"saving custom FID stats to {outf}")
    np.savez_compressed(outf, mu=mu, sigma=sigma)

    # KID stats
    outf = os.path.join(stats_folder, f"{name}_{mode}{model_modifier}_{split}_{res}_kid.npz".lower())
    print(f"saving custom KID stats to {outf}")
    np.savez_compressed(outf, feats=np_feats)


def get_folder_features_ex(data_root, dataset_name, dataset_split, model=None, num_workers=12, num=None,
                           shuffle=False, batch_size=128, device=torch.device("cuda"),
                           mode="clean", custom_fn_resize=None, description="", verbose=True,
                           custom_image_tranform=None):
    # get all relevant files in the dataset
    files = get_dataset_files(data_root, dataset_name, dataset_split)
    # use a subset number of files if needed
    if num is not None:
        if shuffle:
            random.shuffle(files)
        files = files[:num]

    np_feats = get_files_features_ex(files, model, num_workers=num_workers,
                                     batch_size=batch_size, device=device, mode=mode,
                                     custom_fn_resize=custom_fn_resize,
                                     custom_image_tranform=custom_image_tranform,
                                     description=description, verbose=verbose)
    return np_feats


def get_files_features_ex(l_files, model=None, num_workers=12,
                          batch_size=128, device=torch.device("cuda"),
                          mode="clean", custom_fn_resize=None,
                          description="", verbose=True,
                          custom_image_tranform=None):
    # wrap the images in a dataloader for parallelizing the resize operation
    if isinstance(l_files, np.ndarray):
        dataset = ResizeDatasetNumPy(l_files, mode=mode)
    else:
        dataset = ResizeDataset(l_files, mode=mode)

    if custom_image_tranform is not None:
        dataset.custom_image_tranform = custom_image_tranform
    if custom_fn_resize is not None:
        dataset.fn_resize = custom_fn_resize

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                             drop_last=False, num_workers=num_workers)

    # collect all inception features
    l_feats = []
    if verbose:
        pbar = tqdm(dataloader, desc=description)
    else:
        pbar = dataloader

    for batch in pbar:
        l_feats.append(get_batch_features(batch, model, device))
    np_feats = np.concatenate(l_feats)
    return np_feats


def __sample_from_model(base, prior, prior_type, batch_size, postprocess_func=None, temperature=0.7,
                        last_latent_dim=None, data_name=None):
    global GENERATED_SAMPLES

    if prior_type == "diffusion":
        if GENERATED_SAMPLES.ready:
            real_images = GENERATED_SAMPLES.iterative_reuse_of_data(batch_size)
            return real_images
        latents = prior.sample_latents(n_samples=batch_size)
        generated_images = base.sample(latents)

    elif prior_type == "gaussian":
        if GENERATED_SAMPLES.ready:
            real_images = GENERATED_SAMPLES.iterative_reuse_of_data(batch_size)
            return real_images
        last_latent = prior.sample(shape=(batch_size, *last_latent_dim), temperature=temperature)
        generated_images = base.sample([last_latent], temperature=temperature)
    else:
        raise ValueError("Unknown prior.")

    if data_name == "celeba":
        # The resizing is done for CLIP model as we evaluate
        # FID / KID metrics for CelebA using CLIP features.
        generated_images = resize(generated_images)

    real_images = postprocess_func(generated_images)  # Return an image from [0-255].
    GENERATED_SAMPLES.append_gen_images(real_images)
    return real_images


def create_model_sampler(base, prior, prior_type, postprocess_func, temperature=0.7, last_latent_dim=None,
                         data_name=None):
    def gen(temp):
        """Helper function to fit the structure of defined by cleanfid."""
        return __sample_from_model(base, prior, prior_type, batch_size=temp.size(0), postprocess_func=postprocess_func,
                                   temperature=temperature, last_latent_dim=last_latent_dim, data_name=data_name)
    return gen


def precompute_statistics(logger, data_root, data_name, dataset_split, dataset_res, mode, model_name, metric, device):
    """Input dataset_split must be ("train", "val")."""
    stat_filename = construct_stat_filename(data_name, mode, model_name, dataset_split, dataset_res, metric)
    stats_folder = os.path.join(os.path.dirname(cleanfid.__file__), "stats")
    files = [file.split('/')[-1] for file in glob.glob(f"{stats_folder}/*")]

    if stat_filename in SUPPORTED_STATISTICS or stat_filename in files:
        # For cifar10 clean-fid has precomputed stats that will be loaded automatically.
        logger.info("Precomputed stats already exist for the dataset.")
        return

    make_custom_stats_ex(data_root=data_root, name=data_name, dataset_split=dataset_split, dataset_res=dataset_res,
                         mode=mode, model_name=model_name, batch_size=64, device=device)
    logger.info("Finished precomputation of statistics.")


def calculate_fid_kid(model_generator, data_name, dataset_res, num_gen, dataset_split, batch_size, score_type, mode,
                      device, model_name=None):
    dataset_res = 224 if data_name == "celeba" else dataset_res
    if score_type == "FID":
        score = fid.compute_fid(gen=model_generator, dataset_name=data_name, dataset_res=dataset_res, num_gen=num_gen,
                                dataset_split=dataset_split, device=device, mode=mode, model_name=model_name,
                                batch_size=batch_size)
    elif score_type == "KID":
        score = fid.compute_kid(gen=model_generator, dataset_name=data_name, dataset_res=dataset_res, num_gen=num_gen,
                                dataset_split=dataset_split, device=device, mode=mode, batch_size=batch_size)
    else:
        raise ValueError(f"Unknown score type {score_type}.")
    return score


@torch.no_grad()
def evaluate_model(base, prior, prior_type, postprocess_func, data_name, dataset_res, batch_size, num_gen,
                   dataset_split, device, fid_kwargs=None, kid_kwargs=None, ssim_psnr_kwargs=None, temperature=0.7,
                   last_latent_dim=None):
    global GENERATED_SAMPLES
    metrics = {}
    model_gen = create_model_sampler(base, prior, prior_type, postprocess_func, temperature=temperature,
                                     last_latent_dim=last_latent_dim, data_name=data_name)

    # Calculate FID scores.
    for kwarg in fid_kwargs:
        metric, mode, model_name = "FID", kwarg["mode"], kwarg["model_name"]
        metrics[f"{metric}{'_clean' if mode=='clean' else ''}_{model_name.split('_')[0]}"] = \
            calculate_fid_kid(model_generator=model_gen, data_name=data_name, dataset_res=dataset_res, num_gen=num_gen,
                              dataset_split=dataset_split, batch_size=batch_size, device=device, mode=mode,
                              score_type=metric, model_name=model_name)
        GENERATED_SAMPLES.set_ready_for_usage()

    # Calculate KID scores.
    for kwarg in kid_kwargs:
        metric, mode, model_name = "KID", kwarg["mode"], kwarg["model_name"]
        metrics[f"{metric}{'_clean' if mode=='clean' else ''}_{model_name.split('_')[0]}"] = \
            calculate_fid_kid(model_generator=model_gen, data_name=data_name, dataset_res=dataset_res, num_gen=num_gen,
                              dataset_split=dataset_split, batch_size=batch_size, device=device, mode=mode,
                              score_type=metric, model_name=model_name)
        GENERATED_SAMPLES.set_ready_for_usage()

    # Calculating SSIM.
    if ssim_psnr_kwargs:
        ssim = SSIM(data_range=ssim_psnr_kwargs["data_range"])
        psnr = PSNR(data_range=ssim_psnr_kwargs["data_range"])
        ssim.reset()
        psnr.reset()

        pbar = tqdm(ssim_psnr_kwargs["dataloader"], desc="compute SSIM and PSNR of a model",
                    total=len(ssim_psnr_kwargs["dataloader"]))
        for data in pbar:
            batch = data[0].to(device) if isinstance(data, list) else data.to(device)
            generated_batch = model_gen(torch.zeros(size=(batch.size(0), 1)))

            target, generated = generated_batch.float(), discretize(batch).float()
            # Note that both target and generated must be in the same range. That is the reason why we use discretize.

            ssim.update([target, generated])
            psnr.update([target, generated])

        metrics["SSIM"] = ssim.compute()
        metrics["PSNR"] = psnr.compute().item()
        GENERATED_SAMPLES.set_ready_for_usage()

    GENERATED_SAMPLES.reset()
    return metrics
