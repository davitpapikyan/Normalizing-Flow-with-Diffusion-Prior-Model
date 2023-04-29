import argparse
import glob
import logging
import os

import cleanfid
import torch

from compute import precompute_statistics


def clean_stat_directory():
    """Helper function to clean all the precomputed statistics for FID/KID calculation."""
    stats_folder = os.path.join(os.path.dirname(cleanfid.__file__), "stats")
    files = glob.glob(f"{stats_folder}/*")
    for f in files:
        os.remove(f)


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser(description="Precomputing / removing statistics for FID & KID",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--action", required=True, action="store", default="precompute",
                        choices=["precompute", "clean"], help="""Either 'precompute' or 'clean'. In case of the former 
                        the script will precompute statistics, otherwise it will remove all the precomputed stats.""")
    parser.add_argument("--data_root", required=True, action="store",
                        help="The path to the data folder where different datasets reside.")
    args = parser.parse_args()

    if args.action == "precompute":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Precalculating statistics for CelebA.
        precompute_statistics(logging, args.data_root, data_name="celeba", dataset_split="train", dataset_res=224,
                              mode="legacy_tensorflow", model_name="inception_v3", metric="FID", device=device)
        precompute_statistics(logging, args.data_root, data_name="celeba", dataset_split="train", dataset_res=224,
                              mode="legacy_tensorflow", model_name="clip_vit_b_32", metric="FID", device=device)
        precompute_statistics(logging, args.data_root, data_name="celeba", dataset_split="train", dataset_res=224,
                              mode="clean", model_name="inception_v3", metric="FID", device=device)
        precompute_statistics(logging, args.data_root, data_name="celeba", dataset_split="train", dataset_res=224,
                              mode="clean", model_name="clip_vit_b_32", metric="FID", device=device)

        # Precalculating statistics for ImageNet32.
        precompute_statistics(logging, args.data_root, data_name="imagenet32", dataset_split="train", dataset_res=32,
                              mode="legacy_tensorflow", model_name="inception_v3", metric="FID", device=device)
        precompute_statistics(logging, args.data_root, data_name="imagenet32", dataset_split="train", dataset_res=32,
                              mode="legacy_tensorflow", model_name="clip_vit_b_32", metric="FID", device=device)
        precompute_statistics(logging, args.data_root, data_name="imagenet32", dataset_split="train", dataset_res=32,
                              mode="clean", model_name="inception_v3", metric="FID", device=device)
        precompute_statistics(logging, args.data_root, data_name="imagenet32", dataset_split="train", dataset_res=32,
                              mode="clean", model_name="clip_vit_b_32", metric="FID", device=device)

        # Precalculating statistics for ImageNet64.
        precompute_statistics(logging, args.data_root, data_name="imagenet64", dataset_split="train", dataset_res=64,
                              mode="legacy_tensorflow", model_name="inception_v3", metric="FID", device=device)
        precompute_statistics(logging, args.data_root, data_name="imagenet64", dataset_split="train", dataset_res=64,
                              mode="legacy_tensorflow", model_name="clip_vit_b_32", metric="FID", device=device)
        precompute_statistics(logging, args.data_root, data_name="imagenet64", dataset_split="train", dataset_res=64,
                              mode="clean", model_name="inception_v3", metric="FID", device=device)
        precompute_statistics(logging, args.data_root, data_name="imagenet64", dataset_split="train", dataset_res=64,
                              mode="clean", model_name="clip_vit_b_32", metric="FID", device=device)

    elif args.action == "clean":
        clean_stat_directory()
