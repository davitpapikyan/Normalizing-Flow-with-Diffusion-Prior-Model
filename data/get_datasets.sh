#!/bin/bash
# Downloads and extracts necessary datasets.
# Run from ./data directory.

mkdir -p ./datasets/

# ImageNet 32x32.
sudo mkdir -p ./datasets/Imagenet32/

wget -P ./datasets/ -nc https://image-net.org/data/downsample/Imagenet32_train.zip
mkdir -p ./datasets/Imagenet32/train/
mv ./datasets/Imagenet32_train.zip ./datasets/Imagenet32/train/
unzip -d ./datasets/Imagenet32/train/ ./datasets/Imagenet32/train/Imagenet32_train.zip
rm -rf ./datasets/Imagenet32/train/Imagenet32_train.zip

wget -P ./datasets/ -nc https://image-net.org/data/downsample/Imagenet32_val.zip
mkdir -p ./datasets/Imagenet32/val/
mv ./datasets/Imagenet32_val.zip ./datasets/Imagenet32/val/
unzip ./datasets/Imagenet32/val/Imagenet32_val.zip -d ./datasets/Imagenet32/val/
rm -rf ./datasets/Imagenet32/val/Imagenet32_val.zip