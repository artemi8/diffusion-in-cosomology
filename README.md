# Cosmological Data Generation Using Diffusion Transformer (DiT) 
## Overview 
This repository contains the implementation of the Diffusion Transformer (DiT) model for generating synthetic cosmological data. The DiT model is trained using Quijote ΛCDM simulations to replicate large-scale cosmic structures such as galaxies, clusters, and voids. This approach leverages the latent diffusion process and transformer-based architecture to efficiently generate high-resolution synthetic images.

## Key Features
- Implements DiT-B2, DiT-B4, and DiT-B8 models for high-resolution image synthesis.
- Multi-GPU training supported via Distributed Data Parallel (DDP).
- Pre-trained model checkpoints available for quick deployment.
- Scripts for calculating Power Spectrum, Minkowski Functionals, and other cosmological metrics.

## Installation
To install the necessary dependencies, clone this repository and install via Poetry:
```bash
git clone https://github.com/artemi8/diffusion-in-cosomology.git
cd diffusion-in-cosomology
poetry install
```

## Training and Sampling
### Preparation Before Training
To extract ImageNet features with `1` GPUs on one node:

```bash
torchrun --nnodes=1 --nproc_per_node=1 extract_features.py --model DiT-XL/2 --data-path /path/to/imagenet/train --features-path /path/to/store/features
```

### Training DiT
We provide a training script for DiT in [`fast_dit/train.py`](train.py). This script can be used to train class-conditional 
DiT models, but it can be easily modified to support other types of conditioning. 

To launch DiT-XL/2 (256x256) training with `1` GPUs on one node:

```bash
accelerate launch --mixed_precision fp16 train.py --model DiT-XL/2 --features-path /path/to/store/features
```

To launch DiT-XL/2 (256x256) training with `N` GPUs on one node:
```bash
accelerate launch --multi_gpu --num_processes N --mixed_precision fp16 train.py --model DiT-XL/2 --features-path /path/to/store/features
```

## Evaluation (FID, Inception Score, etc.)

We include a [`fast_dit/sample_ddp.py`](sample_ddp.py) script which samples a large number of images from a DiT model in parallel. This script 
generates a folder of samples as well as a `.npz` file which can be directly used with [ADM's TensorFlow
evaluation suite](https://github.com/openai/guided-diffusion/tree/main/evaluations) to compute FID, Inception Score and
other metrics. For example, to sample 50K images from our pre-trained DiT-XL/2 model over `N` GPUs, run:

```bash
torchrun --nnodes=1 --nproc_per_node=N sample_ddp.py --model DiT-XL/2 --num-fid-samples 50000
```

There are several additional options; see [`sample_ddp.py`](sample_ddp.py) for details.

To run the evaluation suite to test the model run:

```bash
python fast_dit/evaluation/cosmo_eval/evalstats.py --real_samples real/data/path/folder --gen_samples genereated/data/path/folder --save_path save/path/folder/ --pixel_min 10 --pixel_max 150 --num_samples 5000 --box_size 1000 --MAS CIC
```


## Acknowledgements
This project builds on the [DiT (Diffusion Transformer)](https://github.com/facebookresearch/DiT) a work by Facebook Research, but the implementation used was from a faster implementation of the original repositoryc called [fast-DiT](https://github.com/chuanyangjin/fast-DiT). Special thanks to their team for the development of the original model. 