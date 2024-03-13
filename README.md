# Active Test-Time Adaptation: Theoretical Analyses and An Algorithm

[![Static Badge](https://img.shields.io/badge/ICLR-2024-orange)](https://openreview.net/forum?id=YHUGlwTzFB)
[![License][license-image]][license-url]

[license-url]: https://github.com/divelab/ATTA/blob/main/LICENSE
[license-image]:https://img.shields.io/badge/license-GPL3.0-green.svg


This is the official implementation of the ICLR 2024 accepted paper: Active Test-Time Adaptation: Theoretical Analyses and An Algorithm.

## News
- Code released [Mar 12th, 2024]

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Run SimTTA](#run-simtta)
- [Locked Environments for references](#locked-environments-for-references)
- [Cite](#cite)

## Introduction

Test-time adaptation (TTA) addresses distribution shifts for streaming test data in unsupervised settings. Currently, most TTA methods can only deal with minor shifts and rely heavily on heuristic and empirical studies. 
To advance TTA under domain shifts, we propose the novel problem setting of active test-time adaptation (ATTA) that integrates active learning within the fully TTA setting.
We provide a learning theory analysis, demonstrating that incorporating limited labeled test instances enhances overall performances across test domains with a theoretical guarantee. We also present a sample entropy balancing for implementing ATTA while avoiding catastrophic forgetting (CF). 
We introduce a simple yet effective ATTA algorithm, known as SimATTA, using real-time sample selection techniques. 
Extensive experimental results confirm consistency with our theoretical analyses and show that the proposed ATTA method yields substantial performance improvements over TTA methods while maintaining efficiency and shares similar effectiveness to the more demanding active domain adaptation (ADA) methods.

![Framework](/docs/imgs/ATTA.pdf)

## Installation

- Ubuntu 20.04
- Python 3.10
- PyTorch 1.10 or 2.1
- scikit-learn=1.2.2
- others

### An installation example is provided below:

```shell
conda create -n atta python=3.10
conda activate atta
conda install -y pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -y -c conda-forge tqdm pandas tensorboard matplotlib scikit-learn=1.2.2
pip install cilog psutil pynvml munch wilds gdown typed-argument-parser ruamel.yaml
```
To run the code in PyTorch **1.10**,
please remove all `@torch.compile` decorators.

### Install this package as a package

```shell
pip install -e .
```
After installing this project as a package, you may replace the `python -m ATTA.kernel.alg_main` with `attatg` in the following commands.

## Run SimTTA

```shell
python -m ATTA.kernel.alg_main --task train --config_path TTA_configs/PACS/SimATTA.yaml --atta.SimATTA.cold_start 100 --atta.SimATTA.nc_increase 1 --gpu_idx 0 --exp_round 1 [--atta.gpu_clustering]
python -m ATTA.kernel.alg_main --task train --config_path TTA_configs/VLCS/SimATTA.yaml --atta.SimATTA.cold_start 100 --atta.SimATTA.nc_increase 1 --gpu_idx 0 --exp_round 1 [--atta.gpu_clustering]
python -m ATTA.kernel.alg_main --task train --config_path TTA_configs/OfficeHome/SimATTA.yaml --atta.SimATTA.cold_start 100 --atta.SimATTA.nc_increase 1 --gpu_idx 1 --exp_round 1 [--atta.gpu_clustering]
python -m ATTA.kernel.alg_main --task train --config_path TTA_configs/TinyImageNetC/SimATTA.yaml --atta.SimATTA.cold_start 100 --atta.SimATTA.el [1e-1, 1e-2] --atta.SimATTA.nc_increase [1, 1.1, 1.2] --gpu_idx 0 --exp_round 1 --atta.gpu_clustering
```

- For GPU K-Means, add `--atta.gpu_clustering` to the above commands. In this implementation, we use PyTorch to perform batched K-Means on GPU, but it is also recommended to use the JAX library for GPU K-Means. Although a JAX's implementation is generally much faster than PyTorch's, the library ott's K-Means implementation is not as efficient as the PyTorch implementation provided. Therefore, to use JAX's K-Means, you need to implement a more efficient K-Means algorithm by yourself (e.g., transform the PyTorch implementation to JAX).
- `atta.SimATTA.cold_start` is the number of labeled samples where we maintain a contrain $alpha\ge 0.2$ to avoid training corruptions.
- `atta.SimATTA.nc_increase` is the number of clusters to increase at each iteration.
- `atta.SimATTA.el` is the bound $\epsilon_l$ for low entropy sample selections.
- `atta.SimATTA.eh` is the bound $\epsilon_h$ for high entropy sample selections.
- `atta.SimATTA.gpu_idx` is the GPU index to use.
- `atta.SimATTA.target_cluster [0, 1]` is a flag to determine whether to use the incremental clustering selection strategy.
- `atta.SimATTA.LE [0, 1]` is a flag to determine whether to use the low entropy sample selection strategy.

Pre-trained model checkpoints for PACS and VLCS are provided in `<project_root>/storage`.

## Locked Environments for references
Requirements are provided in `environment_PyTorch110_locked.yml` and `environment_PyTorch21_locked.yml`.

- `environment_PyTorch21_locked.yml`: PyTorch 2.1 environment.
- `environment_PyTorch110_locked.yml`: PyTorch 1.10 environment. 

## Cite
If you find this repo useful, please consider citing our paper:
```bibtex
@inproceedings{
gui2024atta,
title={Active Test-Time Adaptation: Theoretical Analyses and An Algorithm},
author={Shurui Gui and Xiner Li and Shuiwang Ji},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=YHUGlwTzFB}
}
```
