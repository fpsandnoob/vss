<div align="center">

# Solving Zero-Shot Sparse-View CT Reconstruction With Variational Score Solver

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
<!-- [![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020) -->

</div>

## Abstract

Computed tomography (CT) stands as a ubiquitous medical diagnostic tool. Nonetheless, the radiation-related concerns associated with CT scans have raised public apprehensions, particularly limiting its utility in intraoperative scenarios. Mitigating radiation dosage in CT imaging poses an inherent challenge as it inevitably compromises the fidelity of CT reconstructions, impacting diagnostic accuracy. While previous deep learning techniques have exhibited promise in enhancing CT reconstruction quality, they remain hindered by the reliance on paired data, which is often arduous to procure. In this study, we present a novel approach named Variational Score Solver (VSS) for solving sparse-view reconstruction without paired data. Specifically, our approach entails the acquisition of a probability distribution from densely sampled CT reconstructions, employing a latent diffusion model. High-quality reconstruction outcomes are achieved through an iterative process, wherein the diffusion model serves as the prior term, subsequently integrated with the data consistency term. Notably, rather than directly employing the prior diffusion model, we distill prior knowledge by identifying the fixed point of the diffusion model. This framework empowers us to exercise precise control over the CT reconstruction process. Moreover, we depart from modeling the reconstruction outcomes as deterministic values, opting instead for a dynamic distribution-based approach. This enables us to achieve more accurate reconstructions utilizing a trainable model. Our approach introduces a fresh perspective to the realm of zero-shot CT reconstruction, circumventing the constraints of supervised learning. Our extensive qualitative and quantitative experiments unequivocally demonstrate that VSS surpasses other contemporary unsupervised and achieves comparable results compared with the most advance supervised methods in sparse-view reconstruction tasks.
<p align="center">
    <img src="figures/fig_framework.png" style="background-color:white;padding:10px">
</p>

## Installation

#### Pip

* Pytorch >= 2.0
* Diffusers == 0.15.1
* Transformers == 4.35.2
* [Operator Discretization Library (ODL)](https://github.com/odlgroup/odl)
* [Torch-radon](https://github.com/matteo-ronchetti/torch-radon) (Need to be patched. Patch in: [Helix2fan](https://github.com/faebstn96/helix2fan))

```bash
# clone project
git clone https://github.com/fpsandnoob/vss
cd vss

# [OPTIONAL] create conda environment
conda create -n myenv python=3.10
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

## Data

#### Model Checkpoint
From this [link](https://drive.google.com/file/d/14BwMCOtFpREKjY1GqNcrvlrRbvOhhbVo/view?usp=sharing), download the checkpoints (including DDPM and LDM for 256, LDM for 512) and unzip them to ''data/ckpt'' directory.

#### Preprocessed Data
From this [link](https://drive.google.com/file/d/1hZuEn_y_BYPWvjGN2QeyP99JICe7pNtL/view?usp=drive_link), download the preprocessed data and unzip them to ''data/dose'' directory.

<!-- #### Conda

```bash
# clone project
git clone https://github.com/YourGithubName/your-repo-name
cd your-repo-name

# create conda environment and install dependencies
conda env create -f environment.yaml -n myenv

# activate conda environment
conda activate myenv
``` -->

## How to inference

You can run VSS like this

```bash
# train on CPU
python src/eval.py experiment=mayo_256_sparse_18_ldm_dlir_reddiff model.im_out_dir=eval_data/mayo_sparse_view_18_vss
python src/eval.py experiment=mayo_256_sparse_32_ldm_dlir_reddiff model.im_out_dir=eval_data/mayo_sparse_view_32_vss
python src/eval.py experiment=mayo_256_sparse_64_ldm_dlir_reddiff model.im_out_dir=eval_data/mayo_sparse_view_64_vss
```

or you can create your own inference config in ''configs/experiment''
```yaml
# @package _global_

# to execute this experiment run:
# python train.py experiment=example

defaults:
  - override /guidence: latent_dlir.yaml
  - override /degrade_op: sparse_view_64.yaml
  - override /data: ct_ddpm.yaml
  - override /model: ldm_reddiff_mayo_512.yaml
  - override /trainer: gpu.yaml

# all parameters below will be merged with parameters from default configurations set above
# this allows you to overwrite only specified parameters

tags: ["eval", "ni", "sr_test"]

seed: 3614

data:
  pin_memory: False
  resolution: 512

degrade_op:
  im_shape: [512, 512]

guidence:
  _target_: src.models.components.guidence_modules.LatentDeepLatentIterativeReconstruct
  # src.models.components.guidence_modules.LatentDeepLatentIterativeReconstructEnsembler
  scale: 0.1
  optimizer:
    _target_: src.models.components.guidence_modules.AdamOptimizer
    # _target_: src.models.components.guidence_modules.SGDOptimizer
    # _target_: src.models.components.guidence_modules.MomentumOptimizer
  diff_module:
    _target_: src.models.components.guidence_modules.NormModule
    ord: 2
  
```
