# Gradient Variance Loss
[![report](https://img.shields.io/badge/arxiv-report-red)](https://arxiv.org/abs/2202.00997)

[ICASSP 2022] Official implementation of the Gradient Variance loss presented in the paper [paper](https://arxiv.org/abs/2202.00997) "Gradient Variance Loss for Structure-Enhanced Image Super-Resolution".

## Requirements
for installing required packages run
` pip install -r requirements.txt`

## Usage
To train the VDSR model with the gradient variance loss run the following command

` python train.py --dataroot [path to DIV2K dataset] --cuda`


## Introduction
"Gradient Variance Loss for Structure-Enhanced Image Super-Resolution"

By Lusine Abrahamyan, Anh Minh Truong, Wilfried Philips and Nikos Deligiannis.
### Approach
We observe that gradient maps of images generated
by the models trained with the L1/L2 losses have significantly lower variance than the gradient maps of the original
high-resolution images. 

In this work, we introduce a structure-enhancing loss
function, coined Gradient Variance (GV) loss, to minimize the difference between the variances of predicted and original gradient maps and generate
textures with perceptual-pleasant details.

### Performance
Public benchmark test results and DIV2K validation results (PSNR(dB) / SSIM).

<img src="https://github.com/lusinlu/skipnet_evaluation/blob/main/figures/params_vs_top1.png" width="300" height="250">

## Citation
If you find the code useful for your research, please consider citing our works

```
@article{abrahamyangvloss,
  title={Gradient Variance Loss for Structure-Enhanced Image Super-Resolution},
  author={Lusine, Abrahamyan and  Anh Minh, Truong and  Wilfried, Philips and Nikos, Deligiannis},
  journal={Proceedings of the International Conference on Acoustics, Speech, and Signal Processing (ICASSP)},
  publisher = {IEEE},
  year={2022}
}
```

## Acknowledgement
Codes for the VDSR model are from [pytorch-vdsr](https://github.com/twtygqyy/pytorch-vdsr).



