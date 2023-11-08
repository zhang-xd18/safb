## Overview

This is the PyTorch implementation of paper [Sensing-aided CSI Feedback with Deep Learning for Massive MIMO Systems](https://arxiv.org).

## Requirements

To use this project, you need to ensure the following requirements are installed.

- Python >= 3.9
- [PyTorch == 1.10.0]([https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/previous-versions/#v1100))
- [thop](https://github.com/Lyken17/pytorch-OpCounter)

## Project Preparation

#### A. Data Preparation

The channel state information (CSI) matrix is generated according to the Saleh-Valenzuela channel model using MATLAB. The detailed parameters are listed in Table I of the paper. We also provide a pre-processed version for direct download in [Google Drive](https://drive.google.com/drive/folders/13FqZMJWk0kPifM2kBhBxOzN1IR5HR0Rd?usp=sharing), which is easier to use for this task. You can also download it from [Baidu Netdisk](https://pan.baidu.com/s/1YB5OTqq6zvxtPq3n9GhG5w?pwd=17tu).
You can also generate your own dataset based on the Saleh-Valenzuela channel model. The details of data generation and pre-processing can be found in our paper.

#### B. Checkpoints Downloading

The model checkpoints should be downloaded if you would like to reproduce our results. All the checkpoints files can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1Q3lJSlKeBoIQMu75dRqRYG_oC1SeA6OO?usp=sharing) or [Baidu Netdisk](https://pan.baidu.com/s/1nkP7w78zmxpa8MjTcS7Wiw?pwd=9ehy).

#### C. Project Tree Arrangement

We recommend you to arrange the project tree as follows.

```
home
├── safb  # The cloned safb repository
│   ├── dataset
│   ├── models
│   ├── utils
│   ├── main.py
├── SV  # The dataset folder
│   ├── H5_train.mat
│   ├── H10_train.mat
│   ├── ...
├── Checkpoints # The checkpoint folder
│   ├── Joint_cr64_L5.pth
│   ├── FB_cr64_L5.pth
│   ├── RE_L5.pth
│   ├── ...
├── run.sh  # The bash script
...
```

## Train Sensing-aided DL-based CSI feedback from Scratch

An example of `run.sh` is listed below. Simply use it with `bash run.sh`. It will start the sensing-aided DL-based CSI feedback training from scratch. Change the training mode using `--mode` with  `Joint` for training in the joint manner, `FB` for training FBNet separately, and `RE` for training RENet separately, respectively. Change number of scatters using `--L` and change compression ratio with `--cr`. (Please remove the comments when using the `run.sh`)

``` bash
python /home/safb/main.py \
  --data-dir /home/SV \ # path to the dataset
  --mode Joint \ # training mode, chosen from ['FB','RE','Joint']
  --epochs 1000 \ # training epochs
  --batch-size 200 \
  --workers 0 \ 
  --cr 64 \  # compression ratio, typical values: 64, 128, 256
  --L 5 \ # number of scatters Ns, typical values: 5, 10
  --scheduler cosine \ lr scheduler, chosen from ['cosine','const'], other schedulers can also be explored
  --gpu 0 \
  --root /home/results # path to save training log and checkpoints
```

## Results and Reproduction

The main results reported in our paper are presented as follows. All the listed results can be found in Table II, Table III, and Table IV of our paper. They are achieved from training this sensing-aided DL-based CSI feedback scheme in a separate manner or in a joint manner. (Separate manner: 500 epochs for both FBNet and RENet. Joint manner: 1000 epochs. More details for training settings can be found in our paper.)

### Results for seperate training manner
Number of <br> scatters | Compression <br> Ratio | FBNet | RENet | JNet-Sep | Params | FlOPs | Checkpoints
:--: | :--: | :--: | :--: | :--: | :--: | :--: | :--:
5 | 64 <br> 128 <br> 256 | -17.60 <br> -12.86 <br> -5.664 | -10.51 | -9.699 <br> -8.488 <br> -4.653 | 1.908M <br> 1.880M <br> 1.866M | 60.146K <br> 31.394K <br> 17.786K | FB-cr64-L5.pth + RE_L5.pth <br> FB-cr128-L5.pth + RE_L5.pth  <br> FB-cr256-L5.pth + RE_L5.pth |
10 | 64 <br> 128 <br> 256 | -10.48 <br> -4.764 <br> -2.487 | -13.54 | -8.689 <br> -4.329 <br> -2.219 | 2.500M <br> 2.445M <br> 2.419M | 111.666K <br> 57.314K <br> 30.906K | FB-cr64-L10.pth + RE_L10.pth <br> FB-cr128-L10.pth + RE_L10.pth  <br> FB-cr256-L10.pth + RE_L10.pth |

### Results for joint training manner
Number of <br> scatters | Compression <br> Ratio | JNet-Joint | Params | Flops | Checkpoints
:--: | :--: | :--: | :--: | :--: | :--:
5 | 64 <br> 128 <br> 256 | -9.826 <br> -8.459 <br> -4.314 | 1.908M <br> 1.880M <br> 1.866M | 60.146K <br> 31.394K <br> 17.786K | Joint_cr64_L5.pth <br> Joint_cr128_L5.pth  <br> Joint_cr256_L5.pth|
10 | 64 <br> 128 <br> 256 | -9.207 <br> -5.393 <br> -2.582 | 2.500M <br> 2.445M <br> 2.419M | 111.666K <br> 57.314K <br> 30.906K | Joint_cr64_L10.pth <br> Joint_cr128_L10.pth <br> Joint_cr256_L10.pth |


As aforementioned, we provide model checkpoints for all the results. Our code library supports easy inference. 

**To reproduce all these results, simple add `--evaluate` to `run.sh` and pick the corresponding pre-trained model(s) with `--pretrained`.** An example is shown as follows.

``` bash
python /home/safb/main.py \
    --data-dir /home/SV/ \ # path to the dataset
    --evaluate \ # flag for evaluate
    -b 200 \
    --workers 0 \
    --cpu \
    --cr 64 \  # compression ratio, typical values: 64, 128, 256
    --L 5 \  # number of scatters Ns, typical values: 5, 10
    --mode Joint \  # inference mode, chosen from ['Joint', 'FB', 'RE']
    --root /home/results \ # path to save the log
    --pretrained /home/Checkpoints/Joint-cr64-L5.pth \ # path to the checkpoint
    # [optional] --pretrained2 /home/Checkpoints/RE-L5.pth \ # to evaluate the feedback performance of the model trained in a separate manner, two pretrained models FBNet and RENet should be provided. MORE CLEAR EXAMPLES ARE PROVIDED IN THE REPOSITORY `eval.sh`.
```

## Acknowledgment

The repository is modified from the [CRNet](https://github.com/Kylin9511/CRNet) open source code. Please refer to it if you are interested. 
Thank Zhilin Lu, Sijie Ji, and Chao-Kai Wen for their open source repositories [CRNet](https://github.com/Kylin9511/CRNet), [CLNet](https://github.com/SIJIEJI/CLNet), and [CsiNet](https://github.com/sydney222/Python_CsiNet), which are greatly helpful in establishing this work.
