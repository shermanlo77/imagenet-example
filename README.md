# ImageNet Example

Sherman Lo, 2024
Queen Mary, University of London

Example code of training a neural network on the ImageNet dataset and making
predictions. This code requires one or more Nvidia GPUs. This serves as the
basis for a [blog post](https://blog.hpc.qmul.ac.uk/ddp-imagenet).

## What does the code do?

The main script is `imagenet_train.py`. It trains a  neural network (with
`DistributedDataParallel`) on the ImageNet dataset. With
`DistributedDataParallel,` multiple GPUs can be used. The code also reports the
training error and the time it takes after each epoch. It also reports the
validation error and the total training time.

No checkpointing or saving of the model is implemented. It mainly serves to
benchmark PyTorch code on one or more Nvidia GPUs.

The code is based on `pytorch/vision`'s reference code for classification,
available on their
[GitHub](https://github.com/pytorch/vision/tree/main/references/classification).

## What neural networks are supported?

- AlexNet, a neural network back around 2012. This neural network is suitable
  for commercial-grade GPUs, eg GTX and RTX series.
  [[Pytorch documentation]](https://pytorch.org/vision/main/models/generated/torchvision.models.alexnet.html)
- ConvNeXt, a neural network from 2022. This neural network is larger and is
  only suitable for enterprise GPUs, eg V100, A100, H100.
  [[Pytorch documentation]](https://pytorch.org/vision/main/models/generated/torchvision.models.convnext_large.html#torchvision.models.convnext_large)
  [[Facebook's original code]](https://github.com/facebookresearch/ConvNeXt)

## How to get started?

Install the required packages, virtual environments are recommended.

```shell
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

This code uses the PyTorch wrapper of the ImageNet dataset, see the
[documentation](https://pytorch.org/vision/main/generated/torchvision.datasets.ImageNet.html).
You will need to have access to or download the ImageNet 2012 dataset from their
[website](https://www.image-net.org/).

The scripts `preliminary/*.py` do exploratory analysis on ImageNet and a
pre-trained AlexNet.

## How to run?

Verify the file `config.yaml` which may be modified. You may modify
`imagenet_path` which specifies the location of the ImageNet dataset.

```text
torchrun [--nnodes NNODES] [--nproc-per-node NPROC_PER_NODE] imagenet_train.py
    [--workers N] [--test] model
```

where the positional arguments and options are:

```text
  model               name of model to run, eg alexnet or convnext
```

```text
--nnodes NNODES       Number of nodes
--nproc-per-node NPROC_PER_NODE
                      Number of processes per node; recommend: gpu
--workers N           Number of data loading workers per process
--test                Indicate to do a test run, only uses one batch of data
```
