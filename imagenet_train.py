"""Example code, training a neural network with DistributedDataParallel

Example code which trains a neural network (with DistributedDataParallel) on the
ImageNet dataset. With DistributedDataParallel, multiple GPUs can be used. The
code also reports the training error and the time it takes after each epoch. It
also reports the validation error and the total training time.

No checkpointing or saving of the model is implemented. It mainly serves to
benchmark PyTorch code on one or more Nvidia GPUs.

The name of the config file is defined in CONFIG_PATH

This code uses the PyTorch wrapper of the ImageNet dataset, see the
documentation
https://pytorch.org/vision/main/generated/torchvision.datasets.ImageNet.html
You will need to have access to or download the ImageNet 2012 dataset, see
https://www.image-net.org/
Set the path to the dataset the config file

Specify a model you want to train/use in the argument. The available models, as
well as the training parameters like the number of epochs, are defined in the
yaml file defined in CONFIG_PATH and the function get_model()


usage: use with torchrun
       imagenet_train.py [-h] [--workers N] [--test] model

example: torchrun --nproc-per-node gpu --rdzv-backend=c10d \
         --rdzv-endpoint=localhost:0 imagenet_train.py \
         --workers 11 alexnet

positional arguments:
  model        name of model to run

options:
  -h, --help   show this help message and exit
  --workers N  number of data loading workers for each process
  --test       indicate to do a test run, only uses one batch of data
"""

import argparse
import datetime
import logging
import os
import time

import torch
from torch import nn
import torchvision
import yaml


logging.basicConfig(level=logging.INFO)


# the process id to be assigned to be the master
MASTER_RANK = 0
# seed for rng
SEED = 1735602683889510163
# location of the config yaml file
CONFIG_PATH = "config.yaml"
# name of the variable in the config file which contains the path
IMAGENET_CONFIG_NAME = "imagenet_path"


# torchrun variables
RANK = int(os.environ["RANK"])
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])


class DummyLogger:
    """A Logger which does nothing

    A Logger which does nothing, this is used so only one worker does logging
    """

    def __init__(self):
        pass

    def info(self, *args, **kwargs):
        pass


# only the master process does logging
if int(os.environ["RANK"]) == MASTER_RANK:
    LOGGER = logging.getLogger(__name__)
else:
    LOGGER = DummyLogger()


def get_model(args):
    """Return a PyTorch neural network and their corresponding transform

    Args:
        args: the output of argparse.ArgumentParser with argument "model". This
            specifies what model to return. Supported values are "alexnet" and
            "convnext"

    Raises:
        ValueError: when the specified model name is unknown

    Returns:
        model: a PyTorch model which can be trained
        transform: a transforms which can be passed to a dataset which
            pre-process the dataset
    """
    match args.model:
        case "alexnet":
            model = torchvision.models.alexnet()
            transform = torchvision.models.AlexNet_Weights.DEFAULT.transforms()
        case "convnext":
            model = torchvision.models.convnext_base()
            transform = (
                torchvision.models.ConvNeXt_Base_Weights.DEFAULT.transforms()
            )
        case _:
            raise ValueError("Unrecognised model")
    return model, transform


def main(args):
    # load configs; pytorch parameters for each model and location of imagenet
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
        imagenet_path = config[IMAGENET_CONFIG_NAME]
        config = config[args.model]

    # init ddp
    device = torch.device(f"cuda:{LOCAL_RANK}")
    torch.cuda.set_device(device)
    torch.distributed.init_process_group(
        backend="nccl", world_size=WORLD_SIZE, rank=RANK
    )
    torch.distributed.barrier()

    # load model
    model, transform = get_model(args)
    model.to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[device])

    # load imagenet
    dataset = torchvision.datasets.ImageNet(imagenet_path, transform=transform)
    dataset_test = torchvision.datasets.ImageNet(
        imagenet_path, split="val", transform=transform
    )

    # for testing, reduce size of dataset
    if args.test:
        dataset = torch.utils.data.Subset(
            dataset, range(config["batch_size"] * WORLD_SIZE)
        )
        dataset_test = torch.utils.data.Subset(
            dataset_test, range(config["batch_size"] * WORLD_SIZE)
        )

    # set sampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset_test, shuffle=False
    )

    # set data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config["batch_size"],
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=config["batch_size"],
        sampler=test_sampler,
        num_workers=args.workers,
        pin_memory=True,
    )

    # set optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["lr"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config["lr_step_size"], gamma=config["lr_gamma"]
    )
    criterion = nn.CrossEntropyLoss()

    # start training
    LOGGER.info("Training start")

    start_time = time.perf_counter()
    model.train()

    for epoch in range(config["epochs"]):
        epoch_start_time = time.perf_counter()

        train_sampler.set_epoch(epoch)
        total_loss = torch.zeros(1).to(device)

        for image, target in data_loader:
            image = image.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        lr_scheduler.step()

        torch.distributed.reduce(
            total_loss, MASTER_RANK, torch.distributed.ReduceOp.SUM
        )
        total_loss = total_loss.cpu().detach().numpy()[0]

        total_time = time.perf_counter() - epoch_start_time

        LOGGER.info(
            "Epoch %i: loss %f, time %f s", epoch, total_loss, total_time
        )

    LOGGER.info("Training end")
    # report training error and benchmark
    total_time = time.perf_counter() - start_time
    total_time = str(datetime.timedelta(seconds=int(total_time)))
    LOGGER.info("Training time %s", total_time)

    # report validation error
    LOGGER.info("Evaluation start")
    n_hit = torch.zeros(1).to(device)
    model.eval()
    for image, target in data_loader_test:
        image = image.to(device)
        target = target.to(device)
        output = model(image)
        n_hit += torch.sum(torch.argmax(output, 1) == target)

    torch.distributed.reduce(n_hit, MASTER_RANK, torch.distributed.ReduceOp.SUM)
    n_hit = n_hit.cpu().detach().numpy()[0]

    LOGGER.info("Evaluation end")
    LOGGER.info("Test accuracy: %f%%", n_hit / len(dataset_test) * 100)


def get_args_parser():
    parser = argparse.ArgumentParser(
        description="PyTorch Classification Training"
    )
    parser.add_argument("model", help="name of model to run")
    parser.add_argument(
        "--workers",
        default=0,
        type=int,
        metavar="N",
        help="number of data loading workers for each process",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="indicate to do a test run, only uses one batch of data",
    )
    return parser


if __name__ == "__main__":
    main(get_args_parser().parse_args())
