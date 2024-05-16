import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

import os


RANK = int(os.environ["RANK"])
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])

MASTER_RANK = 0

# Number of workers to used in torch.utils.data.DataLoader()
# NSLOTS is specifically for Apocrita but should be edited for other systems
NUM_WORKERS = int(os.getenv("NSLOTS")) // torch.cuda.device_count() - 1

N_EPOCH = 5
BATCH_SIZE = 100


# A simple and small custom neural network on the MNIST dataset
# This is for demonstration and testing only
class Net(nn.Module):

    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


def main():

    device = torch.device(f"cuda:{LOCAL_RANK}")
    torch.cuda.set_device(device)
    torch.distributed.init_process_group(backend="nccl", world_size=WORLD_SIZE,
                                         rank=RANK)
    torch.distributed.barrier()

    model = Net()
    model.to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[device])

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    # The first call of torchvision.datasets.MNIST() will download the dataset
    #
    # We place this condition so only the master process will download the mnist
    # dataset first
    #
    # We place a barrier such that that other processes must wait for the master
    # process to finish downloading the mnist dataset before reading the dataset
    # from file
    if RANK == MASTER_RANK:
        dataset = torchvision.datasets.MNIST(
            root='.', train=True, transform=transform, download=True)
        torch.distributed.barrier()
    else:
        # all remaining processes can read the mnist dataset once the master
        # process finishes downloading the mnist dataset
        torch.distributed.barrier()
        dataset = torchvision.datasets.MNIST(
                root='.', train=True, transform=transform, download=True)

    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)

    model.train()  # set the model to training mode

    for epoch in range(N_EPOCH):

        sampler.set_epoch(epoch)
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

        # gather and sum the loss from each worker
        torch.distributed.reduce(total_loss, MASTER_RANK,
                                 torch.distributed.ReduceOp.SUM)

        # only the master rank prints the loss at every epoch
        if RANK == MASTER_RANK:
            print(f"Total loss: {total_loss[0]}")


if __name__ == "__main__":
    main()
