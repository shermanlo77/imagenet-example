import os

import dotenv
import torch
import torchvision

MASTER_RANK = 0
SEED = 1735602683889510163

BATCH_SIZE = 32
NUM_WORKERS = 32

def main():

    dotenv.load_dotenv(".env")

    device = torch.device(f"cuda:0")

    # load data
    transform = torchvision.models.AlexNet_Weights.DEFAULT.transforms()
    dataset_test = torchvision.datasets.ImageNet(
        os.environ["IMAGENET_PATH"], split="val", transform=transform)

    # set data loaders
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # set model
    model = torchvision.models.alexnet(weights="DEFAULT")
    model.to(device)

    # test
    n_hit = 0
    model.eval()
    for image, target in data_loader_test:
        image = image.to(device)
        target = target.to(device)
        output = model(image)
        n_hit += torch.sum(torch.argmax(output, 1) == target).cpu()

    print(f"Test accuracy: {n_hit / len(dataset_test) * 100}%")


if __name__ == "__main__":
    main()
