"""
Save a sample of images from ImageNet
"""

import os
import yaml

import torchvision

OUTPUT_PATH = "preliminary/sample"
# location of the config yaml file
CONFIG_PATH = "config.yaml"
# name of the variable in the config file which contains the path
IMAGENET_CONFIG_NAME = "imagenet_path"


def main():
    if not os.path.isdir(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
        imagenet_path = config[IMAGENET_CONFIG_NAME]

    imagenet = torchvision.datasets.ImageNet(imagenet_path)
    classes = imagenet.classes
    data = [20000, 50000, 70000, 200000]

    for i in data:
        image_i, class_i = imagenet[i]
        class_i = classes[class_i][0].replace(" ", "_")
        image_i.save(os.path.join(OUTPUT_PATH, f"{class_i}.jpg"))


if __name__ == "__main__":
    main()
