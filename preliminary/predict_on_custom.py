"""
Predict the custom images using AlexNet and print the results

Predict the custom images using AlexNet and print the resulting predicted class
and certainity

The custom images are the .jpg files in the "preliminary" folder
"""

import os
import yaml

import PIL
import torch
import torch.nn
import torchvision

# location of the config yaml file
CONFIG_PATH = "config.yaml"
# name of the variable in the config file which contains the path
IMAGENET_CONFIG_NAME = "imagenet_path"


def main():
    image_paths = [
        os.path.join("preliminary", "custom_hotdog.jpg"),
        os.path.join("preliminary", "custom_cake.jpg"),
        os.path.join("preliminary", "custom_bee.jpg"),
        os.path.join("preliminary", "custom_archery.jpg"),
        os.path.join("preliminary", "custom_ruins.jpg"),
        os.path.join("preliminary", "custom_seagull.jpg"),
    ]

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
        imagenet_path = config[IMAGENET_CONFIG_NAME]

    imagenet = torchvision.datasets.ImageNet(imagenet_path)
    classes = imagenet.classes
    softmax = torch.nn.Softmax(dim=1)

    model = torchvision.models.alexnet(weights="DEFAULT")
    model.eval()
    transform = torchvision.models.AlexNet_Weights.DEFAULT.transforms()

    for image_path in image_paths:
        print(f"Image {image_path}")
        with PIL.Image.open(image_path) as image_i:
            prediction = softmax(model(transform(image_i)[None, :]))
            i_predict_class = torch.argmax(prediction)

            print(f"Predict class: {classes[i_predict_class]}")
            print(f"Certainity: {prediction[0, i_predict_class] * 100} %")

        print("=====")


if __name__ == "__main__":
    main()
