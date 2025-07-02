"""
Predict a sample of ImageNet images using AlexNet and print the results

Predict a sample of ImageNet images using AlexNet and print the true & resulting
predicted class and certainity. Sample of images from the training set, and
validation set are used. Also save those sample images
"""

import os
import yaml

import torch
import torch.nn
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
    softmax = torch.nn.Softmax(dim=1)

    model = torchvision.models.alexnet(weights="DEFAULT")
    model.eval()
    transform = torchvision.models.AlexNet_Weights.DEFAULT.transforms()

    data = [2634, 59934, 800000, 1000000]

    print("=====")
    print("TRAINING SET")
    print("=====")

    for i, i_data in enumerate(data):
        print(f"Image {i}")
        image_i, class_i = imagenet[i_data]
        print(f"True class: {classes[class_i]}")

        prediction = softmax(model(transform(image_i)[None, :]))
        i_predict_class = torch.argmax(prediction)

        print(f"Predict class: {classes[i_predict_class]}")
        print(f"Certainity: {prediction[0, i_predict_class] * 100} %")

        print("=====")

        image_i.save(os.path.join(OUTPUT_PATH, f"predict_training_{i}.jpg"))

    imagenet = torchvision.datasets.ImageNet(imagenet_path, split="val")
    data = [1000, 46719, 3000, 10214, 18540]

    print("=====")
    print("VALIDATION SET")
    print("=====")
    for i, i_data in enumerate(data):
        print(f"Image {i}")
        image_i, class_i = imagenet[i_data]
        print(f"True class: {classes[class_i]}")

        prediction = softmax(model(transform(image_i)[None, :]))
        i_predict_class = torch.argmax(prediction)

        print(f"Predict class: {classes[i_predict_class]}")
        print(f"Certainity: {prediction[0, i_predict_class] * 100} %")

        print("=====")

        image_i.save(os.path.join(OUTPUT_PATH, f"predict_val{i}.jpg"))


if __name__ == "__main__":
    main()
