import os

import dotenv
import PIL
import torch
import torch.nn as nn
import torchvision

def main():

    image_paths = [
        os.path.join("preliminary", "custom_hotdog.jpg"),
        os.path.join("preliminary", "custom_cake.jpg"),
        os.path.join("preliminary", "custom_bee.jpg"),
        os.path.join("preliminary", "custom_archery.jpg"),
        os.path.join("preliminary", "custom_ruins.jpg"),
        os.path.join("preliminary", "custom_seagull.jpg"),
    ]
    dotenv.load_dotenv(".env")
    imagenet = torchvision.datasets.ImageNet(os.getenv("IMAGENET_PATH"))
    classes = imagenet.classes
    softmax = nn.Softmax(dim=1)


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
