import os

import dotenv
import torch
import torch.nn as nn
import torchvision

OUTPUT_PATH = "preliminary/sample"

def main():

    if not os.path.isdir(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)

    dotenv.load_dotenv(".env")

    imagenet = torchvision.datasets.ImageNet(os.getenv("IMAGENET_PATH"))
    classes = imagenet.classes
    softmax = nn.Softmax(dim=1)


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

    imagenet = torchvision.datasets.ImageNet(os.getenv("IMAGENET_PATH"),
                                             split="val")
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

        image_i.save(os.path.join(OUTPUT_PATH,f"predict_val{i}.jpg"))

if __name__ == "__main__":
    main()
