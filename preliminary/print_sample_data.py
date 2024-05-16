import os

import dotenv
import torchvision

OUTPUT_PATH = "preliminary/sample"

def main():

    if not os.path.isdir(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)

    dotenv.load_dotenv(".env")

    imagenet = torchvision.datasets.ImageNet(os.getenv("IMAGENET_PATH"))
    classes = imagenet.classes
    data = [20000, 50000, 70000, 200000]

    for i in data:
        image_i, class_i = imagenet[i]
        class_i = classes[class_i][0].replace(" ", "_")
        image_i.save(os.path.join(OUTPUT_PATH, f"{class_i}.jpg"))

if __name__ == "__main__":
    main()
