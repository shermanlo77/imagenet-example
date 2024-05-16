import os

import PIL
import torchvision

OUTPUT_PATH = "preliminary/network"

def main():

    if not os.path.isdir(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)

    image_path = os.path.join("preliminary", "custom_bee.jpg")

    model = torchvision.models.alexnet(weights="DEFAULT")
    model.eval()
    transform = torchvision.models.AlexNet_Weights.DEFAULT.transforms()

    with PIL.Image.open(image_path) as image:

        # STAGE 0
        image = transform(image)
        image = image[None, :]
        output_tensor(image[0], 0, 0)

        # STAGE 1
        for i, layer in enumerate(model.features):
            image = layer(image)
            output_tensor(image[0], 1, i)

        # STAGE 2
        image = model.avgpool(image)
        output_tensor(image[0], 2, 0)


def output_tensor(x, i, j):
    for k, slice in enumerate(x):
        slice_pil = torchvision.transforms.functional.to_pil_image(slice)
        slice_pil.save(os.path.join(OUTPUT_PATH, f"{i}_{j}_{k}.png"))

if __name__ == "__main__":
    main()
