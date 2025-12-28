import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
import cv2

dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True
)

img1, _ = dataset[0]
img2, _ = dataset[1]

images = [img1, img2]

 -----------------------------

transformations = [
    ("Original", lambda x: x),
    ("Resize", transforms.Resize((64, 64))),
    ("Grayscale", transforms.Grayscale(num_output_channels=1)),
    ("Horizontal Flip", transforms.RandomHorizontalFlip(p=1.0)),
    ("Rotation", transforms.RandomRotation(degrees=30)),
]


def sharpen_image(pil_img):
    img = np.array(pil_img)
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(img, -1, kernel)
    return sharpened

 -----------------------------
fig, axes = plt.subplots(len(transformations) + 1, 2, figsize=(8, 15))

final_tensors = []

for col, img in enumerate(images):
    current_img = img

    for row, (name, transform) in enumerate(transformations):
        if name != "Original":
            current_img = transform(current_img)

        axes[row, col].imshow(current_img, cmap="gray")
        axes[row, col].set_title(f"{name}")
        axes[row, col].axis("off")

        img_array = np.array(current_img)
        print(f"{name} Image {col+1} -> Shape: {img_array.shape}, "
              f"Pixel Range: [{img_array.min()}, {img_array.max()}]")


    sharpened_img = sharpen_image(current_img)
    axes[len(transformations), col].imshow(sharpened_img, cmap="gray")
    axes[len(transformations), col].set_title("Sharpened (Bonus)")
    axes[len(transformations), col].axis("off")


    tensor_img = transforms.ToTensor()(sharpened_img)
    final_tensors.append(tensor_img)

plt.tight_layout()
plt.show()

 -----------------------------

batch_tensor = torch.stack(final_tensors)
print("\nFinal Batch Tensor Shape:", batch_tensor.shape)
print("Final Tensor Pixel Range:",
      batch_tensor.min().item(), batch_tensor.max().item())

