from matplotlib import pyplot as plt
import cv2
import numpy as np

import torch
import torch.nn as nn
import kornia
import torchvision

def imshow(input: torch.Tensor):
    out: torch.Tensor = torchvision.utils.make_grid(input, nrow=2, padding=1)
    out_np: np.ndarray = kornia.tensor_to_image(out)
    plt.imshow(out_np)
    plt.axis('off')
    plt.show()

img_bgr: np.ndarray = cv2.imread('datasets/canny2buildings1a/test/classB/2.jpg', cv2.IMREAD_COLOR)

x_bgr: torch.Tensor = kornia.image_to_tensor(img_bgr)
x_rgb: torch.Tensor = kornia.color.bgr_to_rgb(x_bgr)
x_rgb = x_rgb.to(0)
x_rgb = x_rgb.float() / 255.

x_rgb = x_rgb.expand(2, -1, -1, -1)  # 4xCxHxW
imshow(x_rgb)

hf_out = kornia.filters.laplacian(x_rgb, kernel_size=1)
imshow(hf_out)
hf_out = kornia.filters.laplacian(x_rgb, kernel_size=3)
imshow(hf_out)
hf_out = kornia.filters.laplacian(x_rgb, kernel_size=5)
imshow(hf_out)
hf_out = kornia.filters.laplacian(x_rgb, kernel_size=7)
imshow(hf_out)
hf_out = kornia.filters.laplacian(x_rgb, kernel_size=9)
imshow(hf_out)
hf_out = kornia.filters.laplacian(x_rgb, kernel_size=11)
imshow(hf_out)
hf_out = kornia.filters.laplacian(x_rgb, kernel_size=13)
imshow(hf_out)
hf_out = kornia.filters.laplacian(x_rgb, kernel_size=15)
imshow(hf_out)
exit(0)

x_grey = kornia.color.rgb_to_grayscale(x_rgb)
imshow(x_grey)
hf_out_grey = kornia.filters.laplacian(x_grey, 3)
imshow(hf_out_grey)