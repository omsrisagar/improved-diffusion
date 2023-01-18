import torch
import matplotlib.pyplot as plt
import torchvision
import torchvision.utils as tv_utils
import kornia
import numpy as np

from guided_filter import GuidedFilter2d, FastGuidedFilter2d

def imshow(input: torch.Tensor, normalize=False):
    out: torch.Tensor = torchvision.utils.make_grid(input, nrow=5, padding=1, normalize=normalize)
    out_np: np.ndarray = kornia.tensor_to_image(out)
    plt.imshow(out_np)
    plt.axis('off')
    plt.show()

def structure_transferring(radius, eps, fast=False, out_filename="result.png"):
    img = plt.imread("sample_images/toy.bmp")
    mask = plt.imread("sample_images/toy-mask.bmp")[...,0]
    if fast:
        GF = FastGuidedFilter2d(radius, eps, 2)
    else:
        GF = GuidedFilter2d(radius, eps)

    tch_img = torch.from_numpy(img).permute(2, 0, 1)[None].float()
    tch_mask = torch.from_numpy(mask)[None, None].float()
    imshow(tch_img, True)
    imshow(tch_mask, True)

    out = GF(tch_mask, tch_img)
    imshow(out, True)

    tv_utils.save_image(out, out_filename, normalize=True)

def filtering(radius, eps, fast=False, out_filename="result.png"):
    # img = plt.imread("sample_images/cat.bmp")
    img = plt.imread("sample_images/cave-noflash.bmp")
    if fast:
        GF = FastGuidedFilter2d(radius, eps, 2)
    else:
        GF = GuidedFilter2d(radius, eps)

    # tch_img = torch.from_numpy(img)[None, None].float()
    tch_img = torch.from_numpy(img).permute(2, 0, 1)[None].float()
    imshow(tch_img, True)

    out = GF(tch_img, tch_img)
    imshow(out, True)

    tv_utils.save_image(out, out_filename, normalize=True)

def denoising(radius, eps, fast=False, out_filename="result.png"):
    img = plt.imread("sample_images/cave-noflash.bmp")
    guide = plt.imread("sample_images/cave-flash.bmp")
    if fast:
        GF = FastGuidedFilter2d(radius, eps, 2)
    else:
        GF = GuidedFilter2d(radius, eps)

    tch_img = torch.from_numpy(img).permute(2, 0, 1)[None].float()
    tch_guide = torch.from_numpy(guide).permute(2, 0, 1)[None].float()

    out = GF(tch_img, tch_guide)

    tv_utils.save_image(out, out_filename, normalize=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["transferring", "filtering", "denoising"], type=str)
    parser.add_argument("--radius", default=30, type=int)
    parser.add_argument("--eps", default=1e-4, type=float)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--output", default="results.png", type=str)
    args = parser.parse_args()

    if args.task == "transferring":
        structure_transferring(args.radius, args.eps, args.fast, args.output)
    elif args.task == "filtering":
        filtering(args.radius, args.eps, args.fast, args.output)
    else:
        denoising(args.radius, args.eps, args.fast, args.output)
