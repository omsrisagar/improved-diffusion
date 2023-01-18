import argparse
import os

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
import torchvision
import kornia
import matplotlib.pyplot as plt

from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from improved_diffusion.train_util import parse_resume_step_from_filename
from improved_diffusion.image_datasets import load_data
from argparse import Namespace
from torchvision import utils
from resizer import Resizer
import math

from improved_diffusion.script_util import write_2images

def imshow(input: th.Tensor, normalize=False):
    out: th.Tensor = torchvision.utils.make_grid(input, nrow=5, padding=1, normalize=normalize)
    out_np: np.ndarray = kornia.tensor_to_image(out)
    plt.imshow(out_np)
    plt.axis('off')
    plt.show()

def sample(args, data, logger, model, diffusion, resizers, orig=None):
    logger.log("creating samples...")
    all_images = []
    all_cond_images = []
    all_orig_images = []
    all_labels = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = next(data)
        if orig is not None:
            orig_images = next(orig)
            orig_images = orig_images.to(dist_util.dev())
        model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            resizers=resizers,
            range_t=args.range_t
        ) # between -1 & 1
        # sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        # sample = sample.permute(0, 2, 3, 1) # channels last is preferred for Computer Vision models in Pytorch
        # sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        gathered_cond_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        dist.all_gather(gathered_cond_samples, model_kwargs['ref_img'])
        # all_images.extend([sample.cpu() for sample in gathered_samples])
        # all_cond_images.extend([sample.cpu() for sample in gathered_cond_samples])
        all_images.extend([sample.cpu() for sample in [sample]])
        all_cond_images.extend([sample.cpu() for sample in [model_kwargs['ref_img']]])
        if orig is not None:
            gathered_orig_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_orig_samples, orig_images)
            all_orig_images.extend([sample.cpu() for sample in gathered_orig_samples])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            # all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
            all_labels.extend([labels.cpu() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    # arr = np.concatenate(all_images, axis=0)
    arr = th.vstack(all_images)
    arr = arr[: args.num_samples]
    arr_cond = th.vstack(all_cond_images)
    arr_cond = arr_cond[: args.num_samples]
    if orig is not None:
        arr_orig = th.vstack(all_orig_images)
        arr_orig = arr_orig[: args.num_samples]
    else:
        arr_orig = None
    if args.class_cond:
        # label_arr = np.concatenate(all_labels, axis=0)
        label_arr = th.vstack(all_labels)
        label_arr = label_arr[: args.num_samples]
    else:
        label_arr = None
    return arr, arr_cond, arr_orig, label_arr

# added
def load_reference(data_dir, batch_size, image_size, class_cond=False):
    data = load_data(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size,
        class_cond=class_cond,
        deterministic=True,
        random_flip=False,
    )
    for large_batch, model_kwargs in data:
        model_kwargs["ref_img"] = large_batch
        yield model_kwargs

def main():
    args = create_argparser().parse_args()

    # th.manual_seed(0)

    dist_util.setup_dist()
    logger.configure(dir=args.save_dir)
    # logger.configure()
    args.step_num = parse_resume_step_from_filename(args.model_path)
    logger.log(f"Step num noted as {args.step_num}")

    logger.log("creating model...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("creating resizers...")
    assert math.log(args.down_N, 2).is_integer()

    shape = (args.batch_size, 3, args.image_size, args.image_size)
    shape_d = (args.batch_size, 3, int(args.image_size / args.down_N), int(args.image_size / args.down_N))
    # down = Resizer(shape, 1 / args.down_N).to(next(model.parameters()).device)
    # up = Resizer(shape_d, args.down_N).to(next(model.parameters()).device)
    # resizers = (down, up)
    # resizers = None
    resizers = 'hf_filter'

    logger.log("loading data...")
    data = load_reference(
        args.base_samples,
        args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )

    sample_dict = Namespace(
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        class_cond=args.class_cond,
        use_ddim=args.use_ddim,
        image_size=args.image_size,
        range_t=args.range_t,
        clip_denoised=args.clip_denoised
    )
    arr, arr_cond, _, label_arr = sample(sample_dict, data, logger, model, diffusion, resizers)
    arr = th.vstack([arr_cond, arr])
    image_path = os.path.join(logger.get_dir(), f'N{args.down_N}_t{args.range_t}', f"output_{(args.step_num):06d}.jpg")
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    write_2images(image_outputs=arr, display_image_num=args.img_disp_nrow, file_name=image_path)

    img_arr_path = os.path.join(logger.get_dir(), f'N{args.down_N}_t{args.range_t}', f"samples_{(args.step_num):06d}.npz")
    out_path = img_arr_path

    if dist.get_rank() == 0:
        # shape_str = "x".join([str(x) for x in arr.shape])
        # out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        # out_path = os.path.join(logger.get_dir(), img_arr_path)
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)

    # logger.log("creating samples...")
    # count = 0
    # while count * args.batch_size < args.num_samples:
    #     model_kwargs = next(data)
    #     model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}
    #     sample = diffusion.p_sample_loop(
    #         model,
    #         (args.batch_size, 3, args.image_size, args.image_size),
    #         clip_denoised=args.clip_denoised,
    #         model_kwargs=model_kwargs,
    #         resizers=resizers,
    #         range_t=args.range_t
    #     )
    #
    #     for i in range(args.batch_size):
    #         out_path = os.path.join(logger.get_dir(),
    #                                 f"{str(count * args.batch_size + i).zfill(5)}.png")
    #         utils.save_image(
    #             sample[i].unsqueeze(0),
    #             out_path,
    #             nrow=1,
    #             normalize=True,
    #             range=(-1, 1),
    #         )
    #
    #     count += 1
    #     logger.log(f"created {count * args.batch_size} samples")

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=4,
        down_N=32,
        range_t=0,
        use_ddim=False,
        base_samples="",
        model_path="",
        save_dir="",
        img_disp_nrow=1,
        step_num=0,
        use_fp16=False, # place holder only not used
        save_latents=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()