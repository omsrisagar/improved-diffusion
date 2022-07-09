"""
Generate a large batch of samples from a super resolution model, given a batch
of samples from a regular model from image_sample.py.
"""

import argparse
import os

import blobfile as bf
import numpy as np
import torch
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from argparse import Namespace

from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    NUM_CLASSES,
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)

from improved_diffusion.script_util import write_2images

def sample(args, data, logger, model, diffusion):
    logger.log("creating samples...")
    all_images = []
    all_lowres_images = []
    all_labels = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = next(data)
        low_res = model_kwargs['low_res']
        upsampled = F.interpolate(low_res, (args.image_size, args.image_size), mode="bilinear")
        all_lowres_images.append(upsampled)
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
        ) # between -1 & 1
        # sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        # sample = sample.permute(0, 2, 3, 1) # channels last is preferred for Computer Vision models in Pytorch
        # sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        # all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        all_images.extend([sample.cpu() for sample in gathered_samples])
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
    arr_lowres = th.vstack(all_lowres_images)
    arr_lowres = arr_lowres[: args.num_samples]
    if args.class_cond:
        # label_arr = np.concatenate(all_labels, axis=0)
        label_arr = th.vstack(all_labels)
        label_arr = label_arr[: args.num_samples]
    else:
        label_arr = None
    return arr, arr_lowres, label_arr

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model...")
    model, diffusion = sr_create_model_and_diffusion(
        **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    logger.log("loading data...")
    data = load_data_for_worker(args.base_samples, args.batch_size, args.class_cond)

    sample_dict = Namespace(
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        class_cond=args.class_cond,
        use_ddim=args.use_ddim,
        image_size=args.large_size,
        clip_denoised=args.clip_denoised
    )
    arr, arr_lowres, label_arr = sample(sample_dict, data, logger, model, diffusion)

    arr = torch.vstack([arr_lowres, arr])
    image_path = os.path.join(logger.get_dir(), f"output_{(args.step_num):06d}.jpg")
    write_2images(image_outputs=arr, display_image_num=args.img_disp_nrow, file_name=image_path)

    img_arr_path = os.path.join(logger.get_dir(), f"samples_{(args.step_num):06d}.npz")

    if dist.get_rank() == 0:
        # shape_str = "x".join([str(x) for x in arr.shape])
        # out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        out_path = os.path.join(logger.get_dir(), img_arr_path)
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")


def load_data_for_worker(base_samples, batch_size, class_cond):
    with bf.BlobFile(base_samples, "rb") as f:
        obj = np.load(f)
        image_arr = obj["arr_0"]
        if class_cond:
            label_arr = obj["arr_1"]
    rank = dist.get_rank()
    num_ranks = dist.get_world_size()
    buffer = []
    label_buffer = []
    while True:
        for i in range(rank, len(image_arr), num_ranks):
            buffer.append(image_arr[i])
            if class_cond:
                label_buffer.append(label_arr[i])
            if len(buffer) == batch_size:
                batch = th.from_numpy(np.stack(buffer)).float()
                # batch = batch / 127.5 - 1.0
                # batch = batch.permute(0, 3, 1, 2)
                res = dict(low_res=batch)
                if class_cond:
                    res["y"] = th.from_numpy(np.stack(label_buffer))
                yield res
                buffer, label_buffer = [], []


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        base_samples="",
        model_path="",
        img_disp_nrow=2,
        step_num=1000
    )
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
