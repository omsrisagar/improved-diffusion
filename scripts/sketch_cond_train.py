"""
Train a super-resolution model.
"""

import argparse

import torch.nn.functional as F

from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import load_data
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from improved_diffusion.train_util import TrainLoop


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_cond_data(
        args.data_dir,
        args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )

    test_data = load_cond_data(
        args.test_dir,
        args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )
    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        test_data=test_data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        image_size=args.image_size,
        sample_interval=args.sample_interval,
        num_samples=args.num_samples,
        use_ddim=args.use_ddim,
        img_disp_nrow=args.img_disp_nrow
    ).run_loop()


def load_cond_data(data_dir, batch_size, image_size, class_cond=False):
    data = load_data(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size,
        class_cond=class_cond,
        cond=True
    )
    for large_batch, model_kwargs in data:
        # model_kwargs["low_res"] = F.interpolate(large_batch, small_size, mode="area")
        yield large_batch, model_kwargs


def create_argparser():
    defaults = dict(
        data_dir="",
        test_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=10,
        save_interval=10000,
        sample_interval=10000,
        num_samples=10000,
        use_ddim=False,
        img_disp_nrow=4,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults()) # large_size, small_size added in this
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
