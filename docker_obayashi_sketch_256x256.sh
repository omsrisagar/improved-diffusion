docker run \
  --rm -it --init \
  --gpus=all \
  --ipc=host \
  --network=host \
  --volume="$PWD/datasets:/home/srikanth/app/datasets" \
  --volume="$PWD/output:/home/srikanth/app/output" \
  -e OPENAI_LOGDIR='obayashi_sketch_256x256' \
  -e CUDA_VISIBLE_DEVICES='4,5,6,7' \
  obayashi \
mpiexec -n 4 python3 scripts/image_train.py --image_size 256 --num_channels 128 --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --use_kl False --noise_schedule linear --rescale_learned_sigmas True --rescale_timesteps False --schedule_sampler uniform --class_cond False --lr 1e-5 --batch_size 6 --save_interval 20000 --sample_interval 1000 --num_samples 24 --img_disp_nrow 6 --data_dir datasets/canny2buildings1a/train/classA --resume_checkpoint output/obayashi_sketch_256x256_old/model080000.pt
