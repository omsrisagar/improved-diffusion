docker run \
  --rm -it --init \
  --gpus=all \
  --ipc=host \
  --network=host \
  --volume="$PWD/datasets:/home/srikanth/app/datasets" \
  --volume="$PWD/output:/home/srikanth/app/output" \
  -e OPENAI_LOGDIR='debug' \
  -e CUDA_VISIBLE_DEVICES='0' \
  obayashi \
mpiexec --allow-run-as-root -n 1 python3 scripts/image_train.py --image_size 64 --num_channels 128 --num_res_blocks 2 --num_heads 1 --learn_sigma False --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --use_kl False --noise_schedule cosine --rescale_learned_sigmas False --rescale_timesteps False --schedule_sampler uniform --class_cond False --lr 5e-5 --batch_size 64 --save_interval 20000 --sample_interval 1000 --num_samples 64 --img_disp_nrow 8 --data_dir datasets/canny2buildings1a/train/classB
