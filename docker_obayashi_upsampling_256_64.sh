docker run \
  --rm -it --init \
  --gpus=all \
  --ipc=host \
  --network=host \
  --volume="$PWD/datasets:/home/srikanth/app/datasets" \
  --volume="$PWD/output:/home/srikanth/app/output" \
  -e OPENAI_LOGDIR='obayashi_upsampling_256_64' \
  -e CUDA_VISIBLE_DEVICES='0,1,2,3,4,5' \
  obayashi \
mpiexec --allow-run-as-root -n 4 python3 scripts/super_res_train.py --large_size 256 --small_size 64 --num_channels 192 --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --use_kl False --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --schedule_sampler uniform --class_cond False --lr 1.25e-5 --batch_size 4 --save_interval 20000 --sample_interval 1000 --num_samples 24 --img_disp_nrow 12 --data_dir datasets/canny2buildings1a/train/classB
