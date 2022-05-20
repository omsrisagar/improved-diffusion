docker run \
  --rm -it --init \
  --gpus=all \
  --ipc=host \
  --network=host \
  --volume="$PWD/datasets:/app/datasets" \
  --volume="$PWD/output:/app/output" \
  -e OPENAI_LOGDIR='obayashi' \
  -e CUDA_VISIBLE_DEVICES='0' \
  obayashi \
mpiexec --allow-run-as-root -n 1 python3 scripts/image_train.py --image_size 64 --num_channels 128 --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --use_kl False --noise_schedule linear --rescale_learned_sigmas True --rescale_timesteps False --schedule_sampler uniform --class_cond False --lr 1e-4 --batch_size 4 --save_interval 100 --sample_interval 100 --num_samples 4 --img_disp_nrow 4 --data_dir datasets/canny2buildings1a/test/classB
