docker run \
  --rm -it --init \
  --gpus=all \
  --ipc=host \
  --network=host \
  --volume="$PWD/datasets:/home/srikanth/app/datasets" \
  --volume="$PWD/output:/home/srikanth/app/output" \
  -e OPENAI_LOGDIR='obayashi_sketch_cond_32' \
  -e CUDA_VISIBLE_DEVICES='0' \
  obayashi \
mpiexec -n 1 python3 scripts/sketch_cond_train.py --image_size 32 --num_channels 128 --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --use_kl False --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --schedule_sampler uniform --class_cond False --lr 1.00e-4 --batch_size 128 --save_interval 20000 --sample_interval 1000 --num_samples 128 --img_disp_nrow 32 --cond True --data_dir datasets/canny2buildings1a/train/classA --test_dir datasets/canny2buildings1a/test/classA