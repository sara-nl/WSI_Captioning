#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train_captioning.py \
  --captioning_config_dir models/configs/Capt.yaml \
  --batch_size 950 \
  --devices 1 \
  --num_workers 4 \
  --num_nodes 1 \
  --shuffle True \
  --accelerator gpu \
  --precision 32 \
  --max_epochs 140 \
  --lm_lr 8e-4 \
  --check_val_every_n_epoch 300 \
  --num_sanity_val_steps 0 \
  --context_length 42 \
  --gradient_clip_val 1.0 \
  --num_samples_val 32 \
  --clip_checkpoint ./data/checkpoints/CLIP/epoch_59_step_59.ckpt
