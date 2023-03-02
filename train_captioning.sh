#!/bin/bash
root_data=/projects/0/examode/caption_generation/colon

CUDA_VISIBLE_DEVICES=0 python train_captioning.py --lmdb_patches_path ${root_data}/embeddings/hipt/hipt_wsi_embeddings/ \
  --crossvalidation_path ${root_data}/texts/cross_validation_folds/10_cross_validation.csv \
  --captioning_config_dir models/configs/Capt.yaml \
  --text_embeddings_path ${root_data}/embeddings/texts/biogpt \
  --text_tokens_path ${root_data}/texts/extracted_tokens/biogpt/token_dict.json \
  --batch_size 950 \
  --val_fold 9 \
  --devices 1 \
  --num_workers 4 \
  --num_nodes 1 \
  --shuffle True \
  --accelerator gpu \
  --precision 32 \
  --max_epochs 80 \
  --vision_lr 6e-5 \
  --lm_lr 6e-4 \
  --check_val_every_n_epoch 80 \
  --num_sanity_val_steps 1 \
  --temperature 1.0 \
  --context_length 42 \
  --gradient_clip_val 1.0 \
  --num_samples_val 32 \
  --clip_checkpoint ${root_data}/checkpoints/CLIP/exp_39_9/epoch_59_step_59.ckpt \
  --save_generations_path ./data/generated_reports_clip_exp_39_1.json
