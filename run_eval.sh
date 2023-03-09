#!/bin/bash
root_data=/projects/0/examode/caption_generation/colon

CUDA_VISIBLE_DEVICES=0 python ./eval_lm.py --lmdb_patches_path ${root_data}/embeddings/hipt/hipt_wsi_embeddings/\
  --texts_path ${root_data}/texts/texts/ \
  --captioning_config_dir ./models/configs/Capt.yaml \
  --crossvalidation_path ${root_data}/texts/cross_validation_folds/10_cross_validation.csv  \
  --generate_batch_size 1 \
  --val_fold 9 \
  --temperature 1.0 \
  --context_length 42 \
  --clip_model_path ./checkpoints/CLIP/exp_39_9/epoch_59_step_59.ckpt \
  --load_from_checkpoint ./checkpoints/CAPT/exp_28/epoch_149_step_750.ckpt
  