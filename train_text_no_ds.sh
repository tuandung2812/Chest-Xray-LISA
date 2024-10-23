#!/bin/bash

python train_text_no_ds.py \
  --version "microsoft/llava-med-v1.5-mistral-7b" \
  --dataset_dir "./dataset" \
  --vision_pretrained "runs/sam_vit_h_4b8939.pth" \
  --dataset "reason_seg" \
  --sample_rates "1" \
  --local_rank 0 \
  --batch_size 1 \
  --epochs 30 \
  --lr 0.00005 \
  --print_freq 1 \
  --steps_per_epoch 2000 \
   --ce_loss_weight 1.0 \
   --dice_loss_weight 1.0 \
   --bce_loss_weight 1.0 \
   --grad_accumulation_steps 2 \
    --model_max_length 2048 \
   --disable_wandb \
   --exp_name "text_only_medsam_vindr_llavamed_bs_16_lr_5e-5"