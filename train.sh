#!/bin/bash

# SBATCH --job-name=setup_server        # Job name
#SBATCH --output=log_slurm/result_lisa_vindr_debug.txt      # Output file
#SBATCH --error=log_slurm/error_lisa_vindr_debug.txt        # Error file
#SBATCH --ntasks=1               # Number of tasks (processes)
#SBATCH --cpus-per-task=8        # Number of CPU cores per task
#SBATCH --mem=16G                 # Memory per node (4 GB)
#SBATCH --gpus=1                 # Number of GPUs per node

# Set the number of threads
deepspeed --master_port 6000 train.py \
  --version "runs/llavamed" \
  --vision-tower "runs/clip" \
  --dataset_dir "./dataset" \
  --vision_pretrained "runs/medsam.pth" \
  --dataset "reason_seg" \
  --sample_rates "1" \
  --local_rank 0 \
  --batch_size 16 \
  --epochs 30 \
  --lr 0.00005 \
  --print_freq 1 \
  --steps_per_epoch 700 \
   --ce_loss_weight 1.0 \
   --dice_loss_weight 1.5 \
   --bce_loss_weight 1.5 \
   --grad_accumulation_steps 2 \
    --model_max_length 2048 \
    --exp_name "final_medsam_vindr_llavamed_bs_16_lr_5e-5"
