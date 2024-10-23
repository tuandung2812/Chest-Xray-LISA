#!/bin/bash

# SBATCH --job-name=setup_server        # Job name
#SBATCH --output=result_lisa_vindr_debug.txt      # Output file
#SBATCH --error=error_lisa_vindr_debug.txt        # Error file
#SBATCH --ntasks=1               # Number of tasks (processes)
#SBATCH --cpus-per-task=8        # Number of CPU cores per task
#SBATCH --mem=16G                 # Memory per node (4 GB)
#SBATCH --gpus=1                 # Number of GPUs per node

# Set the number of threads
export OMP_NUM_THREADS=4
# export LD_LIBRARY_PATH=/home/user01/miniconda3/envs/glamm/lib:$LD_LIBRARY_PATH
# export PYTHONPATH="mccv:$PYTHONPATH"
export CUDA_HOME=/home/user01/aiotlab/dung_paper/cuda
# Set the GPU index
# export CUDA_VISIBLE_DEVICES=1

# Load any necessary modules or set environment variables here
# For example:
module load cuda/11.4  # Load CUDA if necessary
# module load your_gpu_module

# Activate your conda environment if needed
# source activate my_env

# Run your application or command
# conda env list
# pip install -r requirements.txt
source /home/user01/.bashrc
conda init
echo "Start to activate glamm"
conda activate glamm_2
# conda list
# conda env list
# # pip install -r requirements.txt
# # Check GPU status
nvidia-smi
ds_report
deepspeed --master_port 6000 train.py \
  --version "microsoft/llava-med-v1.5-mistral-7b" \
  --dataset_dir "./dataset" \
  --vision_pretrained "runs/medsam.pth" \
  --dataset "reason_seg" \
  --sample_rates "1" \
  --local_rank 0 \
  --batch_size 8 \
  --epochs 30 \
  --lr 0.00005 \
  --print_freq 1 \
  --steps_per_epoch 1 \
   --ce_loss_weight 1.0 \
   --dice_loss_weight 1.0 \
   --bce_loss_weight 1.0 \
   --grad_accumulation_steps 2 \
    --model_max_length 2048 \
    --exp_name "full_medsam_vindr_llavamed_bs_16_lr_5e-5"