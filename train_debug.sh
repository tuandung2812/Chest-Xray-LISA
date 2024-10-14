#!/bin/bash

# SBATCH --job-name=setup_server        # Job name
#SBATCH --output=result_lisa_debug_1.txt      # Output file
#SBATCH --error=error_lisa_debug_1.txt        # Error file
#SBATCH --ntasks=1               # Number of tasks (processes)
#SBATCH --cpus-per-task=8        # Number of CPU cores per task
#SBATCH --mem=16G                 # Memory per node (4 GB)
#SBATCH --gpus=1                 # Number of GPUs per node

# Set the number of threads
export OMP_NUM_THREADS=4
export LD_LIBRARY_PATH=/home/user01/miniconda3/envs/glamm/lib:$LD_LIBRARY_PATH
export PYTHONPATH="mccv:$PYTHONPATH"
# Set the GPU index
# export CUDA_VISIBLE_DEVICES=1
nvidia-smi
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
deepspeed --master_port 24999 train_ds_original.py \
  --version "microsoft/llava-med-v1.5-mistral-7b" \
  --dataset_dir "./dataset" \
  --vision_pretrained "sam_vit_h_4b8939.pth" \
  --dataset "reason_seg" \
  --sample_rates "1" \
  --local_rank 0 \
  --exp_name "lisa-7b"