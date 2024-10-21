#!/bin/bash

# SBATCH --job-name=setup_server        # Job name
#SBATCH --output=result_install.txt      # Output file
#SBATCH --error=error_lisa_install.txt        # Error file
#SBATCH --ntasks=1               # Number of tasks (processes)
#SBATCH --cpus-per-task=8        # Number of CPU cores per task
#SBATCH --mem=16G                 # Memory per node (4 GB)
#SBATCH --gpus=1                 # Number of GPUs per node

# Set the number of threads
export OMP_NUM_THREADS=4
export CUDA_HOME=/home/user01/aiotlab/dung_paper/cuda117
export PATH=/home/user01/aiotlab/dung_paper/cuda117/bin:$PATH
export LD_LIBRARY_PATH=/home/user01/aiotlab/dung_paper/cuda117/lib64:$LD_LIBRARY_PATH
export NVCC_APPEND_FLAGS='-allow-unsupported-compiler'
# Set the GPU index
# export CUDA_VISIBLE_DEVICES=1

# Load any necessary modules or set environment variables here
# For example:
module load cuda/11.4  # Load CUDA if necessary
# module load your_gpu_module
conda activate lisa_1

# conda install -c conda-forge cudatoolkit-dev==11.7 --yes
# conda install -c anaconda cudatoolkit=11.4 --yes
# python -m pip uninstall flash-attn --yes
# python -m pip install flash-attn --no-build-isolation

 DS_BUILD_FUSED_ADAM=1 python -m pip install deepspeed