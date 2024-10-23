#!/bin/bash
# SBATCH --job-name=setup_server        # Job name
#SBATCH --output=result.txt      # Output file
#SBATCH --error=error.txt        # Error file
#SBATCH --time=01:00:00          # Wall time limit (hh:mm:ss)
#SBATCH --ntasks=1               # Number of tasks (processes)
#SBATCH --cpus-per-task=4        # Number of CPU cores per task
#SBATCH --mem=4G                 # Memory per node (4 GB)
#SBATCH --gpus=1                 # Number of GPUs per node

# Load any necessary modules or set environment variables here
# For example:
module load cuda/11.0  # Load CUDA if necessary
# module load your_gpu_module

# Activate your conda environment if needed
# source activate my_env

# Run your application or command
# conda env list
# pip install -r requirements.txt
source /home/user01/.bashrc
conda init
echo "Start to activate anhtn"
conda activate vn-sign
# conda list
# conda env list
# # pip install -r requirements.txt
# # Check GPU status
nvidia-smi

python check_cuda.py
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# pip install pytorch==1.11.0
# python setup.py develop --no_cuda_ext
# python basicsr/demo.py -opt options/test/SIDD/NAFNet-width64.yml --input_path ./demo/noisy.png --output_path ./demo/denoise_img.png
