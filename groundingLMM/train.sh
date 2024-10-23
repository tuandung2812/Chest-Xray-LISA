#!/bin/bash
#!/bin/bash
#!/bin/bash
# SBATCH --job-name=setup_server        # Job name
#SBATCH --output=result_glamm.txt      # Output file
#SBATCH --error=error_glamm.txt        # Error file
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
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# pip install pytorch==1.11.0
# python setup.py develop --no_cuda_ext
# python basicsr/demo.py -opt options/test/SIDD/NAFNet-width64.yml --input_path ./demo/noisy.png --output_path ./demo/denoise_img.png
# DS_BUILD_OPS=1 DS_BUILD_FUSED_ADAM=1 DS_BUILD_AIO=0 DS_SKIP_CUDA_CHECK=1 python -m  pip install deepspeed 
# ds_report
# deepspeed --master_port 60000 train.py --version "MBZUAI/GLaMM-FullScope" --dataset_dir dataset/final_data/ --vision_pretrained "checkpoints/sam_vit_h_4b8939.pth.1" --exp_name "output" --lora_r 8 --lr 3e-4 --pretrained --use_reg_data --use_segm_data --reg_dataset "RefCoco_Reg||RefCocoP_Reg||FlickrGCGVal" --reg_sample_rates "1,1,1" --seg_dataset "Semantic_Segm||Refer_Segm||GranDf_GCG" --segm_sample_rates "1,1,1" --val_dataset "FlickrGCGVal|RefCocoGCGVal|PsgGCGVal" --epochs 10 --steps_per_epoch 500
# DeepSpeed command (customize the arguments as per your needs)
deepspeed --master_port 6000 train.py \
  --version  "MBZUAI/GLaMM-FullScope" \
  --dataset_dir dataset/final_data/ \
  --vision_pretrained "checkpoints/sam_vit_h_4b8939.pth.1" \
  --exp_name "output" \
  --lora_r 8 \
  --lr 3e-4 \
  --pretrained \
  --use_segm_data \
  --seg_dataset "GranDf_GCG" \
  --segm_sample_rates "1" \
  --val_dataset "GranDf_GCG" \
  --epochs 10 \
  --batch_size 8 \
  --steps_per_epoch 500 \
  --mask_validation