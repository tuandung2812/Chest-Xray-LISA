# Setup environment
## Docker supports
- Step 1: Create Docker image
```bash
sh cli/docker_env/build_image.sh
```
- Step 2: Execute Docker container
```bash
sh cli/docker_env/run_container.sh
```
## Non Docker supports
- Step 1: Install dependencies
```bash
sh cli/docker_env/after_docker.sh
```

# Training
**Notes** Ensure directories and file paths are set up corresponding with and without Docker

## 1. Training scripts
```bash
export CUDA_VISIBLE_DEVICES=0
deepspeed \
    --master_port 8888 training_hub/train_lisa/train_lisa.py \
    --filepath-config training_hub/train_lisa/config_lisa/config_lisa_vqa_vindr_medsam_llavamed_docker.yaml \
    --local_rank 0
```

## 2. Global configuration
- Configuration is used by a single file which centralizes all the configuration parameters
- This file is located as in training argument --filepath-config 

## 3. Thirdparty API keys
- This sourcecode requires thirdparty API keys to be set in the environment variables.
- Prepare your thirdparty API keys and set them in the environment variables as follows
```bash
export HF_TOKEN=""
export WANDB_API_KEY=""
```

## 4. Dataloaders
- Directories of datasets including images and labels are set in the configuration file
- Change "dirpath_labels" and "dirpath_images" to your working directory
- The dataset is being stored in A6000 and your PC (/media/hieu/6ac7d369-b609-4b09-97b0-27ed881b25f9/duong/data-center/VinDr/train_png_16bit)

## 5. Models
- Models including tokenizers, vision models and LMMs are from Hugging Face libraries
- Best combination reference is in the configuration file
- Med SAM checkpoint please download from [Med SAM](https://huggingface.co/longformer/longformer-base-4096/tree/main) and locate in the directory of "vision_pretrained" in the configuration file

## 6. Main files
- Dataset and dataloader are defined in src/dataloader_hub/lisa_dataloader
- Model is defined in src/model_hub/lisa
- Training script is defined in training_hub/train_lisa

# References
- [Dung work](https://github.com/tuandung2812/Chest-Xray-LISA/tree/master)
