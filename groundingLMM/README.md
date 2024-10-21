# How to run Glamm finetuning code

## Environment setup
Follow the installation steps as the link: https://github.com/mbzuai-oryx/groundingLMM/blob/main/docs/install.md

**Important**
 - Make sure the CUDA version you use to install flash-attn is compatible as the NVCC driver (the code will run as normal, but won't be able to converge)
 
 
 ## Data Preparation
 - Create a dataset/VinDr directory
 - Download VinDr CXR data from Kaggle (dicom format), place them in directory VinDr
     https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/data
 - Convert them to jpg images, save ALL IMAGES (train and test) to dataset/VinDr/image/jpg/all
    + Can reference process_mimic/convert_dicom.py to do this (need to adjust the path)
 - Place the segmentation annotation file at the following path
     + dataset/VinDr/VinDr_MedGLaMM/train_png_16bit
 - Run prepare_vindr.py to process the data (path might need a bit of adjustments)

## Training glamm
  - cd groundingLMM
  - run finetune_glamm_vindr.sh, adjust medsam model path and output_dir.
