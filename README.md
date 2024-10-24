# Running Glamm

## Environment setup
- Configure the environment as follows https://github.com/mbzuai-oryx/groundingLMM/blob/main/docs/install.md
- Make sure the cuda version you use to compile flash-attention is compatible with your NVCC driver
    + The code might run normally, but fails to converge if not compatible: https://github.com/haotian-liu/LLaVA/issues/368

## Data Processing
- Download VinDrCxr from kaggle (dicom format): https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/data
- Move them to directory *dataset/VinDr/image*
- Convert dicom images to jpg format by running the code process_mimic/convert_dicom.py 
    + you might need to adjust input_dir and output_dir
    + You need to run this code twice for the train and test set
    + **ALL IMAGES** need to be saved at *LISAMed/dataset/VinDr/image/jpg/all*
- Prepare the VQA and segmentation mask data
    + Place the segmentation data on Drive at VinDr/VinDR_MedGLaMM
    + Run the code *prepare_vindr.py*

## Dataset and Dataloader
- The custom dataset of Glamm for VinDr data is located at groundingLMM/dataset/vindr_datasets
- The logic of the dataset is largely based on segm_datasets/Semantic_Segm_ds.py

## Training
- cd groundingLMM
- run *file finetune_glamm_vindr.sh* to start debugging the training process
   + Remember to adjust some checkpoint path (medsam, output_dir).
   + The MedSam model I'm using is medsam_vit_b.pth: https://drive.google.com/drive/folders/1ETWmi4AiniJeWOt6HAsYgTjYv_fkgzoN
  
