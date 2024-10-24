# Running Glamm and LISA

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

## LISA TRAINING
## Dataset and Dataloader
- The custom dataset of LISA for VinDr data is located at **utils/vindr_datasets.py**
- The collate_fn of LISA is located at **utils/dataset.py** (slight modification from the original)
- LISA training: you can reference **train.sh** for LISA training
    + --version: the LLAVA-backbone of the LMM (recommended: microsoft/llava-med-v1.5-mistral-7b)
    + vision_tower: the vision encoder used to embed the image token into LLAVA's text token space. (recommended:openai/clip-vit-large-patch14)
    + vision_pretrained: the segmentation backbone (recommended: medsam_vit_b.pth: https://drive.google.com/drive/folders/1ETWmi4AiniJeWOt6HAsYgTjYv_fkgzoN)
- LISA inference: I'm having a rough inference code at test.py, we will modify that when the dataset format is clear.
  
## Glamm Training
### Dataset and Dataloader
- The custom dataset of Glamm for VinDr data is located at groundingLMM/dataset/vindr_datasets
- The logic of the dataset is largely based on segm_datasets/Semantic_Segm_ds.py

### Training
- cd groundingLMM
- run *file finetune_glamm_vindr.sh* to start debugging the training process
   + Remember to adjust some checkpoint path (medsam, output_dir).
   + The MedSam model I'm using is medsam_vit_b.pth: https://drive.google.com/drive/folders/1ETWmi4AiniJeWOt6HAsYgTjYv_fkgzoN
  
