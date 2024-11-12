# Generating MIMIC QA data
- Run **process_mimic/generate_mimic.py** to create QA-Pairs from the MIMIC caption files
 + Place the **MIMIC_MedGLaMM_caption_v3** folders into **dataset** directory (or change the folder_path variable in **generate_mimic.py**, line 150 accordingly
 + The bug *"It suffers from and"* supposedly comes from line 282 to 294. I fixed it, you can recheck if there's any problems

# Generating VinDr QA data
- Run **process_mimic/generate_mimic.py** to create QA-Pairs from the MIMIC caption files
 + Place the **MIMIC_MedGLaMM_caption_v3** folders into **dataset/VinDr** directory (or change the folder_path variable in **generate_qa_vindr.py**, line 129 accordingly
 + The bug *"It suffers from and"* supposedly comes from line 243 to 255. I fixed it, you can recheck if there's any problems 

# Reformatting and resampling QA data
- Run **process_mimic/reformat_qa.py** and **process_mimic/resample_qa.py**

# Final Preprocessing step
- **VinDr**: run **prepare_vindr.py** and adjust the variable *segment_path* (line 10) to point to the segmentation folder /VinDr_MedGLaMM/train_png_16bit)
  + you can change the output folder of the annotation and the mask with line 13 and 28
- **MIMIC**:run **prepare_mimic.py** and adjust the variable *segment_path* (line 10) to point to the segmentation folder of MIMIC)
  + you can change the output folder of the annotation and the mask with line 13 and 28

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
    + **ALL IMAGES** need to be saved at *aiotlab/dung_paper/groundingLMM/LISAMed/dataset/VinDr/image/jpg/all*
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
  
