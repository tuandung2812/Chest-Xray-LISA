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
  
