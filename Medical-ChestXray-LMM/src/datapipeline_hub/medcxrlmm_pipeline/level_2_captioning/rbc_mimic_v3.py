import os
import json
import tqdm
import supervision as sv    

import sys
sys.path.append("/mnt/12T/02_duong/Medical-ChestXray-Dataset-for-LMM/src")
from datapipeline_hub.medcxrlmm_pipeline.level_2_captioning.utils_captioning import convert_mask_json2numpy

anatomies_target = {
    "left upper lung": 0,
    "left middle lung": 1, 
    "left lower lung": 2, 
    "right upper lung": 3, 
    "right middle lung": 4,  
    "right lower lung": 5,
    "mediastinum": 6, 
    "aorta": 7,
    "spine": 8,  
    "heart": 9, 
}


def main():
    dirpath_anatomies_segmentation = "/mnt/12T/02_duong/data-center/Data-Large-MultiModal-Models/MIMIC_CXR/MIMIC_MedGLaMM"

    filepath_mimic = "/mnt/12T/02_duong/Large-Multimodal-Models-Wrapper/src/data_pipeline_hub/COMG_Anatomy_pipeline/MIMIC_meta_data.json"
    with open(filepath_mimic) as file:
        data_mimic = json.load(file)
    data_mimic_split = data_mimic["train"] + data_mimic["valid"]
    data_mimic_dict = {}
    for sample in tqdm.tqdm(data_mimic_split):
        dicom_id = sample["dicom_id"]
        data_mimic_dict[dicom_id] = sample

    filepath_disease_anatomy_pairs = "/mnt/12T/02_duong/Medical-ChestXray-Dataset-for-LMM/tmp/mimic_disease_anatomy_pairs.json"
    with open(filepath_disease_anatomy_pairs) as file:
        data_disease_anatomy_pairs = json.load(file)

    for dicom_id, data_mapping in tqdm.tqdm(data_disease_anatomy_pairs.items()):
        try:
            data_mimic = data_mimic_dict[dicom_id]
            filepath_segmentation = os.path.join(dirpath_anatomies_segmentation, data_mimic["Path"].replace("2.0.0/files/", "").replace(".jpg", ".json"))
            with open(filepath_segmentation) as file:
                data_anatomies_segmentation = json.load(file)
            masks0, masks = convert_mask_json2numpy(filepath_segmentation)
            
            impression = []
            for data in data_mapping:
                anatomy_name = data["anatomy"]
                disease_name = data["disease"]

                location_info = {}
                location_info["disease"] = {
                    "disease_name": disease_name,
                }
                location_info["anatomy"] = {}
                if anatomy_name in anatomies_target.keys():
                    for catetory in data_anatomies_segmentation["categories"]:
                        if catetory["supercategory"].lower() == anatomy_name:
                            anatomy_mask = masks[catetory["id"]]
                            polygons =  sv.mask_to_polygons(anatomy_mask)
                            anatomy_polygons = [polygon.flatten().tolist() for polygon in polygons]
                else:
                    anatomy_polygons = None

                location_info["anatomy"] = {
                    "anatomy_name": anatomy_name,
                    "anatomy_polygons": anatomy_polygons,
                }
                impression.append(location_info)
            data = {
                "impression": impression,
            }

            filepath_caption = os.path.join(filepath_segmentation.replace("MIMIC_MedGLaMM", "MIMIC_MedGLaMM_caption_v3"))
            is_exist = os.path.exists(filepath_caption)
            assert not is_exist, filepath_caption
            os.makedirs(os.path.dirname(filepath_caption), exist_ok=True)
            with open(filepath_caption, "w") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Error: {e}")
            continue


if __name__ == "__main__":
    main()
