import sys
sys.path.append(".")

import os
import pandas as pd
import json

from dataset_hub.vindr.config_vindr.config_vindr_medcxrlmm import *


DATA_CAPTION_NO_FINDING = {
    "filepath_image": None,
    "disease_name": None,
    "disease_box": None,
    "rad_id": None,
    "anatomy_name": None,
    "anatomy_mask_id": None,
    "anatomy_mask_category_id": None,
    "anatomy_mask_area": None,
    "anatomy_mask_bbox": None,
    "anatomy_mask_iscrowd": None,
    "anatomy_iou": None,
}


def main():
    df = pd.read_csv(FILEPATH_VINDR_MEDCXRLMM_CAPTION_FILEPATHS)

    data_records = []
    for infex, row in df.iterrows():
        filepath_image = row['filepath_image']
        filepath_label = os.path.join(DIRPATH_VINDR_MEDCXRLMM_CAPTION,  row['filepath_label'])
        with open(filepath_label, "r") as f:
            data_caption = json.load(f)
        if len(data_caption["impression"]) == 0:
            record = DATA_CAPTION_NO_FINDING.copy()
            record["filepath_image"] = filepath_image
            data_records.append(record)
            continue
        for impression in data_caption.get("impression"):
            disease_name = impression.get("disease").get("name")
            disease_box = impression.get("disease").get("box")
            rad_id = impression.get("disease").get("rad_id")
            
            for anatomy in impression.get("anatomies"):
                anatomy_name = anatomy.get("name_anatomy")
                anatomy_iou = anatomy.get("iou")
                anatomy_mask = anatomy.get("anatomy_mask")

                anatomy_mask_id = anatomy_mask.get("id")
                anatomy_mask_category_id = anatomy_mask.get("category_id")
                anatomy_mask_area = anatomy_mask.get("area")
                anatomy_mask_bbox = anatomy_mask.get("bbox")
                anatomy_mask_iscrowd = anatomy_mask.get("iscrowd")
                
                record = {
                    "filepath_image": filepath_image,
                    "disease_name": disease_name,
                    "disease_box": disease_box,
                    "rad_id": rad_id,
                    "anatomy_name": anatomy_name,
                    "anatomy_mask_id": anatomy_mask_id,
                    "anatomy_mask_category_id": anatomy_mask_category_id,
                    "anatomy_mask_area": anatomy_mask_area,
                    "anatomy_mask_bbox": anatomy_mask_bbox,
                    "anatomy_mask_iscrowd": anatomy_mask_iscrowd,
                    "anatomy_iou": anatomy_iou,
                }
                
                data_records.append(record)

    df = pd.DataFrame(data_records)
    df.to_csv(FILEPATH_VINDR_MEDCXRLMM_CAPTION_MEDATADATA, index=False)
            

if __name__ == '__main__':
    main()  