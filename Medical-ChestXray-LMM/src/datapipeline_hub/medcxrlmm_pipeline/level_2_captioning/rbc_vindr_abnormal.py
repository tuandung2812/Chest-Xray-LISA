import os
from glob import glob
from tqdm import tqdm
import json

import pandas as pd
import cv2
import numpy as np

from rbc_mimic_abnormal import (
    find_location,
    MAPPING_DISEASE_ANATOMY,
    YOLO_LABEL
)


def config_vindr(split="train"):
    path_fix_image = f"/mnt/12T/02_duong/data-center/VinDr/{split}_png_16bit"
    path_fix_pred = "/mnt/12T/02_duong/data-center/Data-Large-MultiModal-Models/VinDr_CXR"
    path_anatomy_seg = os.path.join(path_fix_pred, "VinDr_anatomy_CXAS", f"{split}_png_16bit")
    paths_file_anatomy_seg = glob(os.path.join(path_anatomy_seg, "*.json"))
    data0 = pd.read_csv("/mnt/12T/02_duong/data-center/VinDr/train.csv")
    return path_fix_image, paths_file_anatomy_seg, data0
    


def main():
    path_fix_image, paths_file_anatomy_seg, data0 = config_vindr()

    paths_file_pair = []
    for path_file_seg in tqdm(paths_file_anatomy_seg):
        path_image = os.path.join(path_fix_image, path_file_seg.split("/")[-1].replace(".json", ".png"))
        assert os.path.exists(path_image), path_image        
        paths_file_pair.append([path_image, path_file_seg])

    for path_image, path_file_seg in tqdm(paths_file_pair):
        try:
            image0 = cv2.imread(path_image, cv2.COLOR_BGR2RGB)
            image = cv2.cvtColor(image0, cv2.COLOR_BGR2RGB)
            image = np.transpose(image, [2, 0, 1])

            image_name = os.path.basename(path_image.replace(".png", ""))
            boxes = data0[data0['image_id'] == image_name].squeeze()

            if boxes is None:
                continue
            masks0, masks = convert_mask_json2numpy(path_file_seg)

            h, w = image.shape[1:]
            impression = []
            for disease_class, conf, *xywh in boxes:
                if conf < THRESH_DETECTION:
                    continue
                # convert to int size
                xyxy = xywh2xyxy(np.array(xywh).reshape(1, 4))[0]
                xyxy[[0, 2]] *= w
                xyxy[[1, 3]] *= h
                xyxy = xyxy.astype(int)
                # find location
                disease_name = YOLO_LABEL[int(disease_class)]
                anatomies_matched = MAPPING_DISEASE_ANATOMY[disease_name]
                locations, masks_merged = find_location(conf, xyxy, masks, disease_name, anatomies_matched, masks0, os.path.basename(path_file_seg).replace(".json", ""))
                if len(locations["anatomies"]) >= 1:
                    impression.append(locations)
            if len(impression) == 0:
                impression = ["Health"]
                continue  # DEBUG

            data = {
                "impression": impression
            }
            # test_visual(image, masks_merged, data, image_name=os.path.basename(path_image))
            if "MIMIC_anatomy_CXAS" in path_file_seg:
                path_caption_save = os.path.join(path_file_seg.replace("MIMIC_anatomy_CXAS", "MIMIC_caption").replace(".json", ".txt"))
            elif "VinDr_anatomy_CXAS" in path_file_seg:
                path_caption_save = os.path.join(path_file_seg.replace("VinDr_anatomy_CXAS", "VinDr_caption"))
            else:
                raise ValueError
            os.makedirs(os.path.dirname(path_caption_save), exist_ok=True)
            with open(path_caption_save, "w") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(e, path_image)
            continue



if __name__ == "__main__":
    main()
