import os
from glob import glob
from tqdm import tqdm
import json

import pandas as pd
import numpy as np
import cv2

from utils_captioning import convert_mask_json2numpy
from model_hub.yolo_v5.utils.general import xywh2xyxy


YOLO_LABEL = ['Aortic enlargement', 'Atelectasis', 'Calcification', 'Cardiomegaly', 
              'Consolidation', 'ILD', 'Infiltration', 'Lung Opacity', 'Nodule/Mass', 
              'Other lesion', 'Pleural effusion', 'Pleural thickening', 
              'Pneumothorax', 'Pulmonary fibrosis']

LUNG_ANATOMIES = [
    "Left upper lung",
    "Left middle lung",
    "Left lower lung",
    "Right upper lung",
    "Right middle lung",
    "Right lower lung"
]


MAPPING_DISEASE_ANATOMY = {
    'Aortic enlargement': ["Aorta"],
    'Atelectasis': LUNG_ANATOMIES,
    'Calcification': ["Heart", "Aorta"] + LUNG_ANATOMIES,
    'Cardiomegaly': ["Heart"],
    'Consolidation': LUNG_ANATOMIES,
    'ILD': LUNG_ANATOMIES,
    'Infiltration': LUNG_ANATOMIES, 
    'Lung Opacity': LUNG_ANATOMIES,
    'Nodule/Mass': LUNG_ANATOMIES,
    'Other lesion': ["all"],
    'Pleural effusion': LUNG_ANATOMIES,
    'Pleural thickening': LUNG_ANATOMIES,
    'Pneumothorax': LUNG_ANATOMIES,
    'Pulmonary fibrosis': LUNG_ANATOMIES,
}

category_ids = {
    "Left upper lung": 0,  # CheXmask
    "Left middle lung": 1,  # CXAS
    "Left lower lung": 2,  # CheXmask
    "Right upper lung": 3,  # CheXmask
    "Right middle lung": 4,  # CXAS
    "Right lower lung": 5,  # CheXmask
    "Mediastinum": 6,  # CXAS
    "Aorta": 7,  # CXAS
    "Spine": 8,  # ChestX
    "Heart": 9,  # CheXmask
}
id2label_dict = {}
for k, v in category_ids.items():
    id2label_dict[str(v)] = k


def get_coco(mask, file_name, base_ann_id=1):
    from model_hub.cxas.utils.create_annotations import (
        get_coco_json_format,
        create_category_annotation,
    )
    from model_hub.cxas.utils.mask_to_coco import mask_to_annotation

    coco_format = get_coco_json_format()
    coco_format["categories"] = create_category_annotation(category_ids)

    coco_format["images"] = []
    img_id = file_name

    coco_format["images"].append(
        {
            "id": img_id, 
            "file_name": file_name,
            "height": mask.shape[1],
            "width": mask.shape[2]
        }
    )
    coco_format["annotations"] = mask_to_annotation(
        mask=mask, base_ann_id=base_ann_id, img_id=img_id
    )
    return coco_format


def get_coors_mask(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    xmin = x
    ymin = y
    xmax = x + w
    ymax = y + h
    return xmin, ymin, xmax, ymax


def find_location(xyxy, masks, disease_name, anatomies_matched, file_name, rad_id):
    location_info = {}
    location_info["disease"] = {
        "name": disease_name,
        "box": xyxy.tolist(),
        "rad_id": rad_id
    }
    location_info["anatomies"] = []

    masks0_updated = get_coco(masks, file_name=file_name)

    for name_anatomy in anatomies_matched:
        for id_anatomy, name_anatomy_ in id2label_dict.items():
            id_anatomy = int(id_anatomy)
            if name_anatomy in name_anatomy_:
                mask = masks[int(id_anatomy)]
                if disease_name in ["Cardiomegaly", "Calcification"]:
                    iou_thresh = 0.6
                else:
                    iou_thresh = 0.3 
                is_located, iou = calculate_overlap(xyxy, mask, iou_thresh)
                if is_located:
                    for ann in masks0_updated["annotations"]:
                        if ann["category_id"] == id_anatomy:
                            mask0_anatomy = ann
                            break
                    location_info["anatomies"].append(
                        {
                            "name_anatomy": name_anatomy_,
                            "anatomy_mask": mask0_anatomy,
                            "iou": iou
                        }
                    )
    return location_info


def calculate_overlap(xyxy, mask, iou_thresh):
    """
    IOU = intersection(a disease box, a anatomy mask) / a disease box
    """
    x1, y1, x2, y2 = xyxy
    mask0 = np.zeros_like(mask)
    mask0[y1:y2, x1:x2] = 1  # diseases
    # over lap between (diseases and anatomy) / diseases
    overlap = np.logical_and(mask0, mask)
    iou = np.sum(overlap) / np.sum(mask0)
    if iou > iou_thresh:
        return True, iou
    else:
        return False, iou


def config_vindr(split="train"):
    dirpath_image = f"/mnt/12T/02_duong/data-center/VinDr/{split}_png_16bit"
    dirpath_pred = "/mnt/12T/02_duong/data-center/Data-Large-MultiModal-Models/VinDr_CXR/VinDr_MedGLaMM/train_png_16bit"
    filepaths_seg = glob(os.path.join(dirpath_pred, "*.json"))
    metadata0 = pd.read_csv("/mnt/12T/02_duong/data-center/VinDr/train.csv")
    return dirpath_image, filepaths_seg, metadata0


def main():
    dirpath_image, filepaths_seg, metadata0 = config_vindr()

    filepath_pair = []
    for filepath_seg in tqdm(filepaths_seg):
        filepath_image = os.path.join(dirpath_image, filepath_seg.split("/")[-1].replace(".json", ".png"))
        assert os.path.exists(filepath_image), filepath_image
        filepath_pair.append([filepath_image, filepath_seg])
    
    for filepath_image, filepath_seg in tqdm(filepath_pair):
        try:
            image0 = cv2.imread(filepath_image, cv2.COLOR_BGR2RGB)
            image = cv2.cvtColor(image0, cv2.COLOR_BGR2RGB)
            image = np.transpose(image, [2, 0, 1])

            filename_image = os.path.basename(filepath_image.replace(".png", ""))
            boxes = metadata0[metadata0['image_id'] == filename_image].squeeze()
            if boxes is None:
                continue
            _, masks = convert_mask_json2numpy(filepath_seg)

            impression = []
            for _, box in boxes.iterrows():
                if box.class_name == "No finding":
                    continue
                xyxy = np.array([box['x_min'], box['y_min'], box['x_max'], box['y_max']], dtype=np.int32)
                disease_name = box.class_name
                anatomies_matched = MAPPING_DISEASE_ANATOMY[disease_name]
                locations = find_location(xyxy, masks, disease_name, anatomies_matched, filename_image, rad_id=box.rad_id)
                if len(locations["anatomies"]) >= 1:
                    impression.append(locations)
            data = {
                "impression": impression
            }
            # if len(impression) > 0:
            #     test_visual(image, masks, data, image_name=filepath_image.split("/")[-1])
            filepath_caption = os.path.join(filepath_seg.replace("VinDr_MedGLaMM", "VinDr_MedGLaMM_caption"))
            os.makedirs(os.path.dirname(filepath_caption), exist_ok=True)
            with open(filepath_caption, "w") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(e, filepath_seg)
            continue


def test_visual(image, masks, data, image_name):
    from model_hub.cxas.visualize_cxas import visualize_mask
    from model_hub.yolo_v5.utils.plots import plot_one_box

    anatomy_name_target = [] 
    for disease in data["impression"]: 
        anatomies = disease["anatomies"]
        for anatomy in anatomies:
            anatomy_name_target.append(anatomy["name_anatomy"])

    for disease in data["impression"]:
        disease_name = disease["disease"]["name"]
        final_disease_box_xyxy = disease["disease"]["box"]
        plot_one_box(
            final_disease_box_xyxy,
            np.transpose(image, (1, 2, 0)),
            label=disease_name, 
            color=[255, 0, 255], line_thickness=10
        )

    visualization = visualize_mask(
        class_names=anatomy_name_target, 
        mask=masks, 
        image=image, 
        img_size=512, 
        cat=True, 
        axis=1,
        category_ids=category_ids
    )
    dirpath_test = f"/mnt/12T/02_duong/Medical-ChestXray-Dataset-for-LMM/tmp/rbc_vindr_abnormal_v2/{image_name.split('.')[0]}"
    os.makedirs(dirpath_test, exist_ok=True)
    visualization.save(os.path.join(dirpath_test, image_name))


if __name__ == "__main__":
    main()
