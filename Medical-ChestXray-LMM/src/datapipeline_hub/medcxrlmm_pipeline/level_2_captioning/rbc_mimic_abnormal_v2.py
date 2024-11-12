import os
from glob import glob
from tqdm import tqdm
import json

import numpy as np
import cv2
from PIL import Image

from utils_captioning import convert_mask_json2numpy, convert_box_txt2numpy
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
    "Spine": 8,  # ChestX -> TODO: VinDR does not have diseases regarding anatomy
    "Heart": 9,  # CheXmask
}
id2label_dict = {}
for k, v in category_ids.items():
    id2label_dict[str(v)] = k

THRESH_DETECTION = 0.5

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


def find_location(conf, xyxy, masks, disease_name, anatomies_matched, masks0, file_name):
    location_info = {}
    location_info["disease"] = {
        "name": disease_name,
        "box": xyxy.tolist(),
        "confidence": conf
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
    return location_info, masks


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
    

def config_mimic():
    dirpath_image = "/mnt/12T/01_hieu/VLM/data/2019.MIMIC-CXR-JPG/2.0.0/files"
    dirpath_pred = "/mnt/12T/02_duong/data-center/Data-Large-MultiModal-Models/MIMIC_CXR/MIMIC_MedGLaMM"
    filepaths_seg = glob(os.path.join(dirpath_pred, "p10/*/*/*.json"))
    return dirpath_image, filepaths_seg


def main():
    dirpath_image, filepaths_seg = config_mimic()

    paths_file_pair = []
    for filepath_seg in tqdm(filepaths_seg):
        index_p = filepath_seg.split("/")[-4]
        filepath_image = os.path.join(dirpath_image, f"{index_p}", filepath_seg.split(f"/{index_p}/")[-1].replace(".json", ".jpg"))
        filepath_det = filepath_seg.replace("MIMIC_MedGLaMM", "MIMIC_disease_YOLO_v5").replace(".json", ".txt").replace(f"/{index_p}/", f"/best_fold_0_mAP_0/{index_p}/")
        assert os.path.exists(filepath_image), filepath_image        
        assert os.path.exists(filepath_det), filepath_det
        paths_file_pair.append([filepath_image, filepath_det, filepath_seg])
    
    for path_image, path_file_det, path_file_seg in tqdm(paths_file_pair):
        # try:
            array = np.array(Image.open(path_image).convert(mode="RGB"))
            image0 = np.transpose(array, [2, 0, 1])
            boxes = convert_box_txt2numpy(path_file_det)
            if boxes is None:
                continue
            masks0, masks = convert_mask_json2numpy(path_file_seg)

            h, w = image0.shape[1:]
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
            test_visual(image0, masks_merged, data, image_name=path_image.split("/")[-1])
            path_caption_save = os.path.join(path_file_seg.replace("MIMIC_MedGLaMM", "MIMIC_MedGLaMM_caption"))
            os.makedirs(os.path.dirname(path_caption_save), exist_ok=True)
            with open(path_caption_save, "w") as f:
                json.dump(data, f, indent=4)
        # except Exception as e:
        #     print(e, path_image)
        #     continue


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
    dirpath_test = f"/mnt/12T/02_duong/Medical-ChestXray-Dataset-for-LMM/tmp/rbc_mimic_abnormal_v2/{image_name.split('.')[0]}"
    os.makedirs(dirpath_test, exist_ok=True)
    visualization.save(os.path.join(dirpath_test, image_name))


if __name__ == "__main__":
    main()
