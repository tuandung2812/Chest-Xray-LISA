import os
from glob import glob
from tqdm import tqdm
import json

import numpy as np
import cv2
from PIL import Image

from utils_captioning import convert_mask_json2numpy, convert_box_txt2numpy
from model_hub.yolo_v5.utils.general import xywh2xyxy
from model_hub.cxas import id2label_dict


YOLO_LABEL = ['Aortic enlargement', 'Atelectasis', 'Calcification', 'Cardiomegaly', 
              'Consolidation', 'ILD', 'Infiltration', 'Lung Opacity', 'Nodule/Mass', 
              'Other lesion', 'Pleural effusion', 'Pleural thickening', 
              'Pneumothorax', 'Pulmonary fibrosis']

NEW_LUNG = [
    "left upper lung",
    "left mid lung",
    "left lower lung",
    "right upper lung",
    "right mid lung",
    "right lower lung"
]


MAPPING_DISEASE_ANATOMY = {
    'Aortic enlargement': ["aorta"],
    'Atelectasis': NEW_LUNG,
    'Calcification': ["heart", "aorta"] + NEW_LUNG,
    'Cardiomegaly': ["heart"],
    'Consolidation': NEW_LUNG,
    'ILD': NEW_LUNG,
    'Infiltration': NEW_LUNG, 
    'Lung Opacity': NEW_LUNG,
    'Nodule/Mass': NEW_LUNG,
    'Other lesion': ["all"],
    'Pleural effusion': NEW_LUNG,
    'Pleural thickening': NEW_LUNG,
    'Pneumothorax': NEW_LUNG,
    'Pulmonary fibrosis': NEW_LUNG,
}

THRESH_DETECTION = 0.5

def get_coco(mask, file_name, base_ann_id=1):
    from model_hub.cxas.utils.create_annotations import (
        get_coco_json_format,
        create_category_annotation,
    )
    from model_hub.cxas.utils.mask_to_coco import mask_to_annotation
    from model_hub.cxas.utils.label_mapper import category_ids

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


def merge_lung(masks):
    mask_left_lung = masks[136]
    mask_right_lung = masks[135]
    mask_left_mid_lung = masks[146]
    mask_right_mid_lung = masks[142]

    xyxy_left_mid_lung = get_coors_mask(mask_left_mid_lung)
    xyxy_right_mid_lung = get_coors_mask(mask_right_mid_lung)

    mask_left_mid_lung[xyxy_left_mid_lung[1]:xyxy_left_mid_lung[3], xyxy_left_mid_lung[0]:xyxy_left_mid_lung[2]] = mask_left_lung[xyxy_left_mid_lung[1]:xyxy_left_mid_lung[3], xyxy_left_mid_lung[0]:xyxy_left_mid_lung[2]]
    mask_right_mid_lung[xyxy_right_mid_lung[1]:xyxy_right_mid_lung[3], xyxy_right_mid_lung[0]:xyxy_right_mid_lung[2]] = mask_right_lung[xyxy_right_mid_lung[1]:xyxy_right_mid_lung[3], xyxy_right_mid_lung[0]:xyxy_right_mid_lung[2]]

    mask_zero = np.zeros_like(mask_left_lung)

    mask_zero[:xyxy_left_mid_lung[1], :] = mask_left_lung[:xyxy_left_mid_lung[1], :]
    mask_left_upper_lung = mask_zero.copy()
    mask_zero.fill(0)
    mask_zero[xyxy_left_mid_lung[3]:, :] = mask_left_lung[xyxy_left_mid_lung[3]:, :]
    mask_left_lower_lung = mask_zero.copy()

    mask_zero.fill(0)
    mask_zero[:xyxy_right_mid_lung[1], :] = mask_right_lung[:xyxy_right_mid_lung[1], :]
    mask_right_upper_lung = mask_zero.copy()
    mask_zero.fill(0)
    mask_zero[xyxy_right_mid_lung[3]:, :] = mask_right_lung[xyxy_right_mid_lung[3]:, :]
    mask_right_lower_lung = mask_zero.copy()

    new_anatomies = np.stack([
        mask_left_upper_lung,
        mask_left_mid_lung,
        mask_left_lower_lung,
        mask_right_upper_lung,
        mask_right_mid_lung,
        mask_right_lower_lung,
    ], axis=0)
    masks_merged = np.concatenate([masks, new_anatomies], axis=0)

    return masks_merged


def find_location(conf, xyxy, masks, disease_name, anatomies_matched, masks0, file_name):
    location_info = {}
    location_info["disease"] = {
        "name": disease_name,
        "box": xyxy.tolist(),
        "confidence": conf
    }
    location_info["anatomies"] = []

    masks = merge_lung(masks)

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
    path_fix_image = "/mnt/12T/01_hieu/VLM/data/2019.MIMIC-CXR-JPG/2.0.0/files"
    path_fix_pred = "/mnt/12T/02_duong/data-center/Data-Large-MultiModal-Models/MIMIC_CXR"
    path_anatomy_seg = os.path.join(path_fix_pred, "MIMIC_anatomy_CXAS")
    paths_file_anatomy_seg = glob(os.path.join(path_anatomy_seg, "p10/*/*/*.json"))
    return path_fix_image, paths_file_anatomy_seg


def main():
    path_fix_image, paths_file_anatomy_seg = config_mimic()

    paths_file_pair = []
    for path_file_seg in tqdm(paths_file_anatomy_seg):
        if "/MIMIC_CXR/" in path_file_seg:
            index_p = path_file_seg.split("/")[-4]
            path_image = os.path.join(path_fix_image, f"{index_p}", path_file_seg.split(f"/{index_p}/")[-1].replace(".json", ".jpg"))
            path_file_det = path_file_seg.replace("MIMIC_anatomy_CXAS", "MIMIC_disease_YOLO_v5").replace(".json", ".txt").replace(f"/{index_p}/", f"/best_fold_0_mAP_0/{index_p}/")
        elif "/VinDr_CXR/" in path_file_seg:
            path_image = os.path.join(path_fix_image, path_file_seg.split("/")[-1].replace(".json", ".png"))
            path_file_det = os.path.join(
                os.path.dirname(path_file_seg).replace("VinDr_anatomy_CXAS", "VinDr_disease_YOLO_v5"), 
                "best_fold_0_mAP_0",
                os.path.basename(path_file_seg).replace(".json", ".txt"))
        else:
            raise ValueError
        assert os.path.exists(path_image), path_image        
        assert os.path.exists(path_file_det), path_file_det
        paths_file_pair.append([path_image, path_file_det, path_file_seg])
    
    for path_image, path_file_det, path_file_seg in tqdm(paths_file_pair):
        try:
            if "MIMIC-CXR" in path_image:
                array = np.array(Image.open(path_image).convert(mode="RGB"))
            elif "/VinDr_CXR/" in path_image:
                array = cv2.imread(path_image, cv2.COLOR_BGR2RGB)
                array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
            else:
                raise ValueError
            
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
        axis=1
    )
    # visualization.save("name_anatomy.png")
    tmp_path_dest = f"tmp/test/{image_name.split('.')[0]}"
    os.makedirs(tmp_path_dest, exist_ok=True)
    cv2.imwrite(f"{tmp_path_dest}/{image_name}", visualization)


if __name__ == "__main__":
    main()
