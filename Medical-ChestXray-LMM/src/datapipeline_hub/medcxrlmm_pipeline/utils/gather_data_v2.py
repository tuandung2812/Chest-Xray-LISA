import os
import json
import shutil
from glob import glob
from tqdm import tqdm

import cv2
import pandas as pd
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask


category_ids = {
    "Left lung": 0,  # CheXmask
    "Right lung": 1,  # CheXmask
    "Heart": 2,  # CheXmask
    "Spine": 3,  # ChestX
}


def load_mscxr():
    path_diseases = "/mnt/12T/02_duong/Large-Multimodal-Models-Wrapper/tmp/ms_cxr_diseases/_annotations.coco.json"
    with open(path_diseases, "r") as f:
        data = json.load(f)
    filenames = {}
    for image in tqdm(data["images"]):
        file_name = image["file_name"].split(".")[0]
        file_path = image["path"]
        filenames[file_name] = file_path
    return filenames


def load_chexmask_mimic():
    path_chexmask = "/mnt/12T/02_duong/data-center/medical-segmentation/physionet.org/files/chexmask-cxr-segmentation-data/0.4/OriginalResolution/MIMIC-CXR-JPG.csv"
    df = pd.read_csv(path_chexmask)
    return df


def convert_mask_json2numpy(path):
    with open(path, 'r') as f:
        data0 = json.load(f)
    coco = COCO(path)
    ann_ids = coco.getAnnIds()
    masks = []
    for _, id in enumerate(ann_ids):
        annotation = coco.loadAnns([id])[0]
        mask = coco.annToMask(annotation)
        masks.append(mask)
    masks = np.stack(masks, axis=0)
    return data0, masks


def get_mask_from_RLE(rle, height, width):
    import numpy as np

    runs = np.array([int(x) for x in rle.split()])
    starts = runs[::2]
    lengths = runs[1::2]

    mask = np.zeros((height * width), dtype=np.uint8)

    for start, length in zip(starts, lengths):
        start -= 1  
        end = start + length
        mask[start:end] = 255

    mask = mask.reshape((height, width))
    
    return mask

    
def get_coco_json_format():
    """
    Get the standard COCO JSON format.

    Returns:
        dict: COCO JSON format skeleton.
    """
    # Standard COCO format
    coco_format = {
        "info": {},
        "licenses": [],
        "images": [{}],
        "categories": [{}],
        "annotations": [{}]
    }

    return coco_format


def create_category_annotation(category_dict):
    """
    Create category annotations in COCO JSON format.

    Args:
        category_dict (dict): Dictionary containing category names and IDs.

    Returns:
        list: List of category annotations.
    """
    category_list = []

    for key, value in category_dict.items():
        category = {
            "supercategory": key,
            "id": value,
            "name": key
        }
        category_list.append(category)

    return category_list


def binary_mask_to_polygon(mask):
    import supervision as sv

    polygons =  sv.mask_to_polygons(mask)
    polygons = [polygon.flatten().tolist() for polygon in polygons]
    return polygons


def toBox(binary_mask: np.array) -> list:
    """
    Convert binary mask to bounding box coordinates.

    Args:
        binary_mask (np.array): Binary mask array.

    Returns:
        list: Bounding box coordinates.
    """
    return mask.toBbox(binary_mask)


def binary_mask_to_rle(binary_mask: np.array) -> dict:
    """
    Convert binary mask to COCO RLE format.

    Args:
        binary_mask (np.array): Binary mask array.

    Returns:
        dict: COCO RLE encoded mask.
    """
    mask_encoded = mask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    mask_encoded['counts'] = mask_encoded['counts'].decode('utf-8')
    return mask_encoded


def mask_to_annotation(mask: np.array, base_ann_id: int = 0, img_id: int = 1) -> list:
    """
    Convert mask array to COCO annotation format.

    Args:
        mask (np.array): Mask array.
        base_ann_id (int, optional): Base annotation ID. Defaults to 1.
        img_id (int, optional): Image ID. Defaults to 1.

    Returns:
        list: List of COCO annotations.
    """
    annotations = []
    for i in range(mask.shape[0]):
        # if mask[i].sum() == 0:
        #     continue
        binary_mask = mask[i]
        polygon = binary_mask_to_polygon(binary_mask)
        mask_encoded = binary_mask_to_rle(binary_mask)
        bbox = toBox(mask_encoded).tolist()
        new_ann_id = base_ann_id + i
        annotation = {
            'id': new_ann_id,
            'image_id': img_id,
            'category_id': i,
            'segmentation': polygon,
            'area': int(np.sum(binary_mask)),
            'bbox': bbox,
            'iscrowd': 0  # Set to 1 if the mask represents a crowd region
        }
        annotations.append(annotation)
    return annotations, new_ann_id + 1


def get_coors_mask(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)
    xmin = x
    ymin = y
    xmax = x + w
    ymax = y + h
    return xmin, ymin, xmax, ymax


def main_1():
    path_dest = "/mnt/12T/02_duong/Large-Multimodal-Models-Wrapper/tmp/tmp_gather_data_v2"
    os.makedirs(path_dest, exist_ok=True)
    dirpath_images = "/mnt/12T/01_hieu/VLM/data/2019.MIMIC-CXR-JPG/2.0.0"
    dirpath_preds_pspnet = "/mnt/12T/02_duong/data-center/Data-Large-MultiModal-Models/MS_CXR/MS_CXR_anatomy_PSPNet"

    mscxr_data = load_mscxr()
    chexmask_data = load_chexmask_mimic()

    coco_format = get_coco_json_format()
    coco_format["categories"] = create_category_annotation(category_ids)

    image_id = 0
    base_ann_id = 0

    for file_name, file_path in tqdm(mscxr_data.items()):
        try:
            filepath_image = os.path.join(dirpath_images, file_path)
            assert os.path.exists(filepath_image), filepath_image
            filepath_pred = os.path.join(dirpath_preds_pspnet, file_path.replace(".jpg", ".json"))
            assert os.path.exists(filepath_pred), filepath_pred
            chexmask_sample = chexmask_data[chexmask_data['dicom_id'] == file_name]
            ## CheXmask
            chexmask_sample = chexmask_sample.reset_index(drop=True)
            chexmask_sample = chexmask_sample.to_dict(orient="records")[0]
            height = chexmask_sample["Height"]
            width = chexmask_sample["Width"]
            left_lung_rle = chexmask_sample["Left Lung"]
            right_lung_rle = chexmask_sample["Right Lung"]
            heart_rle = chexmask_sample["Heart"]

            left_lung = get_mask_from_RLE(left_lung_rle, height, width)
            right_lung = get_mask_from_RLE(right_lung_rle, height, width)
            heart = get_mask_from_RLE(heart_rle, height, width)
            left_lung[left_lung == 255] = 1
            right_lung[right_lung == 255] = 1
            heart[heart == 255] = 1

            ## ChestX
            _, pspnet_out_sample = convert_mask_json2numpy(filepath_pred)
            spine = pspnet_out_sample[-1]
            
            concat_array = np.concatenate(
                [
                    left_lung[np.newaxis, :],
                    right_lung[np.newaxis, :],
                    heart[np.newaxis, :],
                    spine[np.newaxis, :],
                ], axis=0
            )

            # convert to coco format
            image_info = {
                "id": image_id, 
                "file_name": file_name + ".jpg",
                "height": height,
                "width": width
            }
            coco_format["images"].append(image_info)
            anno_info, base_ann_id = mask_to_annotation(concat_array, 
                                                        img_id=image_id, 
                                                        base_ann_id=base_ann_id)
            coco_format["annotations"].extend(anno_info)
            image_id += 1
        
            shutil.copyfile(filepath_image, os.path.join(path_dest, file_name + ".jpg"))
                
        except Exception as e:
            print(e, file_path)
            continue

    with open(os.path.join(path_dest, "_annotations.coco.json"), "w") as f:
        json.dump(coco_format, f, indent=4)


def load_vindr():
    filepaths_pred = glob("/mnt/12T/02_duong/data-center/Data-Large-MultiModal-Models/VinDr_CXR/VinDr_MedGLaMM/train_png_16bit/*.json")
    filenames = {}
    for filepath_pred in tqdm(filepaths_pred[:1000]):
        file_name = os.path.basename(filepath_pred).split(".")[0]
        file_path = file_name + ".png"
        filenames[file_name] = file_path
    return filenames


def load_chexmask_vindr():
    path_chexmask = "/mnt/12T/02_duong/data-center/medical-segmentation/physionet.org/files/chexmask-cxr-segmentation-data/0.4/OriginalResolution/VinDr-CXR.csv"
    df = pd.read_csv(path_chexmask)
    return df


def main_2():
    path_dest = "/mnt/12T/02_duong/Medical-ChestXray-Dataset-for-LMM/tmp/VinDr_MedLMM_COCO"
    os.makedirs(path_dest, exist_ok=True)
    dirpath_images = "/mnt/12T/02_duong/data-center/VinDr/train_png_16bit"
    dirpath_preds_pspnet = "/mnt/12T/02_duong/data-center/Data-Large-MultiModal-Models/VinDr_CXR/VinDr_anatomy_PSPNet/train_png_16bit"

    vindr_data = load_vindr()
    chexmask_data = load_chexmask_vindr()

    coco_format = get_coco_json_format()
    coco_format["categories"] = create_category_annotation(category_ids)

    image_id = 0
    base_ann_id = 0

    for file_name, file_path in tqdm(vindr_data.items()):
        try:
            filepath_image = os.path.join(dirpath_images, file_path)
            assert os.path.exists(filepath_image), filepath_image
            filepath_pred = os.path.join(dirpath_preds_pspnet, file_path.replace(".png", ".json"))
            assert os.path.exists(filepath_pred), filepath_pred
            chexmask_sample = chexmask_data[chexmask_data['image_id'] == file_name]
            ## CheXmask
            chexmask_sample = chexmask_sample.reset_index(drop=True)
            chexmask_sample = chexmask_sample.to_dict(orient="records")[0]
            height = chexmask_sample["Height"]
            width = chexmask_sample["Width"]
            left_lung_rle = chexmask_sample["Left Lung"]
            right_lung_rle = chexmask_sample["Right Lung"]
            heart_rle = chexmask_sample["Heart"]

            left_lung = get_mask_from_RLE(left_lung_rle, height, width)
            right_lung = get_mask_from_RLE(right_lung_rle, height, width)
            heart = get_mask_from_RLE(heart_rle, height, width)
            left_lung[left_lung == 255] = 1
            right_lung[right_lung == 255] = 1
            heart[heart == 255] = 1

            ## ChestX
            _, pspnet_out_sample = convert_mask_json2numpy(filepath_pred)
            spine = pspnet_out_sample[-1]
            
            concat_array = np.concatenate(
                [
                    left_lung[np.newaxis, :],
                    right_lung[np.newaxis, :],
                    heart[np.newaxis, :],
                    spine[np.newaxis, :],
                ], axis=0
            )

            # convert to coco format
            image_info = {
                "id": image_id, 
                "file_name": file_name + ".png",
                "height": height,
                "width": width
            }
            coco_format["images"].append(image_info)
            anno_info, base_ann_id = mask_to_annotation(concat_array, 
                                                        img_id=image_id, 
                                                        base_ann_id=base_ann_id)
            coco_format["annotations"].extend(anno_info)
            image_id += 1
        
            shutil.copyfile(filepath_image, os.path.join(path_dest, file_name + ".png"))
                
        except Exception as e:
            print(e, file_path)
            continue

    with open(os.path.join(path_dest, "_annotations.coco.json"), "w") as f:
        json.dump(coco_format, f, indent=4)


if __name__ == "__main__":
    # main_1()
    main_2()
