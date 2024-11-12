import os
import json
import shutil
from tqdm import tqdm

import numpy as np
from pycocotools import mask


def main_1():
    path = "/mnt/12T/02_duong/Large-Multimodal-Models-Wrapper/tmp/ms_cxr_diseases/_annotations.coco.json"
    path_image = "/mnt/12T/01_hieu/VLM/data/2019.MIMIC-CXR-JPG/2.0.0"
    with open(path, "r") as f:
        data = json.load(f)
    
    for anno in tqdm(data["annotations"]):
        image_id = anno["image_id"]
        for image in data["images"]:
            if image["id"] == image_id:
                image_path = os.path.join(path_image, image["path"])
                assert os.path.exists(image_path), image_path
                shutil.copy(image_path, os.path.dirname(path))

category_ids = {
    "Left Lung": 0,
    "Right Lung": 1,
    "Heart": 2,
}


def main():
    import pandas as pd

    path_image = "/mnt/12T/01_hieu/VLM/data/2019.MIMIC-CXR-JPG/2.0.0"

    path_diseases = "/mnt/12T/02_duong/Large-Multimodal-Models-Wrapper/tmp/ms_cxr_diseases/_annotations.coco.json"
    with open(path_diseases, "r") as f:
        data = json.load(f)
    filename_diseases = {}
    for image in tqdm(data["images"]):
        file_name = image["file_name"].split(".")[0]
        file_path = image["path"]
        filename_diseases[file_name] = file_path
    
    path_chexmask = "/mnt/12T/02_duong/data-center/medical-segmentation/physionet.org/files/chexmask-cxr-segmentation-data/0.4/OriginalResolution/MIMIC-CXR-JPG.csv"
    df = pd.read_csv(path_chexmask)

    coco_format = get_coco_json_format()
    coco_format["categories"] = create_category_annotation(category_ids)

    coco_format["images"] = []
    coco_format["annotations"] = []
    
    image_id = 0
    base_ann_id = 0

    for _, row in tqdm(df.iterrows(), total=len(df)):
        row = row.to_dict()
        dicom_id = row["dicom_id"]
        left_lung_rle = row["Left Lung"]
        right_lung_rle = row["Right Lung"]
        heart_rle = row["Heart"]
        height = row["Height"]
        width = row["Width"]
        if dicom_id in filename_diseases.keys():
            image_path = os.path.join(path_image, filename_diseases[dicom_id])
            assert os.path.exists(image_path), image_path
            shutil.copy(image_path, "tmp/ms_cxr_anatomies")
            
            left_lung_array = get_mask_from_RLE(left_lung_rle, height, width)
            right_lung_array = get_mask_from_RLE(right_lung_rle, height, width)
            heart_array = get_mask_from_RLE(heart_rle, height, width)

            concat_array = np.concatenate([left_lung_array[np.newaxis, :], 
                                           right_lung_array[np.newaxis, :], 
                                           heart_array[np.newaxis, :]], axis=0)
            concat_array[concat_array == 255] = 1
            
            image_info = {
                "id": image_id, 
                "file_name": dicom_id + ".jpg",
                "height": height,
                "width": width
            }
            coco_format["images"].append(image_info)
            anno_info, base_ann_id = mask_to_annotation(concat_array, img_id=image_id, base_ann_id=base_ann_id)
            coco_format["annotations"].extend(anno_info)
            image_id += 1

    with open("tmp/ms_cxr_anatomies/_annotations.coco.json", "w") as f:
        json.dump(coco_format, f, indent=4)


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

    polygon =  sv.mask_to_polygons(mask)
    assert len(polygon) == 1, len(polygon)
    polygon = polygon[0].flatten().tolist()
    return polygon


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
        new_ann_id = base_ann_id + i
        annotation = {
            'id': new_ann_id,
            'image_id': img_id,
            'category_id': i,
            'segmentation': polygon,
            'area': int(np.sum(binary_mask)),
            'bbox': toBox(mask_encoded).tolist(),
            'iscrowd': 0  # Set to 1 if the mask represents a crowd region
        }
        annotations.append(annotation)
    return annotations, new_ann_id + 1


if __name__ == "__main__":
    main_1()
    main()
