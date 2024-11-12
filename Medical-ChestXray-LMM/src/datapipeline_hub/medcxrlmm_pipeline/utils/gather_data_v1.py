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


def load_chexmask_vindr():
    path_chexmask = "/mnt/12T/02_duong/data-center/medical-segmentation/physionet.org/files/chexmask-cxr-segmentation-data/0.4/OriginalResolution/VinDr-CXR.csv"
    df = pd.read_csv(path_chexmask)
    return df


def load_cxas_output_mimic():
    path_cxas_out = "/mnt/12T/02_duong/data-center/Data-Large-MultiModal-Models/MIMIC_CXR/MIMIC_anatomy_CXAS"
    path_dir_images = "/mnt/12T/01_hieu/VLM/data/2019.MIMIC-CXR-JPG/2.0.0/files"

    tmp_part = "p19"
    print(tmp_part)
    paths_pred_file = glob(os.path.join(path_cxas_out, f"{tmp_part}/*/*/*.json"))
    cxas_out = []
    for path_pred in tqdm(paths_pred_file):
        with open(path_pred, "r") as f:
            data = json.load(f)
        path_relative = path_pred.replace(path_cxas_out + "/", "").replace(".json", ".jpg")
        path_image = os.path.join(path_dir_images, path_relative)
        assert os.path.exists(path_image), path_image
        cxas_out.append({
            "path_image": path_image,
            "path_pred": path_pred,
            "data": data
        })
    return cxas_out


def load_cxas_output_vindr():
    path_cxas_out = "/mnt/12T/02_duong/data-center/Data-Large-MultiModal-Models/VinDr_CXR/VinDr_anatomy_CXAS/train_png_16bit"
    path_dir_images = "/mnt/12T/02_duong/data-center/VinDr/train_png_16bit"

    paths_pred_file = glob(os.path.join(path_cxas_out, "*.json"))
    cxas_out = []
    for path_pred in tqdm(paths_pred_file):
        with open(path_pred, "r") as f:
            data = json.load(f)
        path_relative = os.path.basename(path_pred).replace(".json", ".png")
        path_image = os.path.join(path_dir_images, path_relative)
        assert os.path.exists(path_image), path_image
        cxas_out.append({
            "path_image": path_image,
            "path_pred": path_pred,
            "data": data
        })
    return cxas_out


def load_chestx_output():
    pass


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
        # polygon = binary_mask_to_polygon(binary_mask)
        mask_encoded = binary_mask_to_rle(binary_mask)
        bbox = toBox(mask_encoded).tolist()
        new_ann_id = base_ann_id + i
        annotation = {
            'id': new_ann_id,
            'image_id': img_id,
            'category_id': i,
            'segmentation': mask_encoded,
            'area': int(np.sum(binary_mask)),
            'bbox': bbox,
            'iscrowd': 0  # Set to 1 if the mask represents a crowd region
        }
        annotations.append(annotation)
    return annotations, new_ann_id + 1


def get_coors_mask(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)
    xmin = x
    ymin = y
    xmax = x + w
    ymax = y + h
    return xmin, ymin, xmax, ymax


def process_lung_zone(
    mask_left_lung,
    mask_right_lung,
    mask_left_mid_lung,
    mask_right_mid_lung,
):
    xyxy_left_mid_lung = get_coors_mask(mask_left_mid_lung)
    xyxy_right_mid_lung = get_coors_mask(mask_right_mid_lung)
    if xyxy_left_mid_lung is None or xyxy_right_mid_lung is None:
        return None, None, None, None, None, None

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

    mask_zero = np.zeros_like(mask_left_lung)
    mask_zero[xyxy_left_mid_lung[1]: xyxy_left_mid_lung[3], :] = mask_left_lung[xyxy_left_mid_lung[1]: xyxy_left_mid_lung[3], :]
    mask_left_mid_lung = mask_zero.copy()

    mask_zero = np.zeros_like(mask_right_lung)
    mask_zero[xyxy_right_mid_lung[1]: xyxy_right_mid_lung[3], :] = mask_right_lung[xyxy_right_mid_lung[1]: xyxy_right_mid_lung[3], :]
    mask_right_mid_lung = mask_zero.copy()

    return mask_left_mid_lung, mask_right_mid_lung, mask_left_upper_lung, mask_left_lower_lung, mask_right_upper_lung, mask_right_lower_lung


def process_mediastinum(masks):
    cardiomediastinum = masks[115]
    upper_mediastinum = masks[116]
    lower_mediastinum = masks[117]
    anterior_mediastinum = masks[118]
    middle_mediastinum = masks[119]
    posterior_mediastinum = masks[120]
    merged_mask = np.logical_or.reduce([
        cardiomediastinum,  # cardiomediastinum
        upper_mediastinum,  # upper_mediastinum
        lower_mediastinum,  # lower_mediastinum
        anterior_mediastinum,  # anterior_mediastinum
        middle_mediastinum,   # middle_mediastinum
        posterior_mediastinum   # posterior_mediastinum
    ]).astype(int)
    return merged_mask


def export_prediction_as_json(
    mask: np.array,
    out_path: str,
    base_ann_id: int = 1,
) -> None:
        """
        Export prediction as JSON in COCO format.

        Args:
            mask (np.array): Prediction mask.
            outdir (str): Output directory.
            file_name (str): File name.
            img_id (int, optional): Image ID. Defaults to 1.
            base_ann_id (int, optional): Base annotation ID. Defaults to 1.
        """
        coco_format = get_coco_json_format()
        coco_format["categories"] = create_category_annotation(category_ids)

        coco_format["images"] = []
        file_name = out_path.split("/")[-1].split(".")[0]
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

        with open(out_path, "w") as outfile:
            json.dump(coco_format, outfile)
    

def main_1():
    path_dest = "/mnt/12T/02_duong/data-center/Data-Large-MultiModal-Models/MIMIC_CXR/MIMIC_MedGLaMM"
    os.makedirs(path_dest, exist_ok=True)

    cxas_out_data = load_cxas_output_mimic()
    chexmask_data = load_chexmask_mimic()

    # coco_format = get_coco_json_format()
    # coco_format["categories"] = create_category_annotation(category_ids)

    # image_id = 0
    # base_ann_id = 0

    for cxas_data in tqdm(cxas_out_data):
        try:
            file_name = cxas_data["data"]["images"][0]["file_name"]
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
            ## CXAS
            cxas_out_sample = convert_mask_json2numpy(cxas_data["path_pred"])[1]
            aorta = cxas_out_sample[127]
            left_mid_lung = cxas_out_sample[146]
            right_mid_lung = cxas_out_sample[142]
            ## PSPNet
            filepath_pred_pspnet = os.path.join(
                "/mnt/12T/02_duong/data-center/Data-Large-MultiModal-Models/MIMIC_CXR/MIMIC_anatomy_PSPNet/2.0.0/files",
                cxas_data["path_pred"].replace("/mnt/12T/02_duong/data-center/Data-Large-MultiModal-Models/MIMIC_CXR/MIMIC_anatomy_CXAS/", "")
            )
            _, pspnet_out_sample = convert_mask_json2numpy(filepath_pred_pspnet)
            spine = pspnet_out_sample[-1]
            
            # process lung
            left_mid_lung_new, right_mid_lung_new, left_upper_lung, left_lower_lung, right_upper_lung, right_lower_lung = process_lung_zone(
                mask_left_lung=left_lung,
                mask_right_lung=right_lung,
                mask_left_mid_lung=left_mid_lung,
                mask_right_mid_lung=right_mid_lung
            )
            if left_mid_lung_new is None:
                print(f"{cxas_data["path_pred"]} has mid zone is None")
                continue

            # process mediastinum
            mediastinum = process_mediastinum(cxas_out_sample)

            concat_array = np.concatenate(
                [
                    left_upper_lung[np.newaxis, :],
                    left_mid_lung_new[np.newaxis, :],
                    left_lower_lung[np.newaxis, :],
                    right_upper_lung[np.newaxis, :],
                    right_mid_lung_new[np.newaxis, :],
                    right_lower_lung[np.newaxis, :],
                    mediastinum[np.newaxis, :],
                    aorta[np.newaxis, :],
                    spine[np.newaxis, :],
                    heart[np.newaxis, :]
                ], axis=0
            )

            # # convert to coco format
            # image_info = {
            #     "id": image_id, 
            #     "file_name": file_name + ".jpg",
            #     "height": height,
            #     "width": width
            # }
            # coco_format["images"].append(image_info)
            # anno_info, base_ann_id = mask_to_annotation(concat_array, 
            #                                             img_id=image_id, 
            #                                             base_ann_id=base_ann_id)
            # coco_format["annotations"].extend(anno_info)
            # image_id += 1
        
            # shutil.copyfile(cxas_data["path_image"], os.path.join(path_dest, file_name + ".jpg"))

            filepath_relative = os.path.dirname(cxas_data["path_pred"]).replace("/mnt/12T/02_duong/data-center/Data-Large-MultiModal-Models/MIMIC_CXR/MIMIC_anatomy_CXAS/", "")
            dirpath_out = os.path.join(path_dest, filepath_relative)
            os.makedirs(dirpath_out, exist_ok=True)
            filepath_out = os.path.join(dirpath_out, os.path.basename(cxas_data["path_image"]).replace(".jpg", ".json"))
            export_prediction_as_json(
                mask=concat_array,
                out_path=filepath_out,
                base_ann_id=1
            )

        except Exception as e:
            print(e, cxas_data["path_pred"])
            continue
    
    # with open(os.path.join(path_dest, "_annotations.coco.json"), "w") as f:
    #     json.dump(coco_format, f, indent=4)


def main_2():
    path_dest = "/mnt/12T/02_duong/data-center/Data-Large-MultiModal-Models/VinDr_CXR/VinDr_MedGLaMM/train_png_16bit"
    os.makedirs(path_dest, exist_ok=True)

    cxas_out_data = load_cxas_output_vindr()
    chexmask_data = load_chexmask_vindr()

    # coco_format = get_coco_json_format()
    # coco_format["categories"] = create_category_annotation(category_ids)

    # image_id = 0
    # base_ann_id = 0

    for cxas_data in tqdm(cxas_out_data[len(cxas_out_data) // 2 : ]):
        # try:
            file_name = cxas_data["data"]["images"][0]["file_name"]
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
            ## CXAS
            cxas_out_sample = convert_mask_json2numpy(cxas_data["path_pred"])[1]
            aorta = cxas_out_sample[127]
            left_mid_lung = cxas_out_sample[146]
            right_mid_lung = cxas_out_sample[142]
            ## PSPNet
            filepath_pred_pspnet = os.path.join(
                "/mnt/12T/02_duong/data-center/Data-Large-MultiModal-Models/VinDr_CXR/VinDr_anatomy_PSPNet/train_png_16bit",
                cxas_data["path_pred"].replace("/mnt/12T/02_duong/data-center/Data-Large-MultiModal-Models/VinDr_CXR/VinDr_anatomy_CXAS/train_png_16bit/", "")
            )
            _, pspnet_out_sample = convert_mask_json2numpy(filepath_pred_pspnet)
            spine = pspnet_out_sample[-1]
            
            # process lung
            left_mid_lung_new, right_mid_lung_new, left_upper_lung, left_lower_lung, right_upper_lung, right_lower_lung = process_lung_zone(
                mask_left_lung=left_lung,
                mask_right_lung=right_lung,
                mask_left_mid_lung=left_mid_lung,
                mask_right_mid_lung=right_mid_lung
            )
            if left_mid_lung_new is None:
                print(f"{cxas_data["path_pred"]} has mid zone is None")
                continue

            # process mediastinum
            mediastinum = process_mediastinum(cxas_out_sample)

            concat_array = np.concatenate(
                [
                    left_upper_lung[np.newaxis, :],
                    left_mid_lung_new[np.newaxis, :],
                    left_lower_lung[np.newaxis, :],
                    right_upper_lung[np.newaxis, :],
                    right_mid_lung_new[np.newaxis, :],
                    right_lower_lung[np.newaxis, :],
                    mediastinum[np.newaxis, :],
                    aorta[np.newaxis, :],
                    spine[np.newaxis, :],
                    heart[np.newaxis, :]
                ], axis=0
            )

            # # convert to coco format
            # image_info = {
            #     "id": image_id, 
            #     "file_name": file_name + ".jpg",
            #     "height": height,
            #     "width": width
            # }
            # coco_format["images"].append(image_info)
            # anno_info, base_ann_id = mask_to_annotation(concat_array, 
            #                                             img_id=image_id, 
            #                                             base_ann_id=base_ann_id)
            # coco_format["annotations"].extend(anno_info)
            # image_id += 1
        
            # shutil.copyfile(cxas_data["path_image"], os.path.join(path_dest, file_name + ".jpg"))

            filepath_out = os.path.join(path_dest, os.path.basename(cxas_data["path_image"]).replace(".png", ".json"))
            export_prediction_as_json(
                mask=concat_array,
                out_path=filepath_out,
                base_ann_id=1
            )

        # except Exception as e:
        #     print(e, cxas_data["path_pred"])
        #     continue
    
    # with open(os.path.join(path_dest, "_annotations.coco.json"), "w") as f:
    #     json.dump(coco_format, f, indent=4)


if __name__ == "__main__":
    # main_1()
    main_2()
