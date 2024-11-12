import os
import json
from tqdm import tqdm
from dotmap import DotMap

import numpy as np
import skimage
import torch
import torchvision
import matplotlib.pyplot as plt
import torchxrayvision as xrv
from torch.nn import functional as F
from pycocotools import mask

os.environ["CUDA_VISIBLE_DEVICES"] = '1'


names = ['Left Clavicle', 'Right Clavicle', 'Left Scapula', 'Right Scapula',
        'Left Lung', 'Right Lung', 'Left Hilus Pulmonis', 'Right Hilus Pulmonis',
        'Heart', 'Aorta', 'Facies Diaphragmatica', 'Mediastinum',  'Weasand', 'Spine']

category_ids = {
    "Left Clavicle": 0,
    "Right Clavicle": 1,
    "Left Scapula": 2,
    "Right Scapula": 3,
    "Left Lung": 4,
    "Right Lung": 5,
    "Left Hilus Pulmonis": 6,
    "Right Hilus Pulmonis": 7,
    "Heart": 8,
    "Aorta": 9,
    "Facies Diaphragmatica": 10,
    "Mediastinum": 11,
    "Weasand": 12,
    "Spine": 13
}


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


def mask_to_annotation(mask: np.array, base_ann_id: int = 1, img_id: int = 1) -> list:
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
        mask_encoded = binary_mask_to_rle(binary_mask)
        annotation = {
            'id': base_ann_id + i,
            'image_id': img_id,
            'category_id': i,
            'segmentation': mask_encoded,
            'area': int(np.sum(binary_mask)),
            'bbox': toBox(mask_encoded).tolist(),
            'iscrowd': 0  # Set to 1 if the mask represents a crowd region
        }
        annotations.append(annotation)
    return annotations


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


def resize_to_numpy(segmentation, file_size):
    pred = segmentation.float()
    pred = F.interpolate(pred.unsqueeze(0), file_size, mode='nearest')      
    pred = pred[0].bool().to('cpu').numpy()
    return pred 


def export_prediction_as_numpy(mask, out_path) -> None:
    np.save(out_path, mask)


def main_1():
    model = xrv.baseline_models.chestx_det.PSPNet()

    path_dir_images = "/mnt/12T/01_hieu/VLM/data/2019.MIMIC-CXR-JPG/2.0.0"
    path_diseases = "/mnt/12T/02_duong/Large-Multimodal-Models-Wrapper/tmp/ms_cxr_diseases/_annotations.coco.json"
    path_dest = "/mnt/12T/02_duong/data-center/Data-Large-MultiModal-Models/MS_CXR/MS_CXR_anatomy_PSPNet"
    dirpath_visual = "/mnt/12T/02_duong/data-center/Data-Large-MultiModal-Models/MS_CXR/MS_CXR_anatomy_PSPNet_visual"

    with open(path_diseases, "r") as f:
        data = json.load(f)
    for image in tqdm(data["images"]):
        try:
            file_path = os.path.join(path_dir_images, image["path"])

            img = skimage.io.imread(file_path)
            img = xrv.datasets.normalize(img, 255) # convert 8-bit image to [-1024, 1024] range
            image_size = img.shape
            img = img[None, ...] # Make single color channel

            transform = torchvision.transforms.Compose([
                xrv.datasets.XRayCenterCrop(),
                xrv.datasets.XRayResizer(512)
                ])

            img = transform(img)
            img = torch.from_numpy(img)

            with torch.no_grad():
                pred = model(img)

            pred = 1 / (1 + np.exp(-pred))  # sigmoid
            pred[pred < 0.5] = 0
            pred[pred > 0.5] = 1

            pred = pred.squeeze(0)
            pred_resize = resize_to_numpy(
                segmentation=pred,
                file_size=image_size
            )

            # pred = pred.squeeze(0)
            # spine_mask = (pred[-1] == 1).float()
            # spine_mask = F.interpolate(spine_mask.unsqueeze(0).unsqueeze(0), 
            #                            size=image_size).squeeze(1)
            
            dirpath_out = os.path.join(path_dest, os.path.dirname(image["path"]))
            os.makedirs(dirpath_out, exist_ok=True)
            filepath_out = os.path.join(dirpath_out, os.path.basename(image["path"]).replace(".jpg", ".json"))
            export_prediction_as_json(
                mask=pred_resize,
                out_path=filepath_out,
                base_ann_id=1
            )

            # filepath_out = os.path.join(dirpath_out, os.path.basename(image["path"]).replace(".jpg", ".npy"))
            # export_prediction_as_numpy(
            #     mask=pred_resize,
            #     out_path=filepath_out,
            # )

            # dirpath_visual_sample = os.path.join(dirpath_visual, os.path.dirname(image["path"])) 
            # os.makedirs(dirpath_visual_sample, exist_ok=True)
            # import sys
            # sys.path.append("/mnt/12T/02_duong/Large-Multimodal-Models-Wrapper/src")
            # from model_hub.CXAS import (
            #     visualize_from_file
            # )
            # visualize_from_file(
            #     class_names=[key for key in category_ids.keys()], 
            #     category_ids=category_ids,
            #     img_path=os.path.join(path_dir_images, image["path"]), 
            #     label_path=filepath_out, 
            #     img_size=512, 
            #     cat=True, 
            #     axis=1, 
            #     do_store=True,
            #     out_dir=dirpath_visual_sample,
            # )
        except Exception as e:
            print(e, image["path"])
            continue


def main_2():
    model = xrv.baseline_models.chestx_det.PSPNet()
    model = model.to("cuda")

    dirpath_images = "/mnt/12T/01_hieu/VLM/data/2019.MIMIC-CXR-JPG"
    filepath_metadata = "/mnt/12T/02_duong/Large-Multimodal-Models-Wrapper/src/data_pipeline_hub/COMG_Anatomy_pipeline/MIMIC_meta_data.json"
    with open(filepath_metadata) as file:
        data = json.load(file)
    path_dest = "/mnt/12T/02_duong/data-center/Data-Large-MultiModal-Models/MIMIC_CXR/MIMIC_anatomy_PSPNet"
    os.makedirs(path_dest, exist_ok=True)

    for image in tqdm(data["train"]):
        try:
            file_path = os.path.join(dirpath_images, image["Path"])
            img = skimage.io.imread(file_path)
            img = xrv.datasets.normalize(img, 255) # convert 8-bit image to [-1024, 1024] range
            image_size = img.shape
            img = img[None, ...] # Make single color channel

            transform = torchvision.transforms.Compose([
                xrv.datasets.XRayCenterCrop(),
                xrv.datasets.XRayResizer(512)
                ])

            img = transform(img)
            img = torch.from_numpy(img)
            img = img.to("cuda")

            with torch.no_grad():
                pred = model(img)

            pred = pred.detach().cpu()
            pred = 1 / (1 + np.exp(-pred))  # sigmoid
            pred[pred < 0.5] = 0
            pred[pred > 0.5] = 1

            pred = pred.squeeze(0)
            pred_resize = resize_to_numpy(
                segmentation=pred,
                file_size=image_size
            )

            dirpath_out = os.path.join(path_dest, os.path.dirname(image["Path"]))
            os.makedirs(dirpath_out, exist_ok=True)
            filepath_out = os.path.join(dirpath_out, os.path.basename(image["Path"]).replace(".jpg", ".json"))
            export_prediction_as_json(
                mask=pred_resize,
                out_path=filepath_out,
                base_ann_id=1
            )
        except Exception as e:
            print(e, image["Path"])
            continue


def main_3():
    import cv2

    import sys
    sys.path.append("/mnt/12T/02_duong/Medical-ChestXray-Dataset-for-LMM/src")
    from datapipeline_hub.datasets.VinDr_CXR.load_data import load_dataset_vin

    def config_vindr(split):
        data_meta = load_dataset_vin(path="/mnt/12T/02_duong/data-center/VinDr/train.csv")
        path_images_dir = f"/mnt/12T/02_duong/data-center/VinDr/{split}_png_16bit"

        path_fix = "/mnt/12T/02_duong/data-center/Data-Large-MultiModal-Models/VinDr_CXR"
        path_root_save = os.path.join(path_fix, "VinDr_anatomy_PSPNet", f"{split}_png_16bit")
        path_root_save_visual = os.path.join(path_fix, "VinDr_anatomy_PSPNet_visual", f"{split}_png_16bit")

        os.makedirs(path_root_save, exist_ok=True)
        os.makedirs(path_root_save_visual, exist_ok=True)

        return data_meta, path_images_dir, path_root_save, path_root_save_visual    
    
    model = xrv.baseline_models.chestx_det.PSPNet()
    model = model.to("cuda")
    
    data_split, path_images_dir, path_root_save, path_root_save_visual = config_vindr("train")

    data_split = data_split[len(data_split)//2:]

    for sample in tqdm(data_split):
        try:
            sample = DotMap(sample)
            file_path = os.path.join(path_images_dir, sample.Path)
            # img = skimage.io.imread(file_path)
            array = cv2.imread(file_path, cv2.COLOR_BGR2RGB)
            img = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)[:, :, 0]
            img = xrv.datasets.normalize(img, 255) # convert 8-bit image to [-1024, 1024] range
            image_size = img.shape
            img = img[None, ...] # Make single color channel

            transform = torchvision.transforms.Compose([
                xrv.datasets.XRayCenterCrop(),
                xrv.datasets.XRayResizer(512)
                ])

            img = transform(img)
            img = torch.from_numpy(img)
            img = img.to("cuda")

            with torch.no_grad():
                pred = model(img)

            pred = pred.detach().cpu()
            pred = 1 / (1 + np.exp(-pred))  # sigmoid
            pred[pred < 0.5] = 0
            pred[pred > 0.5] = 1

            pred = pred.squeeze(0)
            pred_resize = resize_to_numpy(
                segmentation=pred,
                file_size=image_size
            )

            dirpath_out = path_root_save
            os.makedirs(dirpath_out, exist_ok=True)
            filepath_out = os.path.join(dirpath_out, sample.Path.replace(".png", ".json"))
            export_prediction_as_json(
                mask=pred_resize,
                out_path=filepath_out,
                base_ann_id=1
            )

            # filepath_out = os.path.join(dirpath_out, sample.Path.replace(".png", ".npy"))
            # export_prediction_as_numpy(
            #     mask=pred_resize,
            #     out_path=filepath_out,
            # )

            # dirpath_visual_sample = path_root_save_visual
            # os.makedirs(dirpath_visual_sample, exist_ok=True)
            # from model_hub.cxas import (
            #     visualize_from_file
            # )
            # visualize_from_file(
            #     class_names=[key for key in category_ids.keys()], 
            #     category_ids=category_ids,
            #     img_path=os.path.join(path_images_dir, sample.Path), 
            #     label_path=filepath_out, 
            #     img_size=512, 
            #     cat=True, 
            #     axis=1, 
            #     do_store=True,
            #     out_dir=dirpath_visual_sample,
            # )
        except Exception as e:
            print(e, sample.Path)
            continue


if __name__ == "__main__":
    # main_1()
    # main_2()

    main_3()
