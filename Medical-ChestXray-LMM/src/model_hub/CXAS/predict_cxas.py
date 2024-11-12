import json

import numpy as np
import cv2
from PIL import Image
import torch
from torch.nn import functional as F

from model_hub.cxas.build_cxas import cxas_model_registry
from model_hub.cxas.utils.create_annotations import (
    get_coco_json_format,
    create_category_annotation,
)
from model_hub.cxas.utils.mask_to_coco import mask_to_annotation
from model_hub.cxas.utils.label_mapper import category_ids


class CXAS_Handler:
    def __init__(self, context):
        self.initialize(context)

    def initialize(self, context):
        self.model = cxas_model_registry[context.model_id]()
        self.device = torch.device(context.device)
        self.model.to(self.device)
        self.base_size = context.base_size
        self.threshold = context.threshold

    def handle(self, data):
        try:
            data = self.preprocess(data)
            data = self.inference(data)
            data = self.postprocess(data)
            data["status"] = "Success"
        except Exception as e:
            print(e)
            print(data["path_image"])
            data = {
                "status": "Error"
            }
            return data
        return data

    def preprocess(self, data):
        image_path = data["path_image"]
        if "MIMIC" in image_path:
            array = np.array(Image.open(image_path).convert(mode="RGB"))
        elif "VinDr" in image_path:
            array = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
            array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        elif "/cxr/" in image_path:
            array = np.array(Image.open(image_path).convert(mode="RGB"))
        else:
            raise ValueError
        array = np.transpose(array, [2, 0, 1])
        original_array = np.copy(array)
        orig_file_size = array.shape[-2:]
        array = torch.tensor(array).float() / 255
        array = normalize(array)
        array = F.interpolate(array.unsqueeze(0), self.base_size)
        array = array.to(self.device)

        output =  {
            "orig_data": original_array,
            "orig_size": orig_file_size,
            "pre_data": array
        }
        return output

    @torch.no_grad()
    def inference(self, data):
        data["inference"] = self.model(data["pre_data"])
        return data

    def postprocess(self, data):
        data["post_data"] = (data["inference"]['logits'].sigmoid() > self.threshold).bool()[0]
        return data


def normalize(array):
    assert array.shape[0] == 3, f"{array.shape}"
    # ImageNet normalization
    # Array to be assumed in range [0,1]
    assert (array.min() >= 0) and (array.max() <= 1)

    array = (array - torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)) / (
        torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    )
    return array


def resize_to_numpy(segmentation, file_size):
    pred = segmentation.float()
    pred = F.interpolate(pred.unsqueeze(0), file_size, mode='nearest')      
    pred = pred[0].bool().to('cpu').numpy()
    return pred 


def export_prediction_as_numpy(mask, out_path) -> None:
    np.save(out_path, mask)


def export_prediction_as_json(
    mask: np.array,
    out_path: str,
    base_ann_id: int = 1,
    filepath: str = None,
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
                "width": mask.shape[2],
                "filepath": filepath
            }
        )
        coco_format["annotations"] = mask_to_annotation(
            mask=mask, base_ann_id=base_ann_id, img_id=img_id
        )

        with open(out_path, "w") as outfile:
            json.dump(coco_format, outfile)