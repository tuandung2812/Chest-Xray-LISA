import numpy as np
from PIL import Image
import torch
import torchvision

from model_hub.torchxrayvision.build_xrv import xrv_model_registry
from model_hub.torchxrayvision.utils import datasets


class XRV_Handler:
    def __init__(self, context):
        self.initialize(context)

    def initialize(self, context):
        self.model = xrv_model_registry[context.model_id]()
        self.device = torch.device(context.device)
        self.model.to(self.device)
        self.base_size = context.base_size
        self.threshold = context.threshold
    
    def handle(self, data):
        data = self.preprocess(data)
        data = self.inference(data)
        data = self.postprocess(data)
        return data

    def preprocess(self, data):
        image_path = data["image_path"]
        img = np.array(Image.open(image_path).convert(mode="RGB"))
        orig_file_size = img.shape[:2]
        img = datasets.normalize(img, 255) # convert 8-bit image to [-1024, 1024] range
        img = img.mean(2)[None, ...] # Make single color channel

        transform = torchvision.transforms.Compose([
            # datasets.XRayCenterCrop(), 
            datasets.XRayResizer(512)])
        img = transform(img)
        img = torch.from_numpy(img)
        img = img.to(self.device)

        output = {
            "file_size": orig_file_size,
            "pre_data": img
        }
        return output

    @torch.no_grad()
    def inference(self, data):
        data["inference"] = self.model(data["pre_data"])
        return data

    def postprocess(self, data):
        pred = data["inference"].detach().cpu()
        pred = 1 / (1 + np.exp(-pred))  # sigmoid
        pred[pred < self.threshold] = 0
        pred[pred > self.threshold] = 1
        data["post_data"] = pred
        return data
