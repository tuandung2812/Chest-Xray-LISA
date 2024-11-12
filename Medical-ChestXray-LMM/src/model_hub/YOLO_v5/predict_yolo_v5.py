import numpy as np
import cv2
import torch
from torch.nn import functional as F

from model_hub.yolo_v5.build_yolo_v5 import yolo_v5_registry
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords, xyxy2xywh


class YOLO_v5_Handler:
    def __init__(self, context):
        self.initialize(context)

    def initialize(self, context):
        self.model = yolo_v5_registry[context.model_id](
            checkpoint=context.checkpoint
        )
        self.device = torch.device(context.device)
        self.model.to(self.device)

        self.image_size = context.image_size
        self.conf_thres = context.conf_thres
        self.iou_thres = context.iou_thres
        self.classes = context.classes
        self.agnostic_nms = context.agnostic_nms
        self.threshold = context.threshold
        self.is_mirror = context.is_mirror

    def handle(self, data):
        try:
            data = self.preprocess(data)
            data = self.inference(data)
            data = self.postprocess(data)
            data["status"] = "Success"
            return data
        except Exception as e:
            print(e)
            print(data["path_image"])
            data = {
                "status": "Error"
            }
            return data

    def preprocess(self, data):
        path_image = data["path_image"]
        
        self.img0 = cv2.imread(path_image)  # BGR
        img = letterbox(self.img0, new_shape=self.image_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        if self.is_mirror:
            img = img[..., ::-1].copy()
        img = torch.from_numpy(img).to(self.device)
        img = img.float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        self.img = img.unsqueeze(0)

        output =  {
            "image0": self.img0,
            "data_pre": self.img
        }
        return output

    @torch.no_grad()
    def inference(self, data):
        data["data_inference"] = self.model(data["data_pre"])[0]
        return data

    def postprocess(self, data):
        data_nms = non_max_suppression(
            data["data_inference"], 
            self.conf_thres, 
            self.iou_thres, 
            classes=self.classes, 
            agnostic=self.agnostic_nms
        )

        data_coor = []

        assert len(data_nms) == 1, len(data_nms)
        for det in data_nms:
            gn = torch.tensor(self.img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                if self.is_mirror:
                    a = det[:, 0].clone().detach()
                    det[:, 0] = data["data_pre"].shape[-1] - det[:, 2]
                    det[:, 2] = data["data_pre"].shape[-1] - a

                det[:, :4] = scale_coords(self.img.shape[2:], det[:, :4], self.img0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                    line = [int(cls), *xywh, conf.item()]

                    data_coor.append(line)
                data_visual = reversed(det)
            else:
                data_visual = None

        data["data_coor"] = data_coor
        data["data_visual"] = data_visual

        return data
