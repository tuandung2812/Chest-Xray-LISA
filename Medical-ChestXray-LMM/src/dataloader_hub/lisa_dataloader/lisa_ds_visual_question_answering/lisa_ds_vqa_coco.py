import sys
sys.path.append(".")

import os
import json
import dotmap
import random

import cv2
import torch
from transformers import CLIPImageProcessor

from dataloader_hub.dataloader_base.dataset_base_generic import GenericDatasetBase
from dataloader_hub.dataloader_common.preprocess_image.resize_longest_side import ResizeLongestSide
from dataloader_hub.dataloader_common.preprocess_image.normalize_image import normalize_image
from dataloader_hub.lisa_dataloader.lisa_ds_config import *
from model_hub.llava.llava_0.conversation import conv_templates


class LISAVisualQuestionAnsweringCOCODataset(GenericDatasetBase):
    def __init__(
        self, 
        config
    ):
        super().__init__(config)

    def initialize_dataset_config(self, config):
        self.cfg = dotmap.DotMap(config)
        self.image_transformer = ResizeLongestSide(self.cfg.visual_model_input_size)
        self.image_encoder = CLIPImageProcessor.from_pretrained(self.cfg.hf_visual_model_id)
        self.pixel_mean = torch.Tensor(self.cfg.pixel_mean).view(-1, 1, 1)
        self.pixel_std = torch.Tensor(self.cfg.pixel_std).view(-1, 1, 1)
        self.conversation_template = conv_templates[self.cfg.conversation_template]

    def load_dataset(self):
        with open(self.cfg.filepath_labels) as f:
            data = json.load(f)
        return data
    
    def preprocess_image(self, data):
        image0 = cv2.imread(data["filepath_image"])
        image0 = cv2.cvtColor(image0, cv2.COLOR_BGR2RGB)
        image0_size = image0.shape[:2]
        image_resized = self.image_transformer.apply_image(image0)
        image_resized_size = image_resized.shape[:2]
        image = normalize_image(torch.from_numpy(image_resized).permute(2, 0, 1).contiguous(), self.pixel_mean, self.pixel_std, self.cfg.visual_model_input_size)
        masks = torch.rand(0, *image0_size)
        label = torch.ones(image0_size) * self.ignore_label

        data["image0"] = image0
        data["image0_size"] = image0_size
        data["image_resized"] = image_resized
        data["image_resized_size"] = image_resized_size
        data["image"] = image
        data["masks"] = masks
        data["label"] = label
        return data

    def extract_image_embeddings(self, data):
        data["image_for_llm"] = self.image_encoder.preprocess(data["image0"], return_tensors="pt")["pixel_values"][0]
        return data

    def process_image(self, data):
        data = self.preprocess_image(data)
        data = self.extract_image_embeddings(data)
        return data

    def preprocess_text(self, data, conversation0):
        source = data["text0"]
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence["value"]:
                sentence["value"] = (
                    sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
                )
                sentence["value"] = DEFAULT_IMAGE_TOKEN + "\n" + sentence["value"]
                sentence["value"] = sentence["value"].strip()
                if "mmtag" in conversation0.version:
                    sentence["value"] = sentence["value"].replace(
                        DEFAULT_IMAGE_TOKEN, "<Image>" + DEFAULT_IMAGE_TOKEN + "</Image>"
                    )
        data["text"] = source
        return data

    def form_conversation(self, data, conv0):
        source = data["text"]
        roles = {"human": conv0.roles[0], "gpt": conv0.roles[1]}
        conversations = []
        if roles[source[0]["from"]] != conv0.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]
        conv0.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv0.roles[j % 2], f"{i}"
            conv0.append_message(role, sentence["value"])
        conversations.append(conv0.get_prompt())
        data["conversations"] = conversations
        return data

    def process_text(self, data):
        conversation0 = self.conversation_template.copy()
        data = self.preprocess_text(data, conversation0)
        data = self.form_conversation(data, conversation0)
        return data

    def __getitem__(self, idx):
        idx = random.randint(0, len(self.dataset) - 1)
        item = self.dataset[idx]
        data = {
            "filepath_image": os.path.join(self.cfg.dirpath_images, item["image"]),
            "text0": item["conversations"]
        }
        data = self.process_image(data)
        data = self.process_text(data)
        return data


def test():
    config = {
        "filepath_labels": "/mnt/12T/02_duong/LMM-Research-Foundation/tmp/llava_instruct_150k.json",
        "dirpath_images": "/mnt/12T/02_duong/data-center/coco/2017/train2017",

        "pixel_mean": [123.675, 116.28, 103.53],
        "pixel_std": [58.395, 57.12, 57.375],
        "visual_model_input_size": 1024,
        "hf_visual_model_id": "openai/clip-vit-large-patch14",
        "ignore_label": 255,
        
        "conversation_template": "default",
        
        # training
    }
    dataset = LISAVisualQuestionAnsweringCOCODataset(config)
    data = dataset[0]
    print(data.keys())
    # print(data)


if __name__ == "__main__":
    test()
