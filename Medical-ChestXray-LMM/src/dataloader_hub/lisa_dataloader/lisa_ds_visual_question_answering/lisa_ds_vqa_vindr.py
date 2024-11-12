import sys
sys.path.append(".")

import os
import json
import glob

import numpy as np
import cv2
import torch

from dataloader_hub.dataloader_base.dataset_base_generic import GenericDatasetBase
from dataloader_hub.dataloader_common.preprocess_image.resize_longest_side import ResizeLongestSide
from dataloader_hub.dataloader_common.preprocess_image.normalize_image import normalize_image
from dataloader_hub.lisa_dataloader.lisa_ds_config import *
from model_hub.llava.llava_0.conversation import conv_templates


class LISAVisualQuestionAnsweringVinDrDataset(GenericDatasetBase):
    def __init__(
        self, 
        config,
        dataset_split,
        visual_encoder,
    ):
        super().__init__(config, dataset_split, visual_encoder)

    def initialize_dataset_config(self, config, dataset_split, visual_encoder):
        self.cfg_dataset = config.dataset
        self.visual_encoder = visual_encoder
        self.image_transformer = ResizeLongestSide(config.model.tokenizer.input_size)
        self.pixel_mean = torch.Tensor(self.cfg_dataset.pixel_mean).view(-1, 1, 1)
        self.pixel_std = torch.Tensor(self.cfg_dataset.pixel_std).view(-1, 1, 1)
        self.conversation_template = conv_templates[self.cfg_dataset.conversation_template]
        self.dataset_split = dataset_split

    def load_dataset(self):
        filepath_labels = glob.glob(
            os.path.join(
                self.cfg_dataset.dirpath_labels, 'annotation_final', self.dataset_split, "*.json"
            )
        )

        vindr_data = []
        for filepath_sample in filepath_labels:
            json_id = filepath_sample.split('/')[-1].replace('.json','')
            image_id = json_id.split('_')[-1]
            if '-checkpoint' in image_id:
                image_id = image_id.replace('-checkpoint','')

            image_path = os.path.join(self.cfg_dataset.dirpath_images, image_id + '.jpg')
            mask_path = os.path.join(self.cfg_dataset.dirpath_labels, "mask_final", self.dataset_split, json_id + '.npz')
            # assert os.path.exists(image_path), f"Image path {image_path} does not exist"
            # assert os.path.exists(mask_path), f"Mask path {mask_path} does not exist"
            if os.path.exists(image_path) and os.path.exists(mask_path):
                pass
            else:
                continue
            vindr_data.append(
                {'annotation_path': filepath_sample, 
                 'image_path':image_path, 
                 'mask_path': mask_path}
            )

        return vindr_data
    
    def preprocess_image(self, data):
        image0 = cv2.imread(data["filepath_image"])
        image0 = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)
        image0 = np.stack([image0] * 3, axis=-1)
        image0_size = image0.shape[:2]
        image_resized = self.image_transformer.apply_image(image0)
        image_resized_size = image_resized.shape[:2]
        image = normalize_image(torch.from_numpy(image_resized).permute(2, 0, 1).contiguous(), self.pixel_mean, self.pixel_std, self.cfg_dataset.visual_model_input_size)
        
        masks0 = np.load(data["filepath_mask"])['binary_mask']
        masks = np.stack([masks0], axis= 0)
        masks = torch.from_numpy(masks)

        label = torch.ones(masks.shape[1], masks.shape[2]) * self.cfg_dataset.ignore_label

        data["image0"] = image0
        data["image0_size"] = image0_size
        data["image_resized"] = image_resized
        data["image_resized_size"] = image_resized_size
        data["image"] = image
        data["masks"] = masks
        data["label"] = label
        return data

    def extract_image_embeddings(self, data):
        data["image_for_llm"] = self.visual_encoder.preprocess(data["image0"], return_tensors="pt")["pixel_values"][0]
        return data

    def process_image(self, data):
        data = self.preprocess_image(data)
        data = self.extract_image_embeddings(data)
        return data

    def form_conversation(self, data):
        with open(data["filepath_conv_label"]) as f:
            annotation = json.load(f)
        question, answer = annotation['text'], annotation['answer']        
        answer = answer.replace('<seg>','[SEG]')
        questions = [DEFAULT_IMAGE_TOKEN + "\n" + question + '  Please first respond with segmentation mask, then provide the answer to the question.']
        answers = [answer]

        conversations = []
        conv = self.conversation_template.copy()
        conv.messages = []
        conv.append_message(conv.roles[0], questions[0])
        conv.append_message(conv.roles[1], answers[0])
        conversations.append(conv.get_prompt())
        data["questions"] = questions
        data["conversations"] = conversations
        return data

    def process_text(self, data):
        data = self.form_conversation(data)
        return data

    def __getitem__(self, idx):
        item = self.dataset[idx]
        data = {
            "filepath_image": item["image_path"],
            "filepath_conv_label": item['annotation_path'],
            "filepath_mask": item['mask_path'],
        }
        data = self.process_image(data)
        data = self.process_text(data)

        img_path = data["filepath_image"]
        image = data["image"]
        image_clip = data["image_for_llm"]
        conversations = data["conversations"]
        masks = data["masks"]
        label = data["label"]
        resize = data["image_resized_size"]
        questions = data["questions"]
        inference = True if self.dataset_split == "test" else False

        return (
            img_path,
            image,
            image_clip,
            conversations,
            masks,
            label,
            resize,
            questions,
            None,
            inference,
        )


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
    dataset = LISAVisualQuestionAnsweringVinDrDataset(config)
    data = dataset[0]
    print(data.keys())
    # print(data)


if __name__ == "__main__":
    test()
