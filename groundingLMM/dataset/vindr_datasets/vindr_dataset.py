import glob
import json
import os
import random
import sys
from tqdm import tqdm
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor
import transformers
# Add the LisaMed directory to the Python path

import os
import cv2
import random
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask
import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))
# sys.path.insert(0, '..')
# sys.path.append('../../..') #assure that src directory is in sys.path
sys.path.append('/home/user01/aiotlab/dung_paper/groundingLMM/')
print(sys.path)
from transformers import CLIPImageProcessor
from model.llava import conversation as conversation_lib
from model.SAM.utils.transforms import ResizeLongestSide
from tools.utils import DEFAULT_IMAGE_TOKEN
from dataset.utils.utils import ANSWER_LIST, SEG_QUESTIONS


class VinDrDataset(torch.utils.data.Dataset):
    CLASSES = ('object',)
    IMG_MEAN = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    IMG_STD = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    IMG_SIZE = 1024
    IGNORE_LABEL = 255
    
    def __init__(self, base_dir, tokenizer, global_image_encoder, epoch_samples=500 * 8 * 2 * 10,
                 precision: str = "fp32", image_size: int = 224,  split='train',
                 random_sampling=True, inference=False):
        self.epoch_samples = epoch_samples

        self.base_dir = base_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.global_enc_processor = CLIPImageProcessor.from_pretrained(global_image_encoder)

        self.question_templates = SEG_QUESTIONS
        self.answer_list = ANSWER_LIST
        self.begin_str = f"""The {DEFAULT_IMAGE_TOKEN} provides an overview of the picture.\n"""
        self.split = split
        self.random_sampling = random_sampling
        
        
        jsons = []
        # for split in splits:
            # print('vin_dr_splits: ', split)
        jsons_split = glob.glob(
            os.path.join(
                base_dir, 'annotation_final', self.split, "*.json"
            )
        )
        jsons.extend(jsons_split)
        # print(jsons)
        self.vindr_data = []
        for json_path in jsons:
            json_id = json_path.split('/')[-1].replace('.json','')
            image_id = json_id.split('_')[-1]
            if '-checkpoint' in image_id:
                image_id = image_id.replace('-checkpoint','')

            image_path = os.path.join(self.base_dir, "image/jpg/all", image_id + '.jpg')
            
            mask_path = os.path.join(self.base_dir, "mask_final", split, json_id + '.npy')
            # if not os.path.exists(mask_path):
            #     # print(mask_path)
            self.vindr_data.append({'annotation_path':json_path, 'image_path':image_path, 'mask_path': mask_path})

    def __len__(self):
        return self.epoch_samples

    def _set_len(self, length):
        self.epoch_samples = length

    def grounding_enc_processor(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.IMG_MEAN) / self.IMG_STD
        h, w = x.shape[-2:]
        x = F.pad(x, (0, self.IMG_SIZE - w, 0, self.IMG_SIZE - h))
        return x
    
    def __getitem__(self,idx):
        # print('current idx: ', idx)
        sample = self.vindr_data[idx]
        img_path, json_path , mask_path = sample['image_path'], sample['annotation_path'], sample['mask_path']
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        ori_image = image
        image = np.stack([image] * 3, axis=-1)

        global_enc_img = self.global_enc_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        image = self.transform.apply_image(image)
        image_resize = image.shape[:2]
        grounding_enc_img = self.grounding_enc_processor(torch.from_numpy(image).permute(2, 0, 1).contiguous())
        
                # print('image', image.shape)
        # print('image_clip', image_clip.shape)
        # print('json_path: ', json_path)
        with open(json_path) as f:
            annotation = json.load(f)
        # print(annotation)
        question, answer = annotation['text'], annotation['answer']        
        answer = answer.replace('<seg>','[SEG]')
        questions = [DEFAULT_IMAGE_TOKEN + "\n" + question + '  Please first respond with segmentation mask, then provide the answer to the question.']
        answers = [answer]
        # print(question, answer)
        
        mask = np.load(mask_path)

        image = self.transform.apply_image(image)  # preprocess image for sam
        # print('2nd image', image.shape)
        resize = image.shape[:2]
        
        conversations = []
        conv = conversation_lib.default_conversation_medical.copy()
        conv.messages = []
        conv.append_message(conv.roles[0], questions[0])
        conv.append_message(conv.roles[1], answers[0])
        conversations.append(conv.get_prompt())
        # print('Proconv.get_prompt())
        # print(conversations)

        # image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
        # print(image.shape)
        masks = np.stack([mask], axis= 0)
        masks = torch.from_numpy(masks)

        # print(masks.shape)
        label = torch.ones(masks.shape[1], masks.shape[2]) * self.IGNORE_LABEL
        conversations = conversations 
        
        
        anatomies = ['Aorta', 'Heart', 'Right upper lung', 'Left upper lung', 'Right middle lung', 'Left middle lung', 'Right lower lung', 
'Left lower lung']
        anat = ''
        for anatomy in anatomies:
            if anatomy in questions[0]:
                anat=  anatomy
        labels = [anat]
        if self.split == 'train':
            inference = False
        else:
            inference = True
        bboxes=  None
        return (
            img_path,
            global_enc_img,
            grounding_enc_img,
            bboxes,
            conversations,
            masks,
            label,
            image_resize,
            questions,
            labels,
            
        )

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    output_dir = 'viz'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    IGNORE_INDEX = -100
    IMAGE_TOKEN_INDEX = -200
    DEFAULT_IMAGE_TOKEN = "<image>"
    DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
    DEFAULT_IM_START_TOKEN = "<im_start>"
    DEFAULT_IM_END_TOKEN = "<im_end>"
    IMAGE_PLACEHOLDER = "<image-placeholder>"
    tokenizer = transformers.AutoTokenizer.from_pretrained(
    "microsoft/llava-med-v1.5-mistral-7b",
        cache_dir=None,
        model_max_length=2048,
        padding_side="right",
        use_fast=False,)
    tokenizer.pad_token = tokenizer.unk_token
    num_added_tokens = tokenizer.add_tokens("[SEG]")
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
            "microsoft/llava-med-v1.5-mistral-7b", model_max_length=1024, padding_side="right", use_fast=False
    )
    tokenizer.pad_token = tokenizer.unk_token

    tokenizer.add_tokens(
        [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
    )
    # modifications specific for regions
    reg_tokens = ['<bbox>', '<point>']
    # Adding special tokens for pixel grounding
    segmentation_tokens = ['[SEG]']
    # Adding tokens for GCG
    phrase_tokens = ['<p>', '</p>']
    special_tokens = reg_tokens + segmentation_tokens + phrase_tokens
    tokenizer.add_tokens(special_tokens, special_tokens=True)
    
    dataset = VinDrDataset(base_dir='./LISAMed/dataset/VinDr',tokenizer=tokenizer,global_image_encoder="openai/clip-vit-large-patch14", split = 'train')

    alpha = 0.5
    i = 0
    for data in tqdm(dataset):
        image,gr_image, conversations, masks, label, image_resize, questions, labels = data[1], data[2], data[4], data[5], data[6], data[7], data[8], data[9]
        print('image: ',image.shape, gr_image.shape)
        print('conversations: ',conversations)
        print('masks: ',masks, masks.shape)
        print('labels: ',label, label.shape)
        print('image_resize: ',image_resize)
        print('questions: ',questions)
        print('labels: ',labels)


